from __future__ import annotations
from typing import Dict, Any, Callable, List, Tuple
import logging

import torch
from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.stats import qmc

_log = logging.getLogger(__name__)

class BotorchOptimizer:
    """Bayesian Optimization using BoTorch for black-box flowsheet parameter tuning.
    
    Uses a SingleTaskGP surrogate model with Expected Improvement (EI) to suggest 
    the next best continuous evaluation point.
    """
    
    def __init__(self, maximize: bool = True):
        self.maximize = maximize
        self.device = torch.device("cpu")
        self.dtype = torch.double

    def optimize(
        self, 
        objective_fn: Callable[[torch.Tensor], float], 
        bounds: torch.Tensor, 
        n_initial: int = 5,
        n_iters: int = 15
    ) -> Tuple[torch.Tensor, float, List[Dict[str, Any]]]:
        """Run the Bayesian Optimization loop.
        
        Args:
            objective_fn: Callable taking a 1D tensor of parameters scaled [0, 1] 
                          and returning the true objective value (unscaled score).
            bounds: A `2 x d` tensor of physical bounds [min, max] for each parameter.
            n_initial: Number of initial LHS random points to evaluate.
            n_iters: Number of sequential acquisition function evaluations.
            
        Returns:
            (Best parameters in physical space, Best objective value, History log)
        """
        d = bounds.shape[1]
        
        # 1. Generate initial points using Latin Hypercube Sampling in [0, 1]^d
        sampler = qmc.LatinHypercube(d=d)
        train_x = torch.tensor(sampler.random(n=n_initial), dtype=self.dtype, device=self.device)
        
        # Evaluate initial points (objective_fn handles unscaling)
        train_y = []
        for x in train_x:
            y = objective_fn(x)
            train_y.append(y)
        
        train_y = torch.tensor(train_y, dtype=self.dtype, device=self.device).unsqueeze(-1)
        
        history = [{"iter": 0, "best_x": None, "best_y": train_y.max().item() if self.maximize else train_y.min().item()}]

        for i in range(n_iters):
            # 2. Standardize targets for GP fitting
            y_mean = train_y.mean()
            y_std = train_y.std()
            y_std = y_std if y_std > 1e-6 else 1.0
            train_y_norm = (train_y - y_mean) / y_std
            
            # If minimizing, we flip the normalized targets so EI always maximizes
            if not self.maximize:
                train_y_norm = -train_y_norm
            
            # 3. Fit the SingleTaskGP surrogate model
            model = SingleTaskGP(train_x, train_y_norm)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            
            # 4. Construct Expected Improvement acquisition function
            best_f = train_y_norm.max()
            acqf = LogExpectedImprovement(model=model, best_f=best_f)
            
            # 5. Optimize the acquisition function to find next candidate
            candidate, acq_value = optimize_acqf(
                acq_function=acqf,
                bounds=torch.tensor([[0.0] * d, [1.0] * d], dtype=self.dtype, device=self.device),
                q=1,
                num_restarts=5,
                raw_samples=20,
            )
            
            new_x = candidate.detach()
            
            # 6. Evaluate true objective function
            new_y_val = objective_fn(new_x.squeeze(0))
            new_y = torch.tensor([[new_y_val]], dtype=self.dtype, device=self.device)
            
            # Update datasets
            train_x = torch.cat([train_x, new_x])
            train_y = torch.cat([train_y, new_y])
            
            # Log best
            current_best = train_y.max().item() if self.maximize else train_y.min().item()
            history.append({
                "iter": i + 1,
                "best_y": current_best,
                "new_x": new_x.squeeze(0).tolist(),
                "new_y": new_y_val
            })
            _log.info(f"BO Iter {i+1}/{n_iters} | New Y: {new_y_val:.4f} | Best: {current_best:.4f}")

        # Final extraction
        best_idx = train_y.argmax() if self.maximize else train_y.argmin()
        best_x_norm = train_x[best_idx]
        best_y = train_y[best_idx].item()
        
        # Unscale best_x back to physical bounds
        best_x_phys = bounds[0] + best_x_norm * (bounds[1] - bounds[0])
        
        return best_x_phys, best_y, history
