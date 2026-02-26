import torch
import math
from sep_agents.opt.bo import BotorchOptimizer

def dummy_objective(x_phys: torch.Tensor) -> float:
    # A simple 2D objective function to maximize: negated distance from (0.5, 0.5)
    # Plus a small cosine wave
    x1, x2 = x_phys[0].item(), x_phys[1].item()
    dist = (x1 - 0.5)**2 + (x2 - 0.5)**2
    score = -dist + 0.1 * math.cos(10 * x1)
    return score

def main():
    print("Testing BotorchOptimizer on a Dummy 2D Function...")
    opt = BotorchOptimizer(maximize=True)
    
    # 2D search space [ [min], [max] ]
    bounds = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0]
    ], dtype=torch.double)
    
    print("Optimization starting (5 initial, 10 sequential)...")
    
    # We must wrap the unscaling since the optimizer passes inputs [0, 1]
    def scaled_objective(x_norm: torch.Tensor) -> float:
        x_phys = bounds[0] + x_norm * (bounds[1] - bounds[0])
        return dummy_objective(x_phys)
        
    best_x, best_y, history = opt.optimize(
        objective_fn=scaled_objective,
        bounds=bounds,
        n_initial=5,
        n_iters=10
    )
    
    print(f"\nOptimization Finished!")
    print(f"Target Global Max: ~ (0.5, 0.5) with score ~ 0.028")
    print(f"Found Maximum: {best_x.tolist()} | Score: {best_y}")

if __name__ == "__main__":
    main()
