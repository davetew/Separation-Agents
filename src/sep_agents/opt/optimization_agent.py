"""
Optimization Agent
==================

This agent is responsible for performing numerical optimization of process parameters.
It wraps `scipy.optimize` functionality to minimize a given objective function.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Callable, List, Dict, Any, Tuple, Union

class OptimizationAgent:
    """
    Agent for performing numerical optimization.
    """

    def __init__(self, method: str = "L-BFGS-B"):
        """
        Initialize the OptimizationAgent.

        Args:
            method (str): Optimization method to use (e.g., 'L-BFGS-B', 'SLSQP', 'Nelder-Mead').
        """
        self.method = method

    def optimize(self, 
                 objective_function: Callable[[List[float]], float], 
                 initial_guess: List[float], 
                 bounds: List[Tuple[float, float]] = None,
                 constraints: List[Dict[str, Any]] = None,
                 options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform optimization to minimize the objective function.

        Args:
            objective_function (Callable): The function to minimize. Takes a list of floats and returns a float.
            initial_guess (List[float]): Initial values for the parameters.
            bounds (List[Tuple[float, float]]): Bounds for each parameter (min, max).
            constraints (List[Dict]): Constraints for the optimization (used by SLSQP, COBYLA).
            options (Dict[str, Any]): Solver options.

        Returns:
            Dict[str, Any]: Optimization results including optimized parameters and minimum value.
        """
        
        print(f"Starting optimization with method {self.method}...")
        print(f"Initial guess: {initial_guess}")
        
        result = minimize(
            objective_function,
            initial_guess,
            method=self.method,
            bounds=bounds,
            constraints=constraints,
            options=options
        )

        print(f"Optimization finished. Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Final value: {result.fun}")
        print(f"Final parameters: {result.x}")

        return {
            "success": result.success,
            "message": result.message,
            "optimized_params": result.x.tolist(),
            "min_value": result.fun,
            "n_iterations": result.nit if hasattr(result, 'nit') else 0,
            "n_evaluations": result.nfev
        }
