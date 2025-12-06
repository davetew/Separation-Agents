"""
Orchestrator Agent
==================

This agent is the main entry point. It receives a user goal and delegates tasks to the specialized agents.
"""

from sep_agents.sim.reactor_design_agent import ReactorDesignAgent
from sep_agents.sim.kinetics_agent import KineticsAgent
from sep_agents.cost.economics_agent import EconomicsAgent
from typing import Dict, Any, List

class OrchestratorAgent:
    """
    Orchestrator for managing the engineering design workflow.
    """

    def __init__(self):
        """
        Initialize the OrchestratorAgent.
        """
        self.reactor_agent = ReactorDesignAgent()
        self.kinetics_agent = None
        self.economics_agent = EconomicsAgent()

    def design_process(self, 
                       initial_conditions: Dict[str, Any], 
                       simulation_params: Dict[str, Any],
                       economic_params: Dict[str, Any] = None,
                       reactor_params: Dict[str, Any] = None,
                       primary_product: str = "H2") -> Dict[str, Any]:
        """
        Execute the full design process: Define -> Simulate -> Evaluate.

        Args:
            initial_conditions (Dict[str, Any]): Parameters for `ReactorDesignAgent.define_state`.
            simulation_params (Dict[str, Any]): Parameters for `KineticsAgent.run_simulation`.
            economic_params (Dict[str, Any]): Parameters for `EconomicsAgent`.
            reactor_params (Dict[str, Any]): Parameters for `KineticsAgent` (e.g., constant_volume_specs).
            primary_product (str): The primary product for economic assessment.

        Returns:
            Dict[str, Any]: Summary of the design results.
        """
        
        # 1. Define the System & Initial State
        print("--- Step 1: Defining System & Initial State ---")
        state = self.reactor_agent.define_state(**initial_conditions)
        
        # 2. Run Kinetic Simulation
        print("--- Step 2: Running Kinetic Simulation ---")
        constraint = simulation_params.pop("constraint", "TP") # Default to TP to avoid volume issues
        # Pass initial_conditions dict so Kinetics class creates the system with reaction rates
        self.kinetics_agent = KineticsAgent(state=initial_conditions, constraint=constraint, **(reactor_params if reactor_params else {}))
        sim_results = self.kinetics_agent.run_simulation(**simulation_params)
        
        # 3. Economic Assessment
        print("--- Step 3: Economic Assessment ---")
        # Re-initialize EconomicsAgent with primary_product if needed, or just update params
        # Since primary_product is an init arg, we might need to re-init if it changes
        self.economics_agent = EconomicsAgent(cost_params=self.economics_agent.cost_params, primary_product=primary_product)
        
        if economic_params:
            self.economics_agent.update_params(economic_params)
        
        # Note: In a real scenario, we would feed simulation results (yields, rates) into the economic model.
        # For now, we'll just run the default economic assessment as a placeholder or use the updated params.
        lcop = self.economics_agent.calculate_lcop()
        
        return {
            "simulation_results": sim_results.tail(1).to_dict(orient='records')[0],
            "lcop": lcop,
            "cost_breakdown": self.economics_agent.get_cost_breakdown().to_dict()
        }

    def optimize_process(self, 
                         base_initial_conditions: Dict[str, Any], 
                         base_simulation_params: Dict[str, Any],
                         optimization_variables: List[Dict[str, Any]],
                         economic_params: Dict[str, Any] = None,
                         reactor_params: Dict[str, Any] = None,
                         primary_product: str = "H2",
                         method: str = "L-BFGS-B",
                         max_iter: int = 50) -> Dict[str, Any]:
        """
        Optimize the process parameters to minimize LCOP.

        Args:
            base_initial_conditions (Dict[str, Any]): Baseline initial conditions.
            base_simulation_params (Dict[str, Any]): Baseline simulation parameters.
            optimization_variables (List[Dict[str, Any]]): List of variables to optimize.
                Each dict should have:
                - "name": str
                - "path": List[str] (e.g., ["initial_conditions", "T_C"])
                - "bounds": Tuple[float, float]
                - "initial_value": float (optional, defaults to value in base dict)
            economic_params (Dict[str, Any]): Economic parameters.
            reactor_params (Dict[str, Any]): Reactor parameters.
            primary_product (str): Primary product.
            method (str): Optimization method.
            max_iter (int): Maximum iterations.

        Returns:
            Dict[str, Any]: Optimization results.
        """
        from sep_agents.opt.optimization_agent import OptimizationAgent
        
        opt_agent = OptimizationAgent(method=method)
        
        # Prepare initial guess and bounds
        x0 = []
        bounds = []
        
        for var in optimization_variables:
            path = var["path"]
            # Extract initial value from base dicts if not provided
            if "initial_value" in var:
                val = var["initial_value"]
            else:
                if path[0] == "initial_conditions":
                    val = base_initial_conditions.get(path[1])
                elif path[0] == "simulation_params":
                    val = base_simulation_params.get(path[1])
                else:
                    raise ValueError(f"Unknown path root: {path[0]}")
            
            if val is None:
                raise ValueError(f"Could not find initial value for {var['name']}")
                
            x0.append(val)
            bounds.append(var["bounds"])

        def objective_function(x):
            # Construct current parameters
            current_initial_conditions = base_initial_conditions.copy()
            current_simulation_params = base_simulation_params.copy()
            
            print(f"Evaluating: {x}")
            
            for i, var in enumerate(optimization_variables):
                path = var["path"]
                val = x[i]
                
                if path[0] == "initial_conditions":
                    current_initial_conditions[path[1]] = val
                elif path[0] == "simulation_params":
                    current_simulation_params[path[1]] = val
            
            try:
                # Run design process
                # Suppress print output from design_process to avoid clutter? 
                # For now, let it print so we see progress.
                result = self.design_process(
                    initial_conditions=current_initial_conditions,
                    simulation_params=current_simulation_params,
                    economic_params=economic_params,
                    reactor_params=reactor_params,
                    primary_product=primary_product
                )
                lcop = result["lcop"]
                print(f"  -> LCOP: {lcop:.4f} USD/kg")
                return lcop
            except Exception as e:
                print(f"  -> Simulation failed: {e}")
                return 1e9 # High penalty

        # Run optimization
        result = opt_agent.optimize(
            objective_function=objective_function,
            initial_guess=x0,
            bounds=bounds,
            options={"maxiter": max_iter}
        )
        
        return result
