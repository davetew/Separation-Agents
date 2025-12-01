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
