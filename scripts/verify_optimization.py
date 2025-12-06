import sys
import os
from unittest.mock import MagicMock

# Mock reaktoro and GeoH2 before importing agents
sys.modules["reaktoro"] = MagicMock()
sys.modules["GeoH2"] = MagicMock()
sys.modules["GeoH2.equilibrium"] = MagicMock()
sys.modules["GeoH2.equilibrium"].defineSystem.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
sys.modules["GeoH2.kinetics"] = MagicMock()
sys.modules["GeoH2.economics"] = MagicMock()

# Add the source directories to the path
sys.path.append("/Users/davidtew/Library/CloudStorage/GoogleDrive-davetew@step-function.com/My Drive/Github/Separation-Agents/src")

# Now import OrchestratorAgent
from sep_agents.orchestrator.orchestrator_agent import OrchestratorAgent

def verify_optimization():
    print("Initializing Orchestrator...")
    orchestrator = OrchestratorAgent()

    # Mock the design_process method to avoid running real simulations
    # We'll define a simple objective function: LCOP = (T - 80)^2 + 10
    # Minimum should be at T = 80
    def mock_design_process(initial_conditions, simulation_params, **kwargs):
        T = initial_conditions.get("T_C", 60)
        lcop = (T - 80)**2 + 10
        print(f"    [Mock Simulation] T={T:.2f} -> LCOP={lcop:.2f}")
        return {
            "lcop": lcop,
            "simulation_results": {},
            "cost_breakdown": {}
        }

    orchestrator.design_process = mock_design_process

    # Base conditions
    initial_conditions = {
        "T_C": 60,
        "p_bar": 1,
        "mineral_spec": {"Fayalite": 0.1, "Forsterite": 0.9},
        "w_r": 1.0,
        "c_r": 1e-6,
        "salinity_g_kg": 0.1
    }

    simulation_params = {
        "duration": "10 minutes",
        "n_points": 5,
        "constraint": "TV"
    }

    # Optimization variables
    opt_vars = [
        {
            "name": "Temperature",
            "path": ["initial_conditions", "T_C"],
            "bounds": (50.0, 100.0),
            "initial_value": 60.0
        }
    ]

    print("Starting Optimization...")
    try:
        result = orchestrator.optimize_process(
            base_initial_conditions=initial_conditions,
            base_simulation_params=simulation_params,
            optimization_variables=opt_vars,
            economic_params={},
            reactor_params={},
            primary_product="Mg",
            max_iter=20
        )

        print("\nOptimization Result:")
        print(result)
        
        # Check if we got close to 80
        opt_T = result["optimized_params"][0]
        if abs(opt_T - 80) < 1.0:
            print("\nSUCCESS: Optimization converged to expected value (approx 80).")
        else:
            print(f"\nFAILURE: Optimization did not converge to expected value. Got {opt_T}")

    except Exception as e:
        print(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_optimization()
