
import sys
import os

# Add the source directories to the path
sys.path.append("/Users/davidtew/Library/CloudStorage/GoogleDrive-davetew@step-function.com/My Drive/Github/Separation-Agents/src")
sys.path.append("/Users/davidtew/Library/CloudStorage/GoogleDrive-davetew@step-function.com/My Drive/Github/GeoH2")

from sep_agents.orchestrator.orchestrator_agent import OrchestratorAgent
from GeoH2 import Q_

def test_agents():
    print("Initializing Orchestrator...")
    orchestrator = OrchestratorAgent()

    # Define parameters for the sample problem (Mg/Fe extraction from slag)
    # Note: These are placeholder values for the test
    initial_conditions = {
        "T_C": 60,
        "p_bar": 1,
        "mineral_spec": {"Fayalite": 0.1, "Forsterite": 0.9}, # Simulating slag with Olivine for now
        "w_r": 1.0,
        "c_r": 1e-6,
        "salinity_g_kg": 0.1
    }

    simulation_params = {
        "duration": "1 hour",
        "n_points": 10,
        "constraint": "TV" # Use Constant Volume to trigger fill_reactor
    }

    economic_params = {
        "mat_yield": {'H2': Q_(0, "kg/tonne"), 'Mg': Q_(100, "kg/tonne")}, # Zero H2, some Mg
        "mat_value": {'Mg': Q_(5, "USD/kg")}, # Value for Mg
        "simulation": {"t_process": Q_(1, "hour")},
        "M_target_tonnes": 5000 # Target 5000 tonnes of Mg
    }

    reactor_params = {
        "constant_volume_specs": {
            "volume": Q_(2000, "mL"),
            "fill_gas": {'N2(g)': 0.79, 'O2(g)': 0.21},
            "fill_temperature": Q_(25, "degC"),
            "fill_pressure": Q_(50, "bar"), # High pressure
            "mineralMass": None
        }
    }

    print("Running Design Process...")
    try:
        results = orchestrator.design_process(
            initial_conditions=initial_conditions,
            simulation_params=simulation_params,
            economic_params=economic_params,
            reactor_params=reactor_params,
            primary_product="Mg"
        )
        print("\n--- Design Process Successful! ---")
        print("Results Summary:")
        print(results)
    except Exception as e:
        print(f"\n--- Design Process Failed! ---")
        print(e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agents()
