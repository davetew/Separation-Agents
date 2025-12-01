
import sys
import os
import numpy as np
import pandas as pd
from GeoH2 import Q_

# Add the source directories to the path
sys.path.append("/Users/davidtew/Library/CloudStorage/GoogleDrive-davetew@step-function.com/My Drive/Github/Separation-Agents/src")
sys.path.append("/Users/davidtew/Library/CloudStorage/GoogleDrive-davetew@step-function.com/My Drive/Github/GeoH2")

from sep_agents.sim.equilibrium_agent import EquilibriumAgent

def test_equilibrium_agent():
    print("Initializing EquilibriumAgent...")
    agent = EquilibriumAgent()

    # Define parameters for the sample problem (Mg/Fe extraction from slag)
    initial_conditions = {
        "T_C": 60,
        "p_bar": 1,
        "mineral_spec": {"Fayalite": 0.1, "Forsterite": 0.9},
        "w_r": 1.0,
        "c_r": 1e-6,
        "salinity_g_kg": 0.1
    }

    print("\n--- Test 1: Single Point Equilibrium ---")
    try:
        state = agent.define_state(**initial_conditions)
        eq_state = agent.solve(state)
        print("Equilibrium calculated successfully.")
        
        # Basic check
        import reaktoro as rkt
        aprops = rkt.AqueousProps(eq_state)
        print(f"pH: {aprops.pH():.2f}")
        
    except Exception as e:
        print(f"Single point test failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n--- Test 2: Parameter Sweep (Temperature) ---")
    try:
        # Sweep Temperature from 25 to 100 C
        temps = np.linspace(25, 100, 5)
        results = agent.sweep(initial_conditions, "T_C", temps)
        
        print("Sweep Results:")
        print(results[["T_C", "pH", "Eh (V)"]])
        
        analysis = agent.analyze_results(results)
        print("\nAnalysis:")
        print(analysis)
        
    except Exception as e:
        print(f"Sweep test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_equilibrium_agent()
