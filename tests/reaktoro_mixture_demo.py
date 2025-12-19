
"""
Example: Specifying Multi-Phase Mixtures with Reaktoro Property Package

This script demonstrates how to initializing the IDAES-Reaktoro property package
and specify a mixture containing aqueous, gaseous, and solid species.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pyomo.environ import ConcreteModel, value, units as pyunits
from idaes.core import FlowsheetBlock
from sep_agents.properties.reaktoro import ReaktoroParameterBlock

def run_demo():
    # 1. Create Model & Flowsheet
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)

    # 2. Add Property Package
    # Using 'SUPRCRT - BL' which contains water, gases, and minerals
    print("Initializing Property Package...")
    m.fs.props = ReaktoroParameterBlock(database="SUPRCRT - BL")

    # 3. Create State Block
    m.fs.state = m.fs.props.build_state_block(m.fs.time)
    
    # 4. Define State (T, P, Composition)
    # Target: 
    #   Temperature: 50 C
    #   Pressure: 100 bar
    #   Composition: 
    #       - 1 kg H2O
    #       - 0.1 mol CO2
    #       - 0.5 mol NaCl (Halite)
    #       - 0.1 mol CaCO3 (Calcite)
    
    T_val = 323.15 # K (50 C)
    P_val = 100e5  # Pa (100 bar)
    
    # Define amounts (User Logic)
    amounts = {
        "H2O(aq)": 1000.0 / 18.015, # mol (approx 55.5 mol)
        "CO2(g)": 0.1,              # mol
        "Halite": 0.5,              # mol
        "Calcite": 0.1              # mol
    }
    
    total_moles = sum(amounts.values())
    
    # Set Variables on State Block
    m.fs.state[0].temperature.fix(T_val)
    m.fs.state[0].pressure.fix(P_val)
    m.fs.state[0].flow_mol.fix(total_moles)
    
    # Set Mole Fractions
    # Note: We must initialize ALL species. Ideally fix others to 0.
    # The property package creates a component for EVERY species in the system.
    
    print("\nSetting Composition...")
    for comp_name in m.fs.props.component_list:
        if comp_name in amounts:
            frac = amounts[comp_name] / total_moles
            m.fs.state[0].mole_frac_comp[comp_name].fix(frac)
        else:
            m.fs.state[0].mole_frac_comp[comp_name].fix(0.0)
            
    # 5. Calculate Properties
    print("Calculating Properties (Reaktoro Equilibrium)...")
    m.fs.state[0].calculate_properties()
    
    # 6. Inspect Results
    print("\n--- Results ---")
    print(f"Temperature: {value(m.fs.state[0].temperature)} K")
    print(f"Pressure: {value(m.fs.state[0].pressure)} Pa")
    
    # Check Enthalpy of Phases
    # Note: Reaktoro will distribute species across phases based on equilibrium.
    # e.g. CO2 might dissolve in AqueousPhase, Halite might dissolve.
    
    for p in m.fs.props.phase_list:
        try:
            h = value(m.fs.state[0].enth_mol_phase[p])
            rho = value(m.fs.state[0].dens_mol_phase[p])
            print(f"Phase '{p}': Enthalpy={h:.2f} J/mol, Density={rho:.2f} mol/m3")
        except:
            print(f"Phase '{p}': Not present or properties unavailable.")

    print("\nDemo Completed Successfully.")

if __name__ == "__main__":
    run_demo()
