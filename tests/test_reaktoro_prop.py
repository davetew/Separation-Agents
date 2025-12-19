
try:
    import pytest
except ImportError:
    pytest = None

import sys
import os

# Identify path to src
# Adjust this depending on where you run pytest from
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from sep_agents.properties.reaktoro import ReaktoroParameterBlock

# Check if Reaktoro is available
try:
    import reaktoro
    REAKTORO_AVAILABLE = True
except ImportError:
    REAKTORO_AVAILABLE = False

def test_reaktoro_properties():
    if not REAKTORO_AVAILABLE:
        print("Reaktoro not installed, skipping test.")
        return
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    
    # Initialize Parameter Block
    # Using 'SUPRCRT - BL' as default
    m.fs.props = ReaktoroParameterBlock(database="SUPRCRT - BL")
    
    # Create State Block
    m.fs.state = m.fs.props.build_state_block(
        m.fs.time, 
        defined_state=True # We will define state manually
    )
    
    # Fix State Variables
    m.fs.state[0].temperature.fix(300)   # K
    m.fs.state[0].pressure.fix(101325)   # Pa
    m.fs.state[0].flow_mol.fix(100)      # mol/s
    
    # Fix Composition (Water)
    # We need to pick a valid species name found in the database
    # Typically 'H2O(aq)' or 'H2O' depending on the DB
    # Let's inspect available species from params
    species = m.fs.props.species_list
    print(f"Species: {species}")
    
    h2o_name = "H2O(aq)" if "H2O(aq)" in species else "H2O"
    
    if h2o_name in species:
        m.fs.state[0].mole_frac_comp[h2o_name].fix(1.0)
        # Fix others to 0
        for s in species:
            if s != h2o_name:
                m.fs.state[0].mole_frac_comp[s].fix(0.0)
    else:
        # Fallback if weird species_list
        pytest.skip(f"Could not find H2O in species list: {species}")

    # Initialize / Calculate Properties
    m.fs.state[0].initialize()
    
    # Check results
    # Enthalpy of Aqueous Phase should be populated
    # m.fs.props.phase_list usually has 'AqueousPhase'
    aq_phase = "AqueousPhase"
    if aq_phase in m.fs.props.phase_list:
        enth = value(m.fs.state[0].enth_mol_phase[aq_phase])
        dens = value(m.fs.state[0].dens_mol_phase[aq_phase])
        mw   = value(m.fs.state[0].molecular_weight[aq_phase])
        
        print(f"Enthalpy (J/mol): {enth}")
        print(f"Density (mol/m3): {dens}")
        print(f"MW (kg/mol): {mw}")

        assert enth != 0.0 # Should be non-zero usually (unless reference state makes it so)
        assert dens > 0
        assert mw > 0

if __name__ == "__main__":
    test_reaktoro_properties()
