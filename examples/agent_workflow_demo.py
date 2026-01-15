
import sys
import os
import json
from pathlib import Path
from unittest.mock import MagicMock

# -- Setup Environment --
# Add src to path so we can import our modules
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))
geoh2_path = Path(__file__).parent.parent.parent / "GeoH2"
sys.path.append(str(geoh2_path))

# -- Mocking dependencies if missing (for demonstration purposes) --
try:
    import reaktoro
except ImportError:
    print("(!) Reaktoro not found. Using Mocks for demonstration.")
    sys.modules["reaktoro"] = MagicMock()
    sys.modules["GeoH2"] = MagicMock()
    sys.modules["GeoH2.equilibrium"] = MagicMock()
    
    # defineSystem returns 4 values: system, minerals, solution, gases
    sys.modules["GeoH2.equilibrium"].defineSystem.return_value = (MagicMock(), MagicMock(), MagicMock(), MagicMock())
    
    # Mock sweep in EquilibriumAgent to return a DataFrame mock
    # Note: Since the import runs before we can patch the class method easily without mocking the whole module
    # We will rely on mocking the behavior of the internal call
    
    # Mock return values for smoother demo
    mock_df_obj = MagicMock()
    mock_df_obj.to_dict.return_value = [
        {"pH": 3.0, "Enthalpy (J)": -5000},
        {"pH": 4.0, "Enthalpy (J)": -4800},
    ]
    
    # Mock EquilibriumAgent class
    # We need to patch the class in the module we are about to import
    # But since we haven't imported it yet, we can't easily patch it this way for the tool import.
    # We will let the tool import fail or use the mock modules we just set in sys.modules

# -- Import Tools --
# We import the tools directly from the server file to simulate calling them
# Note: In a real agent workflow, these would be called via MCP protocol
sys.path.append(str(Path(__file__).parent.parent / "mcp_server"))
from server import run_speciation, perform_sweep

# Patching internal logic if using mocks
if 'reaktoro' in sys.modules and isinstance(sys.modules['reaktoro'], MagicMock):
    from sep_agents.sim.equilibrium_agent import EquilibriumAgent
    from sep_agents.sim import reaktoro_adapter
    from sep_agents.dsl.schemas import Stream
    
    # 1. Patch sweep to return our mock dataframe
    EquilibriumAgent.sweep = MagicMock(return_value=mock_df_obj)
    
    # 2. Patch run_reaktoro logic or just let it mock through?
    # run_reaktoro calls agent.solve(state). AqueousProps(eq_state).pH()
    # So we need to ensure those chain calls work.
    
    # Mock return of define_state
    EquilibriumAgent.define_state = MagicMock()
    # Mock return of solve
    EquilibriumAgent.solve = MagicMock()
    
    # Mock AqueousProps to return pH/Eh
    mock_rkt = sys.modules["reaktoro"]
    mock_aprops = MagicMock()
    mock_aprops.pH.return_value = 7.5
    mock_aprops.Eh.return_value = 0.5
    mock_rkt.AqueousProps.return_value = mock_aprops

def print_step(title, description, payload):
    print(f"\n{'='*60}")
    print(f"STEP: {title}")
    print(f"DESC: {description}")
    print(f"INPUT Paylod:\n{json.dumps(payload, indent=2)}")
    print(f"{'-'*60}")

def main():
    print("Starting Agentic Workflow Demo...")

    # --- Scenario 1: Feasibility Study ---
    input_sweep = {
        "initial_conditions": {
            "T_C": 25.0,
            "p_bar": 1.0,
            "mineral_spec": {"Calcite": 1.0, "H2O": 55.0},
            "mineral_spec_type": "mol"
        },
        "param_name": "T_C",
        "values": [25.0, 50.0, 75.0]
    }
    
    print_step("Feasibility Sweep", 
               "Agent evaluates effect of Temperature on Calcite solubility.", 
               input_sweep)
    
    # Needs to mock the agent inside the tool if using mocks
    if 'reaktoro' in sys.modules and isinstance(sys.modules['reaktoro'], MagicMock):
         # We need to ensure equilibrium_agent.EquilibriumAgent returns our mock
         # This is tricky without unittest.patch, but let's see if the sys.modules trick held enough
         pass

    result_sweep = perform_sweep(
        input_sweep["initial_conditions"],
        input_sweep["param_name"],
        input_sweep["values"]
    )
    print("OUTPUT:")
    print(json.dumps(result_sweep, indent=2))


    # --- Scenario 2: Speciation Check ---
    input_stream = {
        "stream": {
            "name": "feed_slurry",
            "phase": "liquid",
            "temperature_K": 298.15,
            "pressure_Pa": 101325,
            "composition_wt": {"H2O": 90.0, "CaCO3": 10.0}
        }
    }
    
    print_step("Speciation Check", 
               "Agent checks the baseline equilibrium pH of the feed slurry.", 
               input_stream)

    result_stream = run_speciation(input_stream["stream"])
    print("OUTPUT:")
    print(json.dumps(result_stream, indent=2))

if __name__ == "__main__":
    main()
