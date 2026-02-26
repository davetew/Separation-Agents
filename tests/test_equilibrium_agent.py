
import sys
import unittest
import pandas as pd
from pathlib import Path
from unittest.mock import MagicMock, patch

# Mock dependencies if not available
try:
    import reaktoro
    import GeoH2
    import GeoH2.equilibrium
except ImportError:
    sys.modules.setdefault("reaktoro", MagicMock())
    sys.modules.setdefault("GeoH2", MagicMock())
    sys.modules.setdefault("GeoH2.equilibrium", MagicMock())
    sys.modules["GeoH2.equilibrium"].defineSystem = MagicMock(
        return_value=("sys", "min", "sol", "gas")
    )

# Import the class under test
# We need to ensure the path is correct. Assuming we run from repo root via python -m unittest, 
# or if running file directly need adjustments.
# But for simplicity let's assume we run from repo root or adjust path:
sys.path.append(str(Path(__file__).parent.parent / "src"))
from sep_agents.sim.equilibrium_agent import EquilibriumAgent

class TestEquilibriumAgent(unittest.TestCase):
    def setUp(self):
        patcher = patch("sep_agents.sim.equilibrium_agent.defineSystem", return_value=("sys", "min", "sol", "gas"))
        self.mock_define_system = patcher.start()
        self.addCleanup(patcher.stop)

    @patch("sep_agents.sim.equilibrium_agent.defineInitialState")
    def test_define_state_mapping_suprcrt(self, mock_define_state):
        # Instantiate agent with the "typo" database name
        agent = EquilibriumAgent(database_name="SUPRCRT - BL")
        
        # Call define_state with "H2O"
        agent.define_state(
            T_C=25.0,
            p_bar=1.0,
            mineral_spec={"H2O": 55.5}
        )
        
        # Verify defineInitialState was called with REMAPPED keys
        # We expect "H2O" -> "H2O(aq)"
        args, kwargs = mock_define_state.call_args
        mineral_spec_arg = kwargs.get("mineralSpec")
        
        self.assertIn("H2O(aq)", mineral_spec_arg)
        self.assertNotIn("H2O", mineral_spec_arg)
        self.assertEqual(mineral_spec_arg["H2O(aq)"], 55.5)
        print("Verified mapping for SUPRCRT: H2O -> H2O(aq)")

    @patch("sep_agents.sim.equilibrium_agent.defineInitialState")
    def test_define_state_mapping_supcrt(self, mock_define_state):
        # Instantiate agent with the correct database name
        agent = EquilibriumAgent(database_name="supcrt")
        
        # Call define_state with "H2O"
        agent.define_state(
            T_C=25.0,
            p_bar=1.0,
            mineral_spec={"H2O": 55.5}
        )
        
        args, kwargs = mock_define_state.call_args
        mineral_spec_arg = kwargs.get("mineralSpec")
        
        self.assertIn("H2O(aq)", mineral_spec_arg)
        self.assertEqual(mineral_spec_arg["H2O(aq)"], 55.5)
        print("Verified mapping for supcrt: H2O -> H2O(aq)")

    @patch("sep_agents.sim.equilibrium_agent.defineInitialState")
    def test_define_state_no_mapping_phreeqc(self, mock_define_state):
        # Instantiate agent with PHREEQC
        agent = EquilibriumAgent(database_name="PHREEQC")
        
        # Call define_state with "H2O"
        agent.define_state(
            T_C=25.0,
            p_bar=1.0,
            mineral_spec={"H2O": 55.5}
        )
        
        args, kwargs = mock_define_state.call_args
        mineral_spec_arg = kwargs.get("mineralSpec")
        
        # PHREEQC uses H2O, so it should stay H2O (mapped to itself or not mapped)
        # In our mapping: "phreeqc": {"H2O": "H2O"}
        self.assertIn("H2O", mineral_spec_arg)
        self.assertEqual(mineral_spec_arg["H2O"], 55.5)
        print("Verified mapping for PHREEQC: H2O -> H2O")

    @patch("sep_agents.sim.equilibrium_agent.defineInitialState")
    @patch("sep_agents.sim.equilibrium_agent.equilibrium")
    @patch("reaktoro.ChemicalProps")
    @patch("reaktoro.AqueousProps")
    def test_sweep_failure_handling(self, mock_aq_props, mock_chem_props, mock_equilibrium, mock_define_state):
        # Mock dependencies
        agent = EquilibriumAgent(database_name="supcrt")
        
        # Setup mocks
        mock_define_state.return_value = MagicMock()
        
        # First call succeeds
        # Second call fails (returns None, simulating non-convergence which caused the C++ error)
        mock_equilibrium.side_effect = [MagicMock(), None]
        
        # Mock ChemicalProps & AqueousProps to succeed for the first call
        mock_props = MagicMock()
        mock_props.temperature.return_value = 300.0
        mock_chem_props.return_value = mock_props

        mock_aq = MagicMock()
        mock_aq.pH.return_value = 7.0
        mock_aq_props.return_value = mock_aq

        # Run sweep
        df = agent.sweep(
            initial_conditions={"T_C": 25.0, "p_bar": 1.0, "mineral_spec": {"H2O": 55.5}},
            param_name="T_C",
            param_values=[25.0, 50.0]
        )
        
        print("\nSweep Result DataFrame:")
        print(df)
        
        # Verify results
        self.assertEqual(len(df), 2)
        
        # Row 0: Success
        self.assertEqual(df.iloc[0]["status"], "converged")
        self.assertIsNotNone(df.iloc[0]["pH"]) # Or whatever logic
        
        # Row 1: Failure
        self.assertEqual(df.iloc[1]["status"], "failed")
        self.assertIn("Equilibrium solver returned None", df.iloc[1]["error"])
        self.assertTrue(pd.isna(df.iloc[1]["pH"])) # Should be None/NaN
        print("Verified sweep failure handling code path.")

if __name__ == "__main__":
    unittest.main()
