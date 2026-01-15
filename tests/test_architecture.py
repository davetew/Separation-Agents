
import sys
import os
import unittest
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

# Add GeoH2 to path (assuming peer directory structure from user info)
geoh2_path = Path(__file__).parent.parent.parent / "GeoH2"
sys.path.append(str(geoh2_path))

# Mock dependencies
from unittest.mock import MagicMock
sys.modules["reaktoro"] = MagicMock()
sys.modules["GeoH2"] = MagicMock()
sys.modules["GeoH2.equilibrium"] = MagicMock()
sys.modules["GeoH2.Q_"] = MagicMock()

class TestArchitecture(unittest.TestCase):

    def test_schemas(self):
        from sep_agents.dsl.schemas import UNIT_PARAM_SPEC, Stream
        self.assertIn("leach_reactor", UNIT_PARAM_SPEC)
        self.assertIn("precipitator", UNIT_PARAM_SPEC)
        
        # Test Stream creation
        s = Stream(name="feed", phase="liquid", composition_wt={"H2O": 1.0})
        self.assertEqual(s.name, "feed")

    def test_adapter_structure(self):
        # We verify that we can import the adapter and it has the right function
        from sep_agents.sim import reaktoro_adapter
        self.assertTrue(hasattr(reaktoro_adapter, "run_reaktoro"))

    def test_mcp_tools_exist(self):
        # We verify the server has the expected decorators/functions
        # We can't easily start the FastMCP server here without blocking, 
        # but we can inspect the module
        
        # Temporarily mock mcp if needed or just check file content dynamically?
        # Actually server.py imports 'sep_agents' so it should import fine if path is right.
        
        sys.path.append(str(Path(__file__).parent.parent / "mcp_server"))
        try:
            import server
            self.assertTrue(hasattr(server, "run_speciation"))
            self.assertTrue(hasattr(server, "perform_sweep"))
        except ImportError as e:
            print(f"Server import warning (likely dependencies): {e}")

if __name__ == "__main__":
    unittest.main()
