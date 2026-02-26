"""
Tests for REE Reaktoro Database configurations.

Tests the REE system builder, speciation solver, and separation factor
calculations from sep_agents.properties.ree_databases.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check if Reaktoro is available
try:
    import reaktoro
    REAKTORO_AVAILABLE = True
except ImportError:
    REAKTORO_AVAILABLE = False


@unittest.skipUnless(REAKTORO_AVAILABLE, "Requires reaktoro installed")
class TestREEDatabases(unittest.TestCase):

    def test_build_light_ree_system(self):
        from sep_agents.properties.ree_databases import build_ree_system
        system = build_ree_system(preset="light_ree")
        self.assertGreater(system.species().size(), 50)
        self.assertGreater(system.phases().size(), 0)

        # Verify REE species are present
        species_names = [sp.name() for sp in system.species()]
        self.assertIn("Ce+3", species_names)
        self.assertIn("La+3", species_names)
        self.assertIn("Nd+3", species_names)
        self.assertIn("Pr+3", species_names)

    def test_build_heavy_ree_system(self):
        from sep_agents.properties.ree_databases import build_ree_system
        system = build_ree_system(preset="heavy_ree")
        species_names = [sp.name() for sp in system.species()]
        self.assertIn("Dy+3", species_names)
        self.assertIn("Y+3", species_names)

    def test_build_full_ree_system(self):
        from sep_agents.properties.ree_databases import build_ree_system
        system = build_ree_system(preset="full_ree")
        species_names = [sp.name() for sp in system.species()]
        # Should have both light and heavy REE
        self.assertIn("Ce+3", species_names)
        self.assertIn("Dy+3", species_names)
        self.assertIn("Y+3", species_names)

    def test_build_custom_elements(self):
        from sep_agents.properties.ree_databases import build_ree_system
        system = build_ree_system(elements=["Ce", "Nd", "Fe"])
        species_names = [sp.name() for sp in system.species()]
        self.assertIn("Ce+3", species_names)
        self.assertIn("Nd+3", species_names)
        self.assertIn("Fe+3", species_names)

    def test_speciation_ce_nd_in_hcl(self):
        """Key test: REE speciation in HCl leach solution."""
        from sep_agents.properties.ree_databases import REEEquilibriumSolver

        solver = REEEquilibriumSolver(preset="light_ree")
        result = solver.speciate(
            temperature_C=80.0,
            acid_mol={"HCl(aq)": 0.5},
            ree_mol={"Ce+3": 0.01, "Nd+3": 0.005},
            other_mol={"NaCl(aq)": 0.1},
        )

        self.assertEqual(result["status"], "ok")
        self.assertGreater(result["pH"], -1)
        self.assertLess(result["pH"], 3)  # Should be acidic

        # REE distribution should contain chloride complexes
        ree_dist = result["ree_distribution"]
        self.assertIn("CeCl+2", ree_dist)
        self.assertIn("NdCl+2", ree_dist)

        # Total Ce species should sum to ~0.01 mol
        total_ce = sum(v for k, v in ree_dist.items() if "Ce" in k)
        self.assertAlmostEqual(total_ce, 0.01, places=4)

    def test_separation_factor(self):
        """Test separation factor calculation."""
        from sep_agents.properties.ree_databases import REEEquilibriumSolver

        solver = REEEquilibriumSolver(preset="light_ree")
        result = solver.speciate(
            temperature_C=80.0,
            acid_mol={"HCl(aq)": 0.5},
            ree_mol={"Ce+3": 0.01, "Nd+3": 0.01},
            other_mol={"NaCl(aq)": 0.1},
        )

        beta = solver.separation_factors(result, "Ce", "Nd")
        # With equal input, beta should be close to 1
        self.assertGreater(beta, 0.5)
        self.assertLess(beta, 2.0)

    def test_invalid_preset(self):
        from sep_agents.properties.ree_databases import build_ree_system
        with self.assertRaises(ValueError):
            build_ree_system(preset="invalid_preset")


@unittest.skipUnless(REAKTORO_AVAILABLE, "Requires reaktoro installed")
class TestREEWithIDAESAdapter(unittest.TestCase):
    """Test REE presets work through the IDAES adapter."""

    def test_idaes_adapter_with_ree_preset(self):
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder
        from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream

        fs = Flowsheet(
            name="ree_test",
            units=[
                UnitOp(
                    id="leach_1",
                    type="leach_reactor",
                    params={"residence_time_s": 3600, "T_C": 80},
                    inputs=["feed"],
                    outputs=["product"],
                ),
            ],
            streams=[
                Stream(
                    name="feed",
                    phase="liquid",
                    composition_wt={"H2O(aq)": 55.5, "HCl(aq)": 0.5, "Ce+3": 0.01, "Nd+3": 0.005},
                ),
            ],
        )
        builder = IDAESFlowsheetBuilder(database_name="light_ree")
        result = builder.build_and_solve(fs)

        self.assertEqual(result["status"], "ok")
        product = result["streams"]["product"]
        # Should have dissolved REE species
        self.assertIn("species_amounts", product)
        # Should have pH from Reaktoro
        self.assertIsNotNone(product.get("pH"))


if __name__ == "__main__":
    unittest.main()
