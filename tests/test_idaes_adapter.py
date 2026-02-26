"""
Tests for the IDAES Flowsheet Adapter.

These tests verify the sequential-modular solver and IDAES model construction.
Tests are designed to work both with and without Reaktoro/IDAES installed.
"""

import sys
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Try real imports first; fall back to mocks
try:
    import reaktoro
    import idaes
    REAL_DEPS = True
except ImportError:
    REAL_DEPS = False
    # Mock heavy dependencies for structural tests
    sys.modules.setdefault("reaktoro", MagicMock())
    sys.modules.setdefault("GeoH2", MagicMock())
    sys.modules.setdefault("GeoH2.equilibrium", MagicMock())
    if "GeoH2.equilibrium" in sys.modules:
        sys.modules["GeoH2.equilibrium"].defineSystem = MagicMock(
            return_value=("sys", "min", "sol", "gas")
        )

from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream, PSD


class TestStreamState(unittest.TestCase):
    """Test the lightweight StreamState container."""

    def test_from_dsl_stream(self):
        from sep_agents.sim.idaes_adapter import StreamState

        s = Stream(
            name="feed",
            phase="solid",
            temperature_K=353.15,
            pressure_Pa=101325,
            composition_wt={"Fe2O3": 0.3, "CaO": 0.4, "SiO2": 0.2, "MgO": 0.1},
        )
        ss = StreamState.from_dsl_stream(s)
        self.assertAlmostEqual(ss.temperature_K, 353.15)
        self.assertAlmostEqual(ss.species_amounts["Fe2O3"], 0.3)
        self.assertEqual(ss.flow_mol, 1.0)  # sum of composition_wt

    def test_copy_independence(self):
        from sep_agents.sim.idaes_adapter import StreamState

        ss1 = StreamState(species_amounts={"H2O": 55.5, "NaCl": 1.0})
        ss2 = ss1.copy()
        ss2.species_amounts["H2O"] = 0
        self.assertAlmostEqual(ss1.species_amounts["H2O"], 55.5)

    def test_to_dict(self):
        from sep_agents.sim.idaes_adapter import StreamState

        ss = StreamState(pH=7.0, Eh_mV=200.0, species_amounts={"H2O": 55.5})
        d = ss.to_dict()
        self.assertEqual(d["pH"], 7.0)
        self.assertIn("H2O", d["species_amounts"])


class TestFlowsheetSolverStructural(unittest.TestCase):
    """Test the sequential solver logic (does not require Reaktoro)."""

    def _make_simple_flowsheet(self):
        """Create a minimal feed -> mill -> cyclone flowsheet."""
        return Flowsheet(
            name="test_fs",
            units=[
                UnitOp(
                    id="mill_1",
                    type="mill",
                    params={"fineness_factor": 0.7, "E_specific_kWhpt": 8.0},
                    inputs=["feed"],
                    outputs=["mill_product"],
                ),
                UnitOp(
                    id="cyclone_1",
                    type="cyclone",
                    params={"d50c_um": 50, "sharpness_alpha": 2.0},
                    inputs=["mill_product"],
                    outputs=["uf", "of"],
                ),
            ],
            streams=[
                Stream(
                    name="feed",
                    phase="solid",
                    composition_wt={"Fe2O3": 0.3, "CaO": 0.4, "SiO2": 0.2, "MgO": 0.1},
                    psd=PSD(
                        bins_um=[500, 250, 125, 63, 32],
                        mass_frac=[0.1, 0.2, 0.3, 0.25, 0.15],
                    ),
                ),
            ],
        )

    def test_topological_solve_order(self):
        """Verify units are solved in correct order."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = self._make_simple_flowsheet()
        builder = IDAESFlowsheetBuilder()

        # _solve_sequential should populate states for all streams
        with patch.object(builder, "_build_model", return_value=MagicMock()):
            states = builder._solve_sequential(fs)

        # Feed + mill_product + uf + of = 4 streams
        self.assertIn("feed", states)
        self.assertIn("mill_product", states)
        self.assertIn("uf", states)
        self.assertIn("of", states)

    def test_separator_mass_balance(self):
        """Verify separator preserves total mass."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="sep_test",
            units=[
                UnitOp(
                    id="lims_1",
                    type="lims",
                    params={"magnetic_recovery": 0.8},
                    inputs=["feed"],
                    outputs=["conc", "tails"],
                ),
            ],
            streams=[
                Stream(name="feed", phase="solid", composition_wt={"Fe3O4": 0.6, "SiO2": 0.4}),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        states = builder._solve_sequential(fs)

        feed_total = sum(states["feed"].species_amounts.values())
        conc_total = sum(states["conc"].species_amounts.values())
        tails_total = sum(states["tails"].species_amounts.values())

        self.assertAlmostEqual(conc_total + tails_total, feed_total, places=6)
        self.assertAlmostEqual(conc_total / feed_total, 0.8, places=6)

    def test_mixer_combines_flows(self):
        """Verify mixer sums species amounts."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="mix_test",
            units=[
                UnitOp(id="mix_1", type="mixer", params={}, inputs=["s1", "s2"], outputs=["mixed"]),
            ],
            streams=[
                Stream(name="s1", phase="liquid", composition_wt={"H2O": 50.0, "NaCl": 1.0}),
                Stream(name="s2", phase="liquid", composition_wt={"H2O": 30.0, "KCl": 0.5}),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        states = builder._solve_sequential(fs)

        self.assertAlmostEqual(states["mixed"].species_amounts["H2O"], 80.0)
        self.assertAlmostEqual(states["mixed"].species_amounts["NaCl"], 1.0)
        self.assertAlmostEqual(states["mixed"].species_amounts["KCl"], 0.5)

    def test_solvent_extraction(self):
        """Verify solvent extraction splits species based on D and O/A ratio."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="sx_test",
            units=[
                UnitOp(
                    id="sx_1",
                    type="solvent_extraction",
                    params={
                        "distribution_coeff": {"Ce+3": 2.0, "Nd+3": 5.0},
                        "organic_to_aqueous_ratio": 2.0
                    },
                    inputs=["feed"],
                    outputs=["loaded_org", "raffinate"],
                ),
            ],
            streams=[
                Stream(name="feed", phase="liquid", composition_wt={"H2O(aq)": 100.0, "Ce+3": 10.0, "Nd+3": 10.0, "Al+3": 5.0}),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        states = builder._solve_sequential(fs)

        org = states["loaded_org"]
        aq = states["raffinate"]

        # Ce+3: D=2, O/A=2 -> D*O/A = 4. amt_aq = 10 / (1+4) = 2.0. amt_org = 10 - 2 = 8.0
        self.assertAlmostEqual(aq.species_amounts["Ce+3"], 2.0)
        self.assertAlmostEqual(org.species_amounts["Ce+3"], 8.0)

        # Nd+3: D=5, O/A=2 -> D*O/A = 10. amt_aq = 10 / (1+10) = 10/11 ~ 0.90909
        self.assertAlmostEqual(aq.species_amounts["Nd+3"], 10.0 / 11.0)
        self.assertAlmostEqual(org.species_amounts["Nd+3"], 10.0 - 10.0 / 11.0)

        # Al+3 defaults to D=0.0
        self.assertAlmostEqual(aq.species_amounts["Al+3"], 5.0)
        self.assertNotIn("Al+3", org.species_amounts)

        # H2O(aq) stays in aqueous
        self.assertAlmostEqual(aq.species_amounts["H2O(aq)"], 100.0)
        self.assertNotIn("H2O(aq)", org.species_amounts)

    def test_ion_exchange(self):
        """Verify ion exchange splits based on S."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="ix_test",
            units=[
                UnitOp(
                    id="ix_1",
                    type="ion_exchange",
                    params={
                        "selectivity_coeff": {"Ce+3": 0.8, "Nd+3": 0.95},
                        "bed_volume_m3": 1.0,
                    },
                    inputs=["feed"],
                    outputs=["resin", "barren"],
                ),
            ],
            streams=[
                Stream(name="feed", phase="liquid", composition_wt={"H2O(aq)": 100.0, "Ce+3": 10.0, "Nd+3": 10.0, "Al+3": 5.0}),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        states = builder._solve_sequential(fs)

        resin = states["resin"]
        barren = states["barren"]

        self.assertAlmostEqual(resin.species_amounts["Ce+3"], 8.0)
        self.assertAlmostEqual(barren.species_amounts["Ce+3"], 2.0)

        self.assertAlmostEqual(resin.species_amounts["Nd+3"], 9.5)
        self.assertAlmostEqual(barren.species_amounts["Nd+3"], 0.5)

        # Al+3 defaults to 0.0
        self.assertNotIn("Al+3", resin.species_amounts)
        self.assertAlmostEqual(barren.species_amounts["Al+3"], 5.0)


@unittest.skipUnless(REAL_DEPS, "Requires reaktoro + idaes installed")
class TestIDAESIntegration(unittest.TestCase):
    """Integration tests with real Reaktoro and IDAES."""

    def test_build_model_creates_flowsheet_block(self):
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="integration_test",
            units=[],
            streams=[
                Stream(name="feed", phase="liquid", composition_wt={"H2O": 55.5}),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        model = builder._build_model(fs)

        self.assertTrue(hasattr(model, "fs"))
        self.assertTrue(hasattr(model.fs, "properties"))

    def test_reactor_equilibrium(self):
        """Test that a leach reactor produces equilibrium output."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="reactor_test",
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
                    composition_wt={"H2O(aq)": 55.5, "Calcite": 1.0},
                ),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        result = builder.build_and_solve(fs)

        self.assertEqual(result["status"], "ok")
        self.assertIn("product", result["streams"])
        prod = result["streams"]["product"]
        # Equilibrium should produce some pH value
        if prod.get("pH") is not None:
            self.assertGreater(prod["pH"], 0)

    def test_full_build_and_solve(self):
        """End-to-end: build + solve + extract."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="e2e_test",
            units=[
                UnitOp(
                    id="lims_1",
                    type="lims",
                    params={"magnetic_recovery": 0.7},
                    inputs=["feed"],
                    outputs=["conc", "tails"],
                ),
            ],
            streams=[
                Stream(name="feed", phase="solid", composition_wt={"Fe3O4": 0.6, "SiO2": 0.4}),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        result = builder.build_and_solve(fs)

        self.assertEqual(result["status"], "ok")
        self.assertIn("conc", result["streams"])
        self.assertIn("tails", result["streams"])
        self.assertIn("lims_1.recovery", result["kpis"])
        self.assertAlmostEqual(result["kpis"]["lims_1.recovery"], 0.7, places=3)

    def test_crystallizer(self):
        """Test that crystallizer produces solid and liquid fractions."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder

        fs = Flowsheet(
            name="cryst_test",
            units=[
                UnitOp(
                    id="cryst_1",
                    type="crystallizer",
                    params={"T_C": 25, "residence_time_s": 3600},
                    inputs=["feed"],
                    outputs=["crystals", "liquor"],
                ),
            ],
            streams=[
                Stream(
                    name="feed",
                    phase="liquid",
                    composition_wt={"H2O(aq)": 55.5, "Calcite": 1.0},
                ),
            ],
        )
        builder = IDAESFlowsheetBuilder()
        builder.database_name = "SUPRCRT - BL"
        result = builder.build_and_solve(fs)

        self.assertEqual(result["status"], "ok")
        cryst = result["streams"]["crystals"]
        liq = result["streams"]["liquor"]

        # Minerals go to crystals (Calcite). Aqueous goes to liquor.
        self.assertTrue(any(v > 0 for k, v in cryst["species_amounts"].items() if not k.endswith("(aq)") and k not in ["H2O", "H+", "OH-", "Na+", "Cl-", "CO2(aq)"]))
        self.assertTrue(liq["species_amounts"].get("H2O(aq)", 0) > 0)


if __name__ == "__main__":
    unittest.main()
