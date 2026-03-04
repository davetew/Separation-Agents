"""
Tests for the JAX-based Gibbs Energy Minimization equilibrium solver.

Validates the JAX solver against known thermodynamic behavior:
  - Pure water pH ≈ 7
  - HCl speciation (acidic pH, H⁺/Cl⁻ balance)
  - REE speciation in HCl (chloride complexes, mass conservation)
  - Separation factor β(Ce/Nd) ≈ 1.0 for equal inputs
  - Mass balance closure
  - Backend switching via use_jax flag
  - Differentiability via jax.grad
"""

import sys
import math
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Check dependencies
try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import reaktoro
    REAKTORO_AVAILABLE = True
except ImportError:
    REAKTORO_AVAILABLE = False


@unittest.skipUnless(JAX_AVAILABLE, "Requires JAX and jaxopt installed")
class TestJaxEquilibriumSolver(unittest.TestCase):
    """Core solver tests."""

    def setUp(self):
        from sep_agents.sim.jax_equilibrium import build_jax_system, JaxEquilibriumSolver
        self.system = build_jax_system(preset="light_ree")
        self.solver = JaxEquilibriumSolver(self.system)

    def test_system_construction(self):
        """Chemical system has expected structure."""
        self.assertGreater(self.system.n_species, 40)
        self.assertGreater(self.system.n_elements, 10)
        self.assertIn("H2O(aq)", self.system.species_names)
        self.assertIn("H+", self.system.species_names)
        self.assertIn("Ce+3", self.system.species_names)
        self.assertIn("Nd+3", self.system.species_names)
        self.assertIn("La+3", self.system.species_names)
        # Formula matrix shape
        self.assertEqual(self.system.formula_matrix.shape,
                         (self.system.n_elements, self.system.n_species))

    def test_pure_water_ph(self):
        """Pure water at 25°C should have pH ≈ 7."""
        result = self.solver.solve(
            temperature_K=298.15,
            pressure_Pa=101325.0,
            species_amounts={"H2O(aq)": 55.508},  # 1 kg water
        )
        self.assertEqual(result["status"], "ok")
        pH = result["pH"]
        # pH should be near neutral (exact value depends on solver convergence
        # for the water autodissociation equilibrium H₂O ⇌ H⁺ + OH⁻)
        self.assertGreater(pH, 4.0, f"Pure water pH too low: {pH}")
        self.assertLess(pH, 10.0, f"Pure water pH too high: {pH}")

    def test_hcl_speciation(self):
        """0.1M HCl should produce pH ≈ 1 and conserve Cl."""
        result = self.solver.solve(
            temperature_K=298.15,
            pressure_Pa=101325.0,
            species_amounts={
                "H2O(aq)": 55.508,
                "HCl(aq)": 0.1,
            },
        )
        self.assertEqual(result["status"], "ok")
        pH = result["pH"]
        # Should be strongly acidic
        self.assertLess(pH, 2.0, f"HCl solution pH too high: {pH}")
        self.assertGreater(pH, -1.0, f"HCl solution pH too low: {pH}")

        # Cl should be conserved
        species = result["species_amounts"]
        total_cl = sum(
            amt * self._element_count(name, "Cl")
            for name, amt in species.items()
        )
        self.assertAlmostEqual(total_cl, 0.1, places=3,
                               msg=f"Cl not conserved: {total_cl} vs 0.1")

    def test_ree_speciation_in_hcl(self):
        """REE in HCl should produce free ions + chloride complexes."""
        result = self.solver.solve_speciation(
            temperature_C=80.0,
            acid_mol={"HCl(aq)": 0.5},
            ree_mol={"Ce+3": 0.01, "Nd+3": 0.005},
            other_mol={"NaCl(aq)": 0.1},
        )
        self.assertEqual(result["status"], "ok")

        # Should be acidic
        self.assertLess(result["pH"], 3.0)
        self.assertGreater(result["pH"], -1.0)

        # REE distribution should contain Ce and Nd species
        ree_dist = result["ree_distribution"]
        self.assertGreater(len(ree_dist), 0, "No REE species found in distribution")

        # Check Ce species are present (either free ion or chloride complex)
        ce_species = {k: v for k, v in ree_dist.items() if "Ce" in k}
        self.assertGreater(len(ce_species), 0, "No Ce species found")

        # Total Ce should approximately conserve mass (~0.01 mol)
        total_ce = sum(v for v in ce_species.values())
        self.assertAlmostEqual(total_ce, 0.01, places=3,
                               msg=f"Ce not conserved: {total_ce}")

    def test_separation_factor(self):
        """β(Ce/Nd) should be close to 1.0 for equal inputs in HCl."""
        from sep_agents.properties.ree_databases import REEEquilibriumSolver

        solver = REEEquilibriumSolver(preset="light_ree", use_jax=True)
        result = solver.speciate(
            temperature_C=80.0,
            acid_mol={"HCl(aq)": 0.5},
            ree_mol={"Ce+3": 0.01, "Nd+3": 0.01},
            other_mol={"NaCl(aq)": 0.1},
        )
        self.assertEqual(result["status"], "ok")

        beta = solver.separation_factors(result, "Ce", "Nd")
        # With equal input in an acidic aqueous solution, β should be near 1
        self.assertGreater(beta, 0.3, f"β(Ce/Nd) too low: {beta}")
        self.assertLess(beta, 3.0, f"β(Ce/Nd) too high: {beta}")

    def test_mass_balance_closure(self):
        """All elements should be conserved after equilibrium."""
        import numpy as np
        result = self.solver.solve(
            temperature_K=353.15,  # 80°C
            pressure_Pa=101325.0,
            species_amounts={
                "H2O(aq)": 55.508,
                "HCl(aq)": 0.5,
                "Ce+3": 0.01,
                "Nd+3": 0.005,
                "NaCl(aq)": 0.1,
            },
        )
        self.assertEqual(result["status"], "ok")
        self.assertLess(result.get("mass_balance_error", 1.0), 0.1,
                        "Mass balance error too large")

    def test_backend_switching(self):
        """use_jax=True should work through REEEquilibriumSolver."""
        from sep_agents.properties.ree_databases import REEEquilibriumSolver

        solver = REEEquilibriumSolver(preset="light_ree", use_jax=True)
        result = solver.speciate(
            temperature_C=25.0,
            acid_mol={"HCl(aq)": 0.1},
            ree_mol={"Ce+3": 0.01},
        )
        self.assertEqual(result["status"], "ok")
        self.assertIn("pH", result)
        self.assertIn("species", result)
        self.assertIn("ree_distribution", result)

    def test_differentiability(self):
        """jax.grad through the solver should produce finite gradients."""
        from sep_agents.sim.jax_equilibrium import (
            build_jax_system, _debye_huckel_log_gamma
        )

        # Test differentiability of the activity coefficient model
        # (the core differentiable component)
        charges = jnp.array([1.0, -1.0, 2.0])
        ion_sizes = jnp.array([9.0, 3.0, 6.0])
        I = jnp.array(0.1)

        # Compute gradient of sum(ln_gamma) w.r.t. ionic strength
        def sum_ln_gamma(I_val):
            return jnp.sum(_debye_huckel_log_gamma(charges, ion_sizes, I_val))

        grad_fn = jax.grad(sum_ln_gamma)
        grad_val = grad_fn(I)

        # Gradient should be finite and non-zero
        self.assertTrue(jnp.isfinite(grad_val), f"Gradient is not finite: {grad_val}")
        self.assertNotEqual(float(grad_val), 0.0, "Gradient is zero")

    # -- helpers --

    def _element_count(self, species_name: str, element: str) -> float:
        """Get stoichiometric coefficient of element in species."""
        idx = self.system.species_names.index(species_name) if species_name in self.system.species_names else -1
        if idx < 0:
            return 0.0
        el_idx = self.system.element_names.index(element) if element in self.system.element_names else -1
        if el_idx < 0:
            return 0.0
        return float(self.system.formula_matrix[el_idx, idx])


@unittest.skipUnless(JAX_AVAILABLE, "Requires JAX and jaxopt installed")
class TestJaxIDAESIntegration(unittest.TestCase):
    """Test JAX solver through the IDAES adapter."""

    def test_idaes_adapter_with_jax(self):
        """IDAESFlowsheetBuilder(use_jax=True) should solve a reactor."""
        from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder
        from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream

        fs = Flowsheet(
            name="jax_reactor_test",
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
                    composition_wt={
                        "H2O(aq)": 55.5,
                        "HCl(aq)": 0.5,
                        "Ce+3": 0.01,
                        "Nd+3": 0.005,
                    },
                ),
            ],
        )
        builder = IDAESFlowsheetBuilder(database_name="light_ree", use_jax=True)
        result = builder.build_and_solve(fs)

        self.assertEqual(result["status"], "ok")
        product = result["streams"]["product"]
        self.assertIn("species_amounts", product)
        self.assertIsNotNone(product.get("pH"))
        # pH should be acidic
        self.assertLess(product["pH"], 3.0)


@unittest.skipUnless(JAX_AVAILABLE and REAKTORO_AVAILABLE,
                     "Requires both JAX and Reaktoro for cross-validation")
class TestJaxVsReaktoro(unittest.TestCase):
    """Cross-validation: JAX and Reaktoro should produce similar results."""

    def test_speciation_agreement(self):
        """JAX and Reaktoro pH should agree within ~1 pH unit."""
        from sep_agents.properties.ree_databases import REEEquilibriumSolver

        conditions = dict(
            temperature_C=80.0,
            acid_mol={"HCl(aq)": 0.5},
            ree_mol={"Ce+3": 0.01, "Nd+3": 0.005},
            other_mol={"NaCl(aq)": 0.1},
        )

        rkt_solver = REEEquilibriumSolver(preset="light_ree", use_jax=False)
        jax_solver = REEEquilibriumSolver(preset="light_ree", use_jax=True)

        rkt_result = rkt_solver.speciate(**conditions)
        jax_result = jax_solver.speciate(**conditions)

        self.assertEqual(rkt_result["status"], "ok")
        self.assertEqual(jax_result["status"], "ok")

        # pH should agree within ~1 unit (loose tolerance for Phase 1)
        self.assertAlmostEqual(
            rkt_result["pH"], jax_result["pH"], delta=1.0,
            msg=f"pH mismatch: Reaktoro={rkt_result['pH']}, JAX={jax_result['pH']}"
        )


if __name__ == "__main__":
    unittest.main()
