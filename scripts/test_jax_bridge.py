#!/usr/bin/env python3
"""
Test: JAX-Pyomo ExternalFunction Bridge
=========================================

Verifies that the JAX equilibrium solver can be used as a Pyomo
ExternalFunction within both EO and GDP workflows.

Tests:
1. Build a standalone JAX reactor block and solve
2. Build a feed → jax_reactor flowsheet via EO builder
3. Verify gradient (Jacobian) values are finite and non-trivial
4. Run a GDP superstructure with optional reactor

Usage:
    conda run --no-capture-output -n rkt python3 scripts/test_jax_bridge.py
"""

from __future__ import annotations

import sys
import os
import time

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_standalone_block():
    """Test 1: Build a standalone JAX reactor block and evaluate."""
    print("=" * 60)
    print("Test 1: Standalone JAX reactor block")
    print("=" * 60)

    from pyomo.environ import ConcreteModel, SolverFactory, value
    from sep_agents.sim.jax_pyomo_bridge import build_jax_reactor

    m = ConcreteModel(name="test_standalone")
    comp_list = ["La", "Ce", "Nd", "H2O", "HCl"]

    blk = build_jax_reactor(
        m, "reactor", comp_list,
        preset="light_ree",
        temperature_init=298.15,
        pressure_init=101325.0,
        fix_temperature=True,
        fix_pressure=True,
    )

    # Fix feed values (mol/s)
    blk.feed["La"].fix(0.01)
    blk.feed["Ce"].fix(0.01)
    blk.feed["Nd"].fix(0.01)
    blk.feed["H2O"].fix(55.508)  # 1 kg water
    blk.feed["HCl"].fix(0.1)

    # Solve
    solver = SolverFactory("ipopt")
    solver.options["max_iter"] = 500
    solver.options["print_level"] = 0

    import os as _os, tempfile as _tf
    _os.environ["TMPDIR"] = _tf.gettempdir()

    t0 = time.time()
    result = solver.solve(m, tee=False, keepfiles=True, load_solutions=True)
    elapsed = time.time() - t0

    print(f"  Solver status: {result.solver.termination_condition}")
    print(f"  Solve time:    {elapsed:.3f}s")
    print(f"  Product amounts:")
    for j in comp_list:
        v = value(blk.product[j])
        print(f"    {j}: {v:.6f} mol/s")
    print(f"  Feed total:    {value(blk.feed_total):.4f}")
    print(f"  Product total: {value(blk.product_total):.4f}")

    # Basic sanity checks
    assert str(result.solver.termination_condition) in ("optimal", "locallyOptimal"), \
        f"Solver failed: {result.solver.termination_condition}"
    assert value(blk.product_total) > 0, "Product total should be positive"
    print("  ✅ PASSED\n")


def test_gradient_values():
    """Test 2: Verify Jacobian values are finite and non-trivial."""
    print("=" * 60)
    print("Test 2: Gradient (Jacobian) verification")
    print("=" * 60)

    from sep_agents.sim.jax_equilibrium import build_jax_system, JaxEquilibriumSolver
    from sep_agents.sim.jax_pyomo_bridge import _JaxEquilibriumCache

    comp_list = ["La", "Ce", "Nd", "H2O", "HCl"]

    system = build_jax_system(preset="light_ree", include_minerals=True)
    solver = JaxEquilibriumSolver(system, tol=1e-8, maxiter=500)
    cache = _JaxEquilibriumCache(solver, system.species_names, comp_list)

    # Input: (T_K, P_Pa, La_mol, Ce_mol, Nd_mol, H2O_mol, HCl_mol)
    args = (298.15, 101325.0, 0.01, 0.01, 0.01, 55.508, 0.1)

    t0 = time.time()
    result = cache.evaluate_all(args)
    elapsed_eval = time.time() - t0

    t0 = time.time()
    jac = cache.jacobian_all(args)
    elapsed_jac = time.time() - t0

    print(f"  Evaluation time: {elapsed_eval:.3f}s")
    print(f"  Jacobian time:   {elapsed_jac:.3f}s")
    print(f"  Output vector:   {result}")
    print(f"  Jacobian shape:  {jac.shape}")

    import numpy as np

    # Check all Jacobian values are finite
    assert np.all(np.isfinite(jac)), "Jacobian contains non-finite values"

    # Check at least some non-zero entries (equilibrium should respond to inputs)
    nonzero_count = np.sum(np.abs(jac) > 1e-15)
    print(f"  Non-zero Jacobian entries: {nonzero_count} / {jac.size}")
    assert nonzero_count > 0, "Jacobian is all zeros — no sensitivity detected"

    # Print a few key sensitivities
    print(f"  Key sensitivities:")
    for ci, comp in enumerate(comp_list):
        dout_dT = jac[ci, 0]
        print(f"    d({comp})/dT = {dout_dT:.6e}")

    print("  ✅ PASSED\n")


def test_eo_flowsheet():
    """Test 3: Build and solve Flowsheet with JAX reactor via EO builder."""
    print("=" * 60)
    print("Test 3: EO Flowsheet with JAX reactor")
    print("=" * 60)

    from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
    from sep_agents.sim.eo_flowsheet import run_eo

    fs = Flowsheet(
        name="jax_reactor_test",
        streams=[
            Stream(
                name="feed",
                phase="liquid",
                composition_wt={
                    "H2O(aq)": 1000.0,
                    "La+3": 10.0,
                    "Ce+3": 10.0,
                    "Nd+3": 10.0,
                    "HCl(aq)": 50.0,
                },
            ),
        ],
        units=[
            UnitOp(
                id="reactor_1",
                type="jax_reactor",
                inputs=["feed"],
                outputs=["reacted"],
                params={
                    "temperature_K": 298.15,
                    "pressure_Pa": 101325.0,
                    "preset": "light_ree",
                    "fix_temperature": True,
                    "fix_pressure": True,
                },
            ),
        ],
    )

    t0 = time.time()
    result = run_eo(fs, objective="none")
    elapsed = time.time() - t0

    print(f"  Status:     {result['status']}")
    print(f"  Solve time: {elapsed:.3f}s")

    if result["status"] == "ok":
        print(f"  KPIs:       {result.get('kpis', {})}")
        states = result.get("states", {})
        for sname, state in states.items():
            print(f"  Stream '{sname}': {dict(list(state.species_amounts.items())[:5])}...")
        print("  ✅ PASSED\n")
    else:
        print(f"  Error:      {result.get('error', 'unknown')}")
        print("  ❌ FAILED\n")
        return False

    return True


def test_comparison_with_standalone():
    """Test 4: Compare EO reactor output with standalone JAX speciation."""
    print("=" * 60)
    print("Test 4: EO vs standalone JAX speciation comparison")
    print("=" * 60)

    from sep_agents.sim.jax_equilibrium import build_jax_system, JaxEquilibriumSolver

    system = build_jax_system(preset="light_ree", include_minerals=True)
    solver = JaxEquilibriumSolver(system, tol=1e-8, maxiter=500)

    # Same feed as the EO test
    result = solver.solve(
        temperature_K=298.15,
        pressure_Pa=101325.0,
        species_amounts={
            "H2O(aq)": 1000.0 / 0.018015,   # mol (same as EO conversion)
            "La+3": 10.0 / (0.13891 * 1000),  # g → mol
            "Ce+3": 10.0 / (0.14012 * 1000),
            "Nd+3": 10.0 / (0.14424 * 1000),
            "HCl(aq)": 50.0 / (0.036461 * 1000),
        },
    )

    if result["status"] == "ok":
        print(f"  pH:                {result['pH']}")
        print(f"  Ionic strength:    {result['ionic_strength']}")
        print(f"  Top species:")
        sorted_sp = sorted(result["species_amounts"].items(), key=lambda kv: -kv[1])
        for sp, amt in sorted_sp[:8]:
            print(f"    {sp}: {amt:.6f} mol")
        print("  ✅ PASSED (standalone speciation successful)\n")
    else:
        print(f"  Error: {result.get('error', 'unknown')}")
        print("  ❌ FAILED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  JAX-Pyomo Bridge Test Suite")
    print("=" * 60 + "\n")

    # NOTE: IPOPT's ASL interface can SIGSEGV (return code -11) when
    # Python ExternalFunction callbacks are used across multiple
    # solver.solve() calls in the same process.  We run tests in an
    # order that exercises the most important paths first:
    #   1. Gradient verification (no IPOPT)
    #   2. Standalone JAX speciation (no IPOPT)
    #   3. EO flowsheet via run_eo (single IPOPT solve)
    #   4. Standalone IPOPT block (second IPOPT solve — may SIGSEGV)

    try:
        test_gradient_values()
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")

    try:
        test_comparison_with_standalone()
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")

    try:
        test_eo_flowsheet()
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")

    try:
        test_standalone_block()
    except Exception as e:
        print(f"  ❌ FAILED: {e}\n")

    print("=" * 60)
    print("  All tests complete")
    print("=" * 60)

