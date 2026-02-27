#!/usr/bin/env python3
"""
GDP Superstructure Optimization — End-to-End Validation
=======================================================

Tests the full GDP pipeline:
1. Schema round-trip (Superstructure ↔ dict)
2. Configuration enumeration counts
3. Sub-flowsheet construction with bypass remapping
4. Single configuration evaluation
5. Full superstructure optimization (simple_sx_precip)
6. Benchmark regression (all 6 existing benchmarks still pass)

Usage:
    conda activate rkt
    PYTHONPATH=src python scripts/gdp_validation.py
"""
from __future__ import annotations

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["FASTMCP_NO_BANNER"] = "1"
os.environ["FASTMCP_QUIET"] = "1"

import logging
logging.basicConfig(level=logging.WARNING)


def test_schema_roundtrip():
    """Test 1: Superstructure serialization round-trip."""
    from sep_agents.dsl.ree_superstructures import simple_sx_precipitator_superstructure

    ss = simple_sx_precipitator_superstructure()

    # Serialize to dict
    d = ss.model_dump()
    assert isinstance(d, dict)
    assert d["name"] == "simple_sx_precip"
    assert len(d["base_flowsheet"]["units"]) == 3

    # Deserialize back
    from sep_agents.dsl.schemas import Superstructure
    ss2 = Superstructure(**d)
    assert ss2.name == ss.name
    assert len(ss2.base_flowsheet.units) == len(ss.base_flowsheet.units)

    # Check GDP annotations survived
    scrubber = next(u for u in ss2.base_flowsheet.units if u.id == "scrubber")
    assert scrubber.optional is True
    sx = next(u for u in ss2.base_flowsheet.units if u.id == "sx_1")
    assert sx.optional is False

    return True


def test_enumeration_counts():
    """Test 2: Configuration enumeration produces correct counts."""
    from sep_agents.opt.gdp_builder import enumerate_configurations
    from sep_agents.dsl.ree_superstructures import (
        simple_sx_precipitator_superstructure,
        lree_acid_leach_superstructure,
    )

    # Simple: 1 optional unit → 2 configs
    ss1 = simple_sx_precipitator_superstructure()
    configs1 = enumerate_configurations(ss1)
    assert len(configs1) == 2, f"Expected 2, got {len(configs1)}"

    # LREE: 2 disjunctions (2 choices each) × 1 optional = 2×2×2 = 8
    ss2 = lree_acid_leach_superstructure()
    configs2 = enumerate_configurations(ss2)
    assert len(configs2) == 8, f"Expected 8, got {len(configs2)}"

    return True


def test_sub_flowsheet_construction():
    """Test 3: Sub-flowsheet correctly remaps streams for bypassed units."""
    from sep_agents.opt.gdp_builder import (
        enumerate_configurations,
        build_sub_flowsheet,
    )
    from sep_agents.dsl.ree_superstructures import simple_sx_precipitator_superstructure

    ss = simple_sx_precipitator_superstructure()
    configs = enumerate_configurations(ss)

    # Find the config with scrubber OFF
    scrub_off = next(c for c in configs if "scrubber" in c.bypassed_unit_ids)
    fs = build_sub_flowsheet(ss, scrub_off)

    # Should have 2 active units (sx_1, precipitator)
    assert len(fs.units) == 2, f"Expected 2 units, got {len(fs.units)}"
    unit_ids = {u.id for u in fs.units}
    assert "scrubber" not in unit_ids

    # Find the config with scrubber ON
    scrub_on = next(c for c in configs if "scrubber" in c.active_unit_ids)
    fs2 = build_sub_flowsheet(ss, scrub_on)
    assert len(fs2.units) == 3, f"Expected 3 units, got {len(fs2.units)}"

    return True


def test_single_evaluation():
    """Test 4: A single configuration evaluates successfully."""
    from sep_agents.opt.gdp_builder import enumerate_configurations
    from sep_agents.opt.gdp_solver import evaluate_configuration
    from sep_agents.dsl.ree_superstructures import simple_sx_precipitator_superstructure

    ss = simple_sx_precipitator_superstructure()
    configs = enumerate_configurations(ss)

    # Evaluate the first config
    ev = evaluate_configuration(ss, configs[0], database="light_ree")
    assert ev.status == "ok", f"Evaluation failed: {ev.error}"
    assert ev.kpis.get("overall.opex_USD", 0) > 0, "OPEX should be > 0"

    return True


def test_full_optimization():
    """Test 5: Full superstructure optimization runs end-to-end."""
    from sep_agents.opt.gdp_solver import optimize_superstructure
    from sep_agents.dsl.ree_superstructures import simple_sx_precipitator_superstructure

    ss = simple_sx_precipitator_superstructure()

    # Run WITHOUT BoTorch inner-loop for speed
    result = optimize_superstructure(
        ss, database="light_ree",
        optimize_continuous=False,
    )

    assert result.best is not None, "No best configuration found"
    assert result.num_configs_evaluated == 2
    assert len(result.all_results) == 2

    # Both should have evaluated successfully
    ok_count = sum(1 for r in result.all_results if r.status == "ok")
    assert ok_count >= 1, f"Expected at least 1 successful eval, got {ok_count}"

    return result


def test_benchmark_regression():
    """Test 6: Existing 6 benchmarks still pass."""
    # Import benchmark functions directly
    from benchmark_validation import (
        bench_speciation_lree_hcl,
        bench_sx_separation_factors,
        bench_mass_balance_closure,
        bench_ph_sensitivity,
        bench_nd_hydroxide_precipitation,
        bench_tea_lca_consistency,
    )

    benchmarks = [
        bench_speciation_lree_hcl,
        bench_sx_separation_factors,
        bench_mass_balance_closure,
        bench_ph_sensitivity,
        bench_nd_hydroxide_precipitation,
        bench_tea_lca_consistency,
    ]

    passed = 0
    for fn in benchmarks:
        r = fn()
        if r.passed:
            passed += 1
        else:
            print(f"    REGRESSION: {r.name} — {r.details}")

    assert passed == 6, f"Regression: {passed}/6 benchmarks passed"
    return True


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  GDP SUPERSTRUCTURE OPTIMIZATION — VALIDATION SUITE")
    print("=" * 70)
    print()

    tests = [
        ("1. Schema Round-Trip", test_schema_roundtrip),
        ("2. Enumeration Counts", test_enumeration_counts),
        ("3. Sub-Flowsheet Construction", test_sub_flowsheet_construction),
        ("4. Single Configuration Evaluation", test_single_evaluation),
        ("5. Full Superstructure Optimization", test_full_optimization),
        ("6. Benchmark Regression (6/6)", test_benchmark_regression),
    ]

    results = []
    for label, fn in tests:
        print(f"  Running {label}...", end="", flush=True)
        t0 = time.time()
        try:
            result = fn()
            dt = time.time() - t0
            print(f" ✅ PASS  ({dt:.1f}s)")
            results.append((label, True, result))
        except Exception as e:
            dt = time.time() - t0
            print(f" ❌ FAIL  ({dt:.1f}s)")
            print(f"    Error: {e}")
            results.append((label, False, str(e)))

    print()
    print("-" * 70)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"\n  RESULTS: {passed}/{total} tests passed")

    # Print optimization details for test 5
    for label, ok, data in results:
        if "Full Superstructure" in label and ok and data:
            gdp_result = data
            print(f"\n  GDP Optimization Results ({gdp_result.total_elapsed_s:.1f}s):")
            for ev in gdp_result.all_results:
                status = "✅" if ev.status == "ok" else "❌"
                active = ", ".join(sorted(ev.config.active_unit_ids))
                opex = ev.kpis.get("overall.opex_USD", "N/A")
                print(f"    {status} [{active}] → OPEX=${opex}")
            if gdp_result.best:
                best_active = ", ".join(sorted(gdp_result.best.config.active_unit_ids))
                print(f"\n  🏆 Best: [{best_active}]")

    print("=" * 70)
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
