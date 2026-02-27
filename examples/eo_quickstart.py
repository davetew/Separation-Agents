#!/usr/bin/env python
"""
EO Quickstart Example
=====================

Demonstrates the three main EO capabilities:
1. Single-unit EO solve
2. Multi-unit EO flowsheet (SX → Precipitator → IX)
3. GDP superstructure optimization

Requirements:
    pip install -e .
    conda install ipopt -c conda-forge

Run:
    PYTHONPATH=src python examples/eo_quickstart.py
"""
from __future__ import annotations

import os, sys

os.environ["FASTMCP_NO_BANNER"] = "1"
os.environ["FASTMCP_QUIET"] = "1"
os.environ.setdefault("TMPDIR", "/tmp")

import warnings
warnings.filterwarnings("ignore")

from sep_agents.dsl.schemas import (
    Flowsheet, UnitOp, Stream, Superstructure, DisjunctionDef
)


# ── Helper ────────────────────────────────────────────────────────────

REE_FEED = Stream(
    name="feed", phase="liquid",
    composition_wt={
        "H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
        "Nd+3": 10, "HCl(aq)": 50,
    },
)


# ═══════════════════════════════════════════════════════════════════════
# 1. Single SX Stage
# ═══════════════════════════════════════════════════════════════════════

def demo_single_sx():
    """Solve a single SX stage with EO and print extraction fractions."""
    from sep_agents.sim.eo_flowsheet import run_eo
    import pyomo.environ as pyo

    fs = Flowsheet(name="single_sx", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
    ], streams=[REE_FEED])

    result = run_eo(fs, objective="none")
    assert result["status"] == "ok", f"Solve failed: {result.get('error')}"

    print("  Extraction fractions:")
    sx = result["model"].u_sx_1
    for j in ["La", "Ce", "Nd"]:
        E = pyo.value(sx.extraction[j])
        print(f"    {j}: {E:.4f}")
    print(f"  Solve time: {result['solve_time_s']:.2f}s")


# ═══════════════════════════════════════════════════════════════════════
# 2. Multi-Unit Flowsheet (SX → Precipitator → IX)
# ═══════════════════════════════════════════════════════════════════════

def demo_multi_unit():
    """Build and solve a 3-unit EO flowsheet."""
    from sep_agents.sim.eo_flowsheet import run_eo

    fs = Flowsheet(name="3unit", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
        UnitOp(id="precip", type="precipitator",
               params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
               inputs=["org"], outputs=["solid", "barren"]),
        UnitOp(id="ix_1", type="ion_exchange",
               params={"selectivity_coeff": {"La": 1.0, "Ce": 2.0, "Nd": 3.0},
                       "bed_volume_m3": 1.0},
               inputs=["barren"], outputs=["loaded", "eluate"]),
    ], streams=[
        REE_FEED,
        Stream(name="org", phase="liquid"),
        Stream(name="barren", phase="liquid"),
    ])

    result = run_eo(fs, objective="minimize_opex")
    assert result["status"] == "ok", f"Solve failed: {result.get('error')}"

    print("  KPIs:")
    for k, v in sorted(result["kpis"].items()):
        print(f"    {k}: {v}")
    print(f"  Solve time: {result['solve_time_s']:.2f}s")


# ═══════════════════════════════════════════════════════════════════════
# 3. GDP Superstructure Optimization
# ═══════════════════════════════════════════════════════════════════════

def demo_gdp():
    """Optimize topology: should the precipitator be included?"""
    from sep_agents.opt.gdp_eo import solve_gdp_eo

    fs = Flowsheet(name="gdp_demo", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
        UnitOp(id="precip", type="precipitator",
               params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
               inputs=["org"], outputs=["solid", "barren"],
               optional=True),
    ], streams=[
        REE_FEED,
        Stream(name="org", phase="liquid"),
    ])

    # Maximize recovery → should activate the precipitator
    ss = Superstructure(
        name="gdp_demo",
        base_flowsheet=fs,
        fixed_units=["sx_1"],
        objective="maximize_recovery",
    )
    result = solve_gdp_eo(ss)

    print(f"  Status:    {result.status}")
    print(f"  Active:    {result.active_units}")
    print(f"  Bypassed:  {result.bypassed_units}")
    print(f"  Objective: {result.objective_value:.4f}")


# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  EO Quickstart Examples")
    print("=" * 60)

    print("\n1. Single SX Stage:")
    demo_single_sx()

    print("\n2. Multi-Unit Flowsheet (SX → Precip → IX):")
    demo_multi_unit()

    print("\n3. GDP Superstructure Optimization:")
    demo_gdp()

    print("\n" + "=" * 60)
    print("  All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
