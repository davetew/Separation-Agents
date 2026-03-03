#!/usr/bin/env python
"""
Waste Stream REE Recovery — GDP Topology Optimisation
=====================================================

Evaluates economically viable REE recovery routes from three unconventional
waste streams using the Separation Agents EO + GDP framework.

Waste Streams
-------------
1. **Coal Fly Ash Leachate**   — dilute LREE (La, Ce, Nd) + heavy gangue (Fe, Al)
2. **Acid Mine Drainage (AMD)** — ultra-dilute mixed REE + high Fe/Mn/Zn
3. **Red Mud Leachate**         — moderate REE + extreme Al/Fe/Ti

Candidate Topologies
--------------------
For each waste stream, GDP evaluates:

- **Route A**: SX-only (baseline)
- **Route B**: SX → Precipitator (REE concentration + purification)
- **Route C**: IX → SX (pre-concentration by IX, then SX separation)
- **Route D**: IX → Precipitator (no SX — membrane/IX route)
- **Route E**: Full chain: SX → Precipitator → IX (all active)
- **Route F**: GDP decides (optional Precip + XOR between SX/IX)

The GDP solver simultaneously selects the best topology AND optimises
continuous parameters (D, OA_ratio, recovery fractions).

Run
---
    TMPDIR=/tmp PYTHONPATH=src python examples/waste_stream_ree_recovery.py

Outputs
-------
    reports/waste_stream_ree_recovery_YYYYMMDD_HHMMSS.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

os.environ["FASTMCP_NO_BANNER"] = "1"
os.environ["FASTMCP_QUIET"] = "1"
os.environ.setdefault("TMPDIR", "/tmp")

# ═══════════════════════════════════════════════════════════════════════
# Waste Stream Definitions
# ═══════════════════════════════════════════════════════════════════════

WASTE_STREAMS = {
    "coal_fly_ash": {
        "description": "Coal Fly Ash Leachate (HCl leach of Class F fly ash)",
        "context": (
            "Coal fly ash from pulverised-coal power plants contains 200–400 ppm "
            "total REE, concentrated in the glassy aluminosilicate fraction. After "
            "HCl leaching at 90°C for 4 h, the resulting liquor has ~0.1% REE "
            "dissolved alongside significant Fe³⁺, Al³⁺, and Ca²⁺."
        ),
        "composition_wt": {
            "H2O(aq)": 1000.0,
            "La+3": 0.8,       # ~80 ppm in leachate
            "Ce+3": 1.5,       # ~150 ppm — dominant LREE in fly ash
            "Nd+3": 0.5,       # ~50 ppm
            "HCl(aq)": 100.0,  # residual acid from leaching
        },
        "pH": 1.5,
    },
    "acid_mine_drainage": {
        "description": "Acid Mine Drainage (AMD) from Coal Mine Discharge",
        "context": (
            "Acidic runoff from coal mines in the Appalachian Basin contains "
            "dissolved REEs at 0.5–5 mg/L. Total flow rates can be enormous "
            "(100s of GPM), but REE concentration is ultra-low compared to "
            "Fe/Mn/Al. IX pre-concentration is typically essential."
        ),
        "composition_wt": {
            "H2O(aq)": 1000.0,
            "La+3": 0.05,      # ~5 ppm
            "Ce+3": 0.10,      # ~10 ppm
            "Nd+3": 0.03,      # ~3 ppm
            "HCl(aq)": 30.0,   # pH ~3.5
        },
        "pH": 3.5,
    },
    "red_mud": {
        "description": "Red Mud Leachate (HCl leach of Bayer process residue)",
        "context": (
            "Red mud — the alkaline residue from alumina refining — contains "
            "500–1500 ppm REE, predominantly Ce, La, Nd. After acid leaching, "
            "the liquor has very high Fe and Al alongside moderate REE. The "
            "gangue-to-REE ratio can exceed 100:1, demanding selective "
            "separation chemistry."
        ),
        "composition_wt": {
            "H2O(aq)": 1000.0,
            "La+3": 2.0,       # ~200 ppm in leachate
            "Ce+3": 3.5,       # ~350 ppm — dominant
            "Nd+3": 1.2,       # ~120 ppm
            "HCl(aq)": 150.0,  # residual from aggressive leach
        },
        "pH": 0.8,
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Route Definitions (each is a subprocess worker)
# ═══════════════════════════════════════════════════════════════════════

# Each route is a template function that returns a (flowsheet_dict, is_gdp,
# superstructure_dict) tuple as JSON-serialisable dicts so we can run them
# in isolated subprocesses (prevents IPOPT state corruption).

WORKER_TEMPLATE = textwrap.dedent(r'''
import json, os, sys, warnings, time
os.environ["FASTMCP_NO_BANNER"] = "1"
os.environ["FASTMCP_QUIET"] = "1"
os.environ.setdefault("TMPDIR", "/tmp")
warnings.filterwarnings("ignore")

from sep_agents.dsl.schemas import (
    Flowsheet, UnitOp, Stream, Superstructure, DisjunctionDef
)

feed_data = json.loads(sys.argv[1])
route_name = sys.argv[2]

feed = Stream(name="feed", phase="liquid",
              composition_wt=feed_data["composition_wt"])

def route_A():
    """SX-only"""
    return Flowsheet(name="sx_only", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
    ], streams=[feed])

def route_B():
    """SX → Precipitator"""
    return Flowsheet(name="sx_precip", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
        UnitOp(id="precip", type="precipitator",
               params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
               inputs=["org"], outputs=["solid", "barren"]),
    ], streams=[
        feed,
        Stream(name="org", phase="liquid"),
    ])

def route_C():
    """IX → SX (pre-concentrate then separate)"""
    return Flowsheet(name="ix_sx", units=[
        UnitOp(id="ix_1", type="ion_exchange",
               params={"selectivity_coeff": {"La": 1.0, "Ce": 2.5, "Nd": 2.0},
                       "bed_volume_m3": 1.0},
               inputs=["feed"], outputs=["loaded", "eluate"]),
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["loaded"], outputs=["org", "raf"]),
    ], streams=[
        feed,
        Stream(name="loaded", phase="liquid"),
    ])

def route_D():
    """IX → Precipitator (no SX)"""
    return Flowsheet(name="ix_precip", units=[
        UnitOp(id="ix_1", type="ion_exchange",
               params={"selectivity_coeff": {"La": 1.0, "Ce": 2.5, "Nd": 2.0},
                       "bed_volume_m3": 1.0},
               inputs=["feed"], outputs=["loaded", "eluate"]),
        UnitOp(id="precip", type="precipitator",
               params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
               inputs=["loaded"], outputs=["solid", "barren"]),
    ], streams=[
        feed,
        Stream(name="loaded", phase="liquid"),
    ])

def route_E():
    """SX → Precip → IX (full chain)"""
    return Flowsheet(name="full_chain", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
        UnitOp(id="precip", type="precipitator",
               params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
               inputs=["org"], outputs=["solid", "barren"]),
        UnitOp(id="ix_1", type="ion_exchange",
               params={"selectivity_coeff": {"La": 1.0, "Ce": 2.5, "Nd": 2.0},
                       "bed_volume_m3": 1.0},
               inputs=["barren"], outputs=["loaded", "eluate"]),
    ], streams=[
        feed,
        Stream(name="org", phase="liquid"),
        Stream(name="barren", phase="liquid"),
    ])

def route_F_gdp():
    """GDP decides: SX fixed, optional Precip, optional IX"""
    fs = Flowsheet(name="gdp_auto", units=[
        UnitOp(id="sx_1", type="solvent_extraction",
               params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                       "organic_to_aqueous_ratio": 1.0},
               inputs=["feed"], outputs=["org", "raf"]),
        UnitOp(id="precip", type="precipitator",
               params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
               inputs=["org"], outputs=["solid", "barren"],
               optional=True),
        UnitOp(id="ix_1", type="ion_exchange",
               params={"selectivity_coeff": {"La": 1.0, "Ce": 2.5, "Nd": 2.0},
                       "bed_volume_m3": 1.0},
               inputs=["barren"], outputs=["loaded", "eluate"],
               optional=True),
    ], streams=[
        feed,
        Stream(name="org", phase="liquid"),
        Stream(name="barren", phase="liquid"),
    ])
    ss = Superstructure(
        name="gdp_waste",
        base_flowsheet=fs,
        fixed_units=["sx_1"],
        objective="maximize_recovery",
    )
    return fs, ss


t0 = time.time()
result_dict = {}

if route_name == "F":
    from sep_agents.opt.gdp_eo import solve_gdp_eo
    _, ss = route_F_gdp()
    r = solve_gdp_eo(ss)
    result_dict = {
        "status": r.status,
        "objective_value": r.objective_value,
        "active_units": r.active_units,
        "bypassed_units": r.bypassed_units,
        "kpis": r.kpis,
        "solve_time_s": r.solve_time_s,
        "is_gdp": True,
    }
else:
    from sep_agents.sim.eo_flowsheet import run_eo
    builders = {"A": route_A, "B": route_B, "C": route_C,
                "D": route_D, "E": route_E}
    fs = builders[route_name]()
    r = run_eo(fs, objective="minimize_opex")
    result_dict = {
        "status": r["status"],
        "kpis": r["kpis"],
        "solve_time_s": r["solve_time_s"],
        "is_gdp": False,
    }

result_dict["wall_time_s"] = round(time.time() - t0, 3)
print(json.dumps(result_dict))
''')


# ═══════════════════════════════════════════════════════════════════════
# Subprocess runner
# ═══════════════════════════════════════════════════════════════════════

def run_route(stream_key: str, route_name: str, feed_data: dict) -> dict:
    """Run a single route in an isolated subprocess."""
    import tempfile
    script_path = os.path.join(tempfile.gettempdir(), "_ws_worker.py")
    with open(script_path, "w") as f:
        f.write(WORKER_TEMPLATE)

    feed_json = json.dumps(feed_data)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    env["TMPDIR"] = "/tmp"

    proc = subprocess.run(
        [sys.executable, script_path, feed_json, route_name],
        capture_output=True, text=True, timeout=60, env=env,
    )

    if proc.returncode != 0:
        return {
            "status": "error",
            "error": proc.stderr[-500:] if proc.stderr else f"RC={proc.returncode}",
            "wall_time_s": 0,
        }

    # Parse the last JSON line
    for line in reversed(proc.stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                pass
    return {"status": "error", "error": "No JSON output", "wall_time_s": 0}


# ═══════════════════════════════════════════════════════════════════════
# Report generator
# ═══════════════════════════════════════════════════════════════════════

ROUTE_LABELS = {
    "A": "SX Only",
    "B": "SX → Precipitator",
    "C": "IX → SX",
    "D": "IX → Precipitator",
    "E": "SX → Precip → IX",
    "F": "GDP Auto-Select",
}


def generate_report(all_results: dict) -> str:
    """Generate a Markdown report comparing routes across waste streams."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# Waste Stream REE Recovery — Route Comparison Report",
        "",
        f"**Generated:** {ts}  ",
        "**Framework:** Separation Agents (EO + GDP)  ",
        "**Solver:** IPOPT 3.14.19 with Big-M transformation",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This report evaluates six candidate process topologies for recovering "
        "Rare Earth Elements (REEs) from three unconventional waste streams: "
        "coal fly ash leachate, acid mine drainage, and red mud leachate.",
        "",
        "Each route is evaluated for **OPEX** (proxy operating cost) and **REE recovery**, "
        "using the Equation-Oriented (EO) solver for fixed topologies and "
        "Generalized Disjunctive Programming (GDP) for automatic topology selection.",
        "",
    ]

    # Per-stream sections
    for stream_key, stream_info in WASTE_STREAMS.items():
        results = all_results.get(stream_key, {})
        lines.extend([
            "---",
            "",
            f"## {stream_info['description']}",
            "",
            f"**Feed REE concentration:** "
            f"La={stream_info['composition_wt'].get('La+3', 0)}, "
            f"Ce={stream_info['composition_wt'].get('Ce+3', 0)}, "
            f"Nd={stream_info['composition_wt'].get('Nd+3', 0)} g/L  ",
            f"**pH:** {stream_info.get('pH', 'N/A')}",
            "",
            f"> {stream_info['context']}",
            "",
            "### Route Comparison",
            "",
            "| Route | Topology | Status | OPEX (proxy) | Recovery KPIs | Time |",
            "|-------|----------|--------|-------------|---------------|------|",
        ])

        for route_name in ["A", "B", "C", "D", "E", "F"]:
            r = results.get(route_name, {"status": "skipped"})
            label = ROUTE_LABELS[route_name]
            status = r.get("status", "error")

            if status == "ok":
                kpis = r.get("kpis", {})
                opex = kpis.get("overall.opex_EO", kpis.get("overall.opex_USD", "—"))
                if isinstance(opex, float):
                    opex = f"${opex:.2f}"

                # Gather recovery KPIs
                rec_parts = []
                for k, v in sorted(kpis.items()):
                    if "recovery" in k or k.startswith("sx_1.E_"):
                        if isinstance(v, float):
                            rec_parts.append(f"{k.split('.')[-1]}={v:.3f}")
                recovery_str = ", ".join(rec_parts[:4]) if rec_parts else "—"

                if r.get("is_gdp"):
                    active = r.get("active_units", [])
                    bypassed = r.get("bypassed_units", [])
                    label = f"**GDP** → {'+'.join(sorted(active))}"
                    if bypassed:
                        label += f" (skip {','.join(bypassed)})"

                wt = r.get("wall_time_s", r.get("solve_time_s", "—"))
                lines.append(
                    f"| {route_name} | {label} | ✅ | {opex} | {recovery_str} | {wt}s |"
                )
            else:
                err = r.get("error", "unknown")[:60]
                lines.append(
                    f"| {route_name} | {label} | ❌ {err} | — | — | — |"
                )

        # Analysis
        lines.extend(["", "### Analysis", ""])

        # Find best route by recovery
        best_route = None
        best_recovery = -1
        for rn in ["A", "B", "C", "D", "E", "F"]:
            r = results.get(rn, {})
            if r.get("status") == "ok":
                kpis = r.get("kpis", {})
                # Sum all recovery-like metrics
                total_rec = sum(v for k, v in kpis.items()
                               if ("recovery" in k or "E_" in k) and isinstance(v, float))
                if total_rec > best_recovery:
                    best_recovery = total_rec
                    best_route = rn

        if best_route:
            lines.append(
                f"**Recommended route: {best_route} ({ROUTE_LABELS.get(best_route, 'GDP')})** "
                f"— highest aggregate recovery metric ({best_recovery:.3f})."
            )

        # Stream-specific commentary
        total_ree = sum(stream_info["composition_wt"].get(k, 0)
                        for k in ["La+3", "Ce+3", "Nd+3"])
        if total_ree < 0.5:
            lines.append(
                "\n⚠️ **Ultra-dilute feed** (<0.5 g/L total REE). "
                "IX pre-concentration (Routes C, D) is likely essential for "
                "economic viability. Direct SX on this dilute stream "
                "would require impractically large O/A ratios."
            )
        elif total_ree > 3:
            lines.append(
                "\n✅ **Moderate-to-high REE concentration** (>3 g/L). "
                "Direct SX (Route A or B) is viable. Adding precipitation "
                "improves product purity at modest additional OPEX."
            )

        lines.append("")

    # Methodology section
    lines.extend([
        "---",
        "",
        "## Methodology",
        "",
        "### Simulation Backend",
        "All fixed-topology routes (A–E) were solved using the **Equation-Oriented (EO)** "
        "backend (`run_eo()`), which formulates the entire flowsheet as a single Pyomo "
        "`ConcreteModel` solved simultaneously by IPOPT.",
        "",
        "Route F uses **Generalized Disjunctive Programming (GDP)** via `solve_gdp_eo()`, "
        "which wraps each optional unit in a Pyomo `Disjunct` and applies the Big-M "
        "transformation before solving with IPOPT.",
        "",
        "### Unit Models",
        "",
        "| Unit | Model | Key Parameters |",
        "|------|-------|----------------|",
        "| SX | McCabe-Thiele | D(La)=2.0, D(Ce)=5.0, D(Nd)=3.5, O/A=1.0 |",
        "| Precipitator | Recovery-fraction | τ=3600s, reagent=10 g/L |",
        "| IX | Competitive Langmuir | K(La)=1.0, K(Ce)=2.5, K(Nd)=2.0, V_bed=1 m³ |",
        "",
        "### Subprocess Isolation",
        "Each route is executed in a separate Python subprocess to prevent IPOPT "
        "AMPL-ASL handler corruption between successive solves.",
        "",
        "### Limitations",
        "- Distribution coefficients are fixed empirical values, not pH-dependent",
        "- No gangue species (Fe, Al) modelled in EO backend (SM backend supports these via Reaktoro)",
        "- OPEX is a proxy metric; absolute values should not be used for feasibility engineering",
        "- No recycle streams",
        "",
        "---",
        "",
        "*Report generated by Separation Agents — waste_stream_ree_recovery.py*",
    ])

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Waste Stream REE Recovery — Route Comparison")
    print("  6 routes × 3 waste streams = 18 evaluations")
    print("=" * 70)

    all_results = {}
    total_t0 = time.time()

    for stream_key, stream_info in WASTE_STREAMS.items():
        print(f"\n{'─' * 60}")
        print(f"  Stream: {stream_info['description']}")
        print(f"{'─' * 60}")

        stream_results = {}
        feed_data = {"composition_wt": stream_info["composition_wt"]}

        for route_name in ["A", "B", "C", "D", "E", "F"]:
            label = ROUTE_LABELS[route_name]
            sys.stdout.write(f"  [{route_name}] {label:25s} ... ")
            sys.stdout.flush()

            try:
                result = run_route(stream_key, route_name, feed_data)
                status = result.get("status", "error")
                wt = result.get("wall_time_s", 0)

                if status == "ok":
                    if result.get("is_gdp"):
                        active = result.get("active_units", [])
                        print(f"✅ ({wt}s) → {'+'.join(sorted(active))}")
                    else:
                        kpis = result.get("kpis", {})
                        opex = kpis.get("overall.opex_EO",
                                        kpis.get("overall.opex_USD", "—"))
                        print(f"✅ ({wt}s) OPEX={opex}")
                else:
                    err = result.get("error", "")[:60]
                    print(f"❌ {err}")

                stream_results[route_name] = result
            except Exception as e:
                print(f"❌ {e}")
                stream_results[route_name] = {"status": "error", "error": str(e)}

        all_results[stream_key] = stream_results

    total_time = round(time.time() - total_t0, 1)
    print(f"\n{'=' * 70}")
    print(f"  Total wall time: {total_time}s")
    print(f"{'=' * 70}")

    # Generate report
    report_md = generate_report(all_results)
    reports_dir = Path(__file__).resolve().parents[1] / "reports"
    reports_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = reports_dir / f"waste_stream_ree_recovery_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"\n  Report saved to: {report_path}")


if __name__ == "__main__":
    main()
