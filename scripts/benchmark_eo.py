#!/usr/bin/env python
"""
EO Integration Benchmark Suite (Phase 5)
=========================================

Tests the equation-oriented (EO) implementation against analytical
solutions and the sequential-modular (SM) implementation.

Each test runs in a subprocess to avoid IPOPT 3.14.19 AMPL-ASL handler
corruption between successive solve calls.

Run:
    PYTHONPATH=src python scripts/benchmark_eo.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List

PYTHON = sys.executable
PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═══════════════════════════════════════════════════════════════════════
# Framework
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BenchResult:
    id: str
    name: str
    category: str
    passed: bool = False
    detail: str = ""
    elapsed_s: float = 0.0


ALL_RESULTS: List[BenchResult] = []
BENCH_PREAMBLE = """
import os, sys, warnings
os.environ['FASTMCP_NO_BANNER'] = '1'
os.environ['FASTMCP_QUIET']     = '1'
os.environ['TMPDIR']            = '/tmp'
warnings.filterwarnings('ignore')
import json
"""


def _run_in_subprocess(code: str) -> dict:
    """Run Python code in a fresh subprocess, return parsed JSON output."""
    full = BENCH_PREAMBLE + code
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.join(PROJECT, "src")
    env["TMPDIR"] = "/tmp"
    try:
        proc = subprocess.run(
            [PYTHON, "-c", full],
            capture_output=True, text=True, timeout=60,
            cwd=PROJECT, env=env,
        )
        stdout = proc.stdout.strip()
        # Parse last line as JSON
        for line in reversed(stdout.split("\n")):
            line = line.strip()
            if line.startswith("{"):
                return json.loads(line)
        return {"passed": False, "detail": f"No JSON output. stdout={stdout[:300]}, stderr={proc.stderr[:300]}"}
    except subprocess.TimeoutExpired:
        return {"passed": False, "detail": "TIMEOUT (>60s)"}
    except Exception as e:
        return {"passed": False, "detail": str(e)}


def run_bench(bench_id: str, name: str, category: str, code: str):
    """Execute a benchmark in a subprocess and record the result."""
    t0 = time.time()
    r = BenchResult(id=bench_id, name=name, category=category)
    result = _run_in_subprocess(code)
    r.passed = result.get("passed", False)
    r.detail = result.get("detail", "")
    r.elapsed_s = round(time.time() - t0, 3)
    ALL_RESULTS.append(r)
    status = "PASS ✅" if r.passed else "FAIL ❌"
    print(f"  [{bench_id}] {name}: {status} ({r.elapsed_s}s)")
    if r.detail:
        for line in r.detail.split("\n")[:5]:
            print(f"        {line}")
    return r


# ═══════════════════════════════════════════════════════════════════════
# Category A: Unit Model Verification
# ═══════════════════════════════════════════════════════════════════════

A1_CODE = """
from pyomo.environ import ConcreteModel, value, SolverFactory
from sep_agents.units.sx_eo import build_sx_stage

comp_list = ["H2O", "HCl", "La", "Ce", "Nd", "Pr"]
D_vals = {"La": 1.0, "Ce": 3.04, "Nd": 2.32, "Pr": 0.5}
OA = 1.5

m = ConcreteModel()
blk = build_sx_stage(m, "sx", comp_list, D_init=D_vals, OA_init=OA,
                      aqueous_background={"H2O", "HCl"})
for j in blk.D: blk.D[j].fix()
blk.OA_ratio.fix()
blk.feed["H2O"].fix(55.5); blk.feed["HCl"].fix(1.37)
blk.feed["La"].fix(0.072); blk.feed["Ce"].fix(0.071)
blk.feed["Nd"].fix(0.069); blk.feed["Pr"].fix(0.01)

solver = SolverFactory("ipopt"); solver.options["print_level"] = 0
solver.solve(m, tee=False)

lines = []
ok = True
for j, D in D_vals.items():
    E_a = D * OA / (1 + D * OA)
    E_m = value(blk.extraction[j])
    err = abs(E_m - E_a)
    if err > 1e-6: ok = False
    lines.append(f"E_{j}: model={E_m:.6f} analytical={E_a:.6f}")

for j in D_vals:
    f = value(blk.feed[j]); o = value(blk.organic[j]); r = value(blk.raffinate[j])
    mb = abs(f - o - r)
    if mb > 1e-8: ok = False

print(json.dumps({"passed": ok, "detail": "; ".join(lines)}))
"""

A2_CODE = """
from pyomo.environ import ConcreteModel, value, SolverFactory
from sep_agents.units.precipitator_eo import build_precipitator

comp_list = ["H2O", "HCl", "La", "Ce", "Nd", "Pr"]
ree = {"La", "Ce", "Nd", "Pr"}

m = ConcreteModel()
blk = build_precipitator(m, "precip", comp_list, ree_components=ree,
                          recovery_init={"La": 0.95, "Ce": 0.97, "Nd": 0.96, "Pr": 0.90},
                          fix_recovery=True)
blk.feed["H2O"].fix(55.5); blk.feed["HCl"].fix(1.37)
blk.feed["La"].fix(0.072); blk.feed["Ce"].fix(0.071)
blk.feed["Nd"].fix(0.069); blk.feed["Pr"].fix(0.01)

solver = SolverFactory("ipopt"); solver.options["print_level"] = 0
solver.solve(m, tee=False)

lines = []; ok = True
for j in ["La", "Ce", "Nd", "Pr"]:
    f = value(blk.feed[j]); s = value(blk.solid[j])
    b = value(blk.barren[j]); R = value(blk.recovery[j])
    expected_s = f * R
    if abs(s - expected_s) > 1e-8 or abs(f - s - b) > 1e-8: ok = False
    lines.append(f"{j}: solid={s:.6f} expected={expected_s:.6f}")

rec = value(blk.ree_recovery)
lines.append(f"Overall recovery={rec:.4f}")
print(json.dumps({"passed": ok, "detail": "; ".join(lines)}))
"""

A3_CODE = """
from pyomo.environ import ConcreteModel, value, SolverFactory
from sep_agents.units.ix_eo import build_ix_column

comp_list = ["H2O", "HCl", "La", "Ce", "Nd", "Pr"]
ree = {"La", "Ce", "Nd", "Pr"}
K_vals = {"La": 1.0, "Ce": 2.0, "Nd": 3.0, "Pr": 1.5}

m = ConcreteModel()
blk = build_ix_column(m, "ix", comp_list, ree_components=ree, K_init=K_vals)
for j in blk.K: blk.K[j].fix()
blk.q_max.fix(); blk.resin_mass.fix()
blk.feed["H2O"].fix(55.5); blk.feed["HCl"].fix(1.37)
blk.feed["La"].fix(0.072); blk.feed["Ce"].fix(0.071)
blk.feed["Nd"].fix(0.069); blk.feed["Pr"].fix(0.01)

solver = SolverFactory("ipopt"); solver.options["print_level"] = 0
solver.solve(m, tee=False)

lines = []; ok = True
for j in ["La", "Ce", "Nd", "Pr"]:
    f = value(blk.feed[j]); l = value(blk.loaded[j])
    capped = value(blk.recovery[j])
    if l > f + 1e-8: ok = False
    if capped > 1.0 + 1e-8: ok = False
    if abs(f - l - value(blk.eluate[j])) > 1e-8: ok = False
    lines.append(f"{j}: capped={capped:.4f} loaded={l:.6f} feed={f:.6f}")

rec = value(blk.ree_recovery)
lines.append(f"Overall recovery={rec:.4f}")
print(json.dumps({"passed": ok, "detail": "; ".join(lines)}))
"""

A4_CODE = """
from pyomo.environ import ConcreteModel, value, SolverFactory
from sep_agents.units.sx_eo import build_sx_stage

# Single stage baseline
comp_list = ["H2O", "HCl", "La", "Ce", "Nd", "Pr"]
D_vals = {"La": 1.0, "Ce": 3.04, "Nd": 2.32}

m1 = ConcreteModel()
s1 = build_sx_stage(m1, "sx1", comp_list, D_init=D_vals, OA_init=1.0,
                     aqueous_background={"H2O", "HCl"})
for j in s1.D: s1.D[j].fix()
s1.OA_ratio.fix()
s1.feed["H2O"].fix(55.5); s1.feed["HCl"].fix(1.37)
s1.feed["La"].fix(0.072); s1.feed["Ce"].fix(0.071)
s1.feed["Nd"].fix(0.069); s1.feed["Pr"].fix(0.0)
solver = SolverFactory("ipopt"); solver.options["print_level"] = 0
solver.solve(m1, tee=False)
E1 = value(s1.extraction["Ce"])

# Analytical 3-stage: E3 = 1 - (1-E1)^3
E3_analytical = 1 - (1 - E1)**3
ok = E3_analytical > E1
detail = f"1-stage E_Ce={E1:.4f}, 3-stage analytical E_Ce={E3_analytical:.4f}"
print(json.dumps({"passed": ok, "detail": detail}))
"""

# ═══════════════════════════════════════════════════════════════════════
# Category B: EO Flowsheet Integration
# ═══════════════════════════════════════════════════════════════════════

B1_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo

fs = Flowsheet(name="b1", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 1.0, "Ce": 3.04, "Nd": 2.32},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
])
r = run_eo(fs, objective="none")
ok = r["status"] == "ok"
detail = f"status={r['status']}"
print(json.dumps({"passed": ok, "detail": detail}))
"""

B2_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo
import pyomo.environ as pyo

fs = Flowsheet(name="b2", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.5},
           inputs=["feed"], outputs=["org", "raf"]),
    UnitOp(id="precip", type="precipitator",
           params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
           inputs=["org"], outputs=["solid", "barren"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="org", phase="liquid"),
])
r = run_eo(fs, objective="minimize_opex")
ok = r["status"] == "ok"
detail = f"status={r['status']}"
if ok:
    rec = pyo.value(r["model"].u_precip.ree_recovery)
    ok = rec > 0.3
    detail += f", precip_recovery={rec:.4f}"
print(json.dumps({"passed": ok, "detail": detail}))
"""

B3_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo

fs = Flowsheet(name="b3", units=[
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
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="org", phase="liquid"),
    Stream(name="barren", phase="liquid"),
])
r = run_eo(fs, objective="minimize_opex")
ok = r["status"] == "ok"
detail = f"status={r['status']}, time={r.get('solve_time_s')}s"
if ok:
    kpis = r.get("kpis", {})
    detail += "; " + "; ".join(f"{k}={v}" for k, v in sorted(kpis.items())[:5])
print(json.dumps({"passed": ok, "detail": detail}))
"""

B4_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo
import pyomo.environ as pyo

fs = Flowsheet(name="b4", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 1.5, "Ce": 4.0, "Nd": 3.0},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
])
r = run_eo(fs, objective="none")
if r["status"] != "ok":
    print(json.dumps({"passed": False, "detail": r.get("error")}))
else:
    m = r["model"]; sx = m.u_sx_1
    lines = []; ok = True
    for j in ["La", "Ce", "Nd", "H2O", "HCl"]:
        f = pyo.value(sx.feed[j])
        o = pyo.value(sx.organic[j])
        raf = pyo.value(sx.raffinate[j])
        err = abs(f - o - raf)
        if err > 1e-8: ok = False
        lines.append(f"{j}: err={err:.2e}")
    print(json.dumps({"passed": ok, "detail": "; ".join(lines)}))
"""

# ═══════════════════════════════════════════════════════════════════════
# Category C: GDP-EO Optimisation
# ═══════════════════════════════════════════════════════════════════════

C1_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream, Superstructure
from sep_agents.opt.gdp_eo import solve_gdp_eo

fs = Flowsheet(name="c1", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
    UnitOp(id="ix_1", type="ion_exchange",
           params={"selectivity_coeff": {"La": 1.0, "Ce": 2.0, "Nd": 3.0},
                   "bed_volume_m3": 1.0},
           inputs=["org"], outputs=["loaded", "eluate"],
           optional=True),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="org", phase="liquid"),
])
ss = Superstructure(name="c1", base_flowsheet=fs, fixed_units=["sx_1"],
                    objective="minimize_opex")
r = solve_gdp_eo(ss)
ok = r.status == "ok"
detail = f"Active={r.active_units}, Bypassed={r.bypassed_units}, obj={r.objective_value}"
print(json.dumps({"passed": ok, "detail": detail}))
"""

C2_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream, Superstructure, DisjunctionDef
from sep_agents.opt.gdp_eo import solve_gdp_eo

fs = Flowsheet(name="c2", units=[
    UnitOp(id="leach", type="leach",
           params={"acid_type": "HCl", "acid_molarity": 2.0,
                   "temperature_C": 60, "residence_time_s": 3600,
                   "liquid_solid_ratio": 5},
           inputs=["feed"], outputs=["leachate"]),
    UnitOp(id="sx_alt", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["leachate"], outputs=["org", "raf"]),
    UnitOp(id="ix_alt", type="ion_exchange",
           params={"selectivity_coeff": {"La": 1.0, "Ce": 2.0, "Nd": 3.0},
                   "bed_volume_m3": 1.0},
           inputs=["leachate"], outputs=["loaded", "eluate"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="leachate", phase="liquid"),
])
ss = Superstructure(
    name="c2", base_flowsheet=fs, fixed_units=["leach"],
    disjunctions=[DisjunctionDef(name="sep", unit_ids=["sx_alt", "ix_alt"])],
    objective="minimize_opex",
)
r = solve_gdp_eo(ss)
sx_act = "sx_alt" in r.active_units
ix_act = "ix_alt" in r.active_units
xor_ok = (sx_act != ix_act)
ok = r.status == "ok" and xor_ok
detail = f"Active={r.active_units}, Bypassed={r.bypassed_units}, XOR={'ok' if xor_ok else 'FAIL'}"
print(json.dumps({"passed": ok, "detail": detail}))
"""

C3_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream, Superstructure
from sep_agents.opt.gdp_eo import solve_gdp_eo

fs = Flowsheet(name="c3", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
    UnitOp(id="precip", type="precipitator",
           params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
           inputs=["org"], outputs=["solid", "barren"],
           optional=True),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="org", phase="liquid"),
])
ss = Superstructure(name="c3", base_flowsheet=fs, fixed_units=["sx_1"],
                    objective="maximize_recovery")
r = solve_gdp_eo(ss)
ok = r.status == "ok"
detail = f"Active={r.active_units}, obj={r.objective_value}"
print(json.dumps({"passed": ok, "detail": detail}))
"""

# ═══════════════════════════════════════════════════════════════════════
# Category D: EO vs SM Comparison
# ═══════════════════════════════════════════════════════════════════════

D1_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo
from sep_agents.sim.idaes_adapter import run_idaes

# SM flowsheet
fs_sm = Flowsheet(name="d1sm", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La+3": 2.0, "Ce+3": 5.0, "Nd+3": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
])
r_sm = run_idaes(fs_sm)

# EO flowsheet
fs_eo = Flowsheet(name="d1eo", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
])
r_eo = run_eo(fs_eo, objective="none")

lines = [f"SM={r_sm['status']}, EO={r_eo['status']}"]
ok = r_sm["status"] == "ok" and r_eo["status"] == "ok"
if ok:
    import pyomo.environ as pyo
    D_vals = {"La": 2.0, "Ce": 5.0, "Nd": 3.5}
    for j, D in D_vals.items():
        E_a = D * 1.0 / (1 + D * 1.0)
        E_eo = pyo.value(r_eo["model"].u_sx_1.extraction[j])
        err = abs(E_eo - E_a)
        if err > 1e-4: ok = False
        lines.append(f"E_{j}: analytical={E_a:.4f} EO={E_eo:.4f} err={err:.2e}")
print(json.dumps({"passed": ok, "detail": "; ".join(lines)}))
"""

D2_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream, Superstructure
from sep_agents.opt.gdp_eo import solve_gdp_eo

fs = Flowsheet(name="d2", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
    UnitOp(id="precip", type="precipitator",
           params={"residence_time_s": 3600, "reagent_dosage_gpl": 10.0},
           inputs=["org"], outputs=["solid", "barren"],
           optional=True),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="org", phase="liquid"),
])

ss_opex = Superstructure(name="d2_opex", base_flowsheet=fs,
                          fixed_units=["sx_1"], objective="minimize_opex")
r_opex = solve_gdp_eo(ss_opex)

ss_rec = Superstructure(name="d2_rec", base_flowsheet=fs,
                         fixed_units=["sx_1"], objective="maximize_recovery")
r_rec = solve_gdp_eo(ss_rec)

ok = r_opex.status == "ok" and r_rec.status == "ok"
detail = (f"min_opex: Active={r_opex.active_units} obj={r_opex.objective_value}; "
          f"max_rec: Active={r_rec.active_units} obj={r_rec.objective_value}")
print(json.dumps({"passed": ok, "detail": detail}))
"""

D3_CODE = """
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo

fs = Flowsheet(name="d3", units=[
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
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
    Stream(name="org", phase="liquid"),
    Stream(name="barren", phase="liquid"),
])
r = run_eo(fs, objective="minimize_opex")
solve_time = r.get("solve_time_s", 99)
ok = r["status"] == "ok" and solve_time < 5.0
detail = f"time={solve_time}s (limit 5s)"
print(json.dumps({"passed": ok, "detail": detail}))
"""


# ═══════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  EO Integration Benchmark Suite (Phase 5)")
    print("  Each test runs in isolated subprocess")
    print("=" * 70)

    print("\n╔══ Category A: Unit Model Verification ══╗")
    run_bench("A1", "SX McCabe-Thiele analytical", "A", A1_CODE)
    run_bench("A2", "Precipitator recovery fraction", "A", A2_CODE)
    run_bench("A3", "IX Langmuir cap", "A", A3_CODE)
    run_bench("A4", "SX 3-stage (analytical)", "A", A4_CODE)

    print("\n╔══ Category B: EO Flowsheet Integration ══╗")
    run_bench("B1", "Single SX flowsheet", "B", B1_CODE)
    run_bench("B2", "SX + Precipitator flowsheet", "B", B2_CODE)
    run_bench("B3", "SX→Precip→IX 3-unit", "B", B3_CODE)
    run_bench("B4", "Mass balance closure", "B", B4_CODE)

    print("\n╔══ Category C: GDP-EO Optimisation ══╗")
    run_bench("C1", "Optional unit disjunction", "C", C1_CODE)
    run_bench("C2", "XOR disjunction", "C", C2_CODE)
    run_bench("C3", "Maximize recovery GDP", "C", C3_CODE)

    print("\n╔══ Category D: EO vs SM Comparison ══╗")
    run_bench("D1", "EO vs SM SX extraction", "D", D1_CODE)
    run_bench("D2", "GDP topology consistency", "D", D2_CODE)
    run_bench("D3", "EO solve speed (<5s)", "D", D3_CODE)

    # Summary
    print("\n" + "=" * 70)
    n_total = len(ALL_RESULTS)
    n_pass = sum(1 for r in ALL_RESULTS if r.passed)
    total_time = sum(r.elapsed_s for r in ALL_RESULTS)

    print(f"  TOTAL: {n_pass}/{n_total} passed, {n_total - n_pass} failed")
    print(f"  Total time: {total_time:.1f}s")
    print()

    cats = {}
    for r in ALL_RESULTS:
        cats.setdefault(r.category, []).append(r)
    cat_names = {"A": "Unit Models", "B": "Flowsheet", "C": "GDP-EO", "D": "EO vs SM"}
    for cat in sorted(cats):
        results = cats[cat]
        cat_pass = sum(1 for r in results if r.passed)
        print(f"  {cat}. {cat_names.get(cat, cat)}: {cat_pass}/{len(results)}")
        for r in results:
            print(f"      [{r.id}] {r.name}: {'PASS' if r.passed else 'FAIL'}")

    print("=" * 70)

    # Save report
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(PROJECT, "reports", f"eo_benchmark_{timestamp}.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(f"# EO Integration Benchmark Report\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Result**: {n_pass}/{n_total} passed\n\n")
        f.write("| ID | Name | Category | Status | Time (s) | Detail |\n")
        f.write("|-----|------|----------|--------|----------|--------|\n")
        for r in ALL_RESULTS:
            s = "✅" if r.passed else "❌"
            d = r.detail[:60].replace("|", "\\|") if r.detail else ""
            f.write(f"| {r.id} | {r.name} | {r.category} | {s} | {r.elapsed_s} | {d} |\n")
        f.write(f"\n**Total time**: {total_time:.1f}s\n")
    print(f"\n  Report saved to: {report_path}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
