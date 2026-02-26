"""
REE Separation Process Design — End-to-End Workflow
====================================================
Following the recommended agent prompt:
1. Speciation Analysis of the feed liquor
2. Flowsheet Synthesis (SX + Precipitator)
3. Baseline Simulation via IDAES
4. Bayesian Optimization (minimize OPEX)
5. Final Report
"""
import json
import yaml
import sys
import os
import logging
logging.basicConfig(level=logging.WARNING)

# Ensure sep_agents is importable from the src directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Step 1: Speciation Analysis ──────────────────────────────────────────────
print("=" * 70)
print("STEP 1: Feed Speciation Analysis")
print("=" * 70)

try:
    from sep_agents.properties.ree_databases import REEEquilibriumSolver
    solver = REEEquilibriumSolver(preset="light_ree")
    result = solver.speciate(
        temperature_C=25.0,
        acid_mol={"HCl(aq)": 50.0},
        ree_mol={"Nd+3": 15.0, "Ce+3": 20.0, "La+3": 10.0},
    )
    if result["status"] == "ok":
        print(f"  pH:  {result['pH']:.2f}")
        print(f"  Eh:  {result.get('Eh_mV', 'N/A')}")
        print(f"  REE Distribution:")
        for el, frac in result.get("ree_distribution", {}).items():
            print(f"    {el}: {frac:.4f}")
    else:
        print(f"  Speciation FAILED: {result.get('error')}")
        sys.exit(1)
except Exception as e:
    print(f"  Speciation error: {e}")
    sys.exit(1)

# ── Step 2: Flowsheet Synthesis ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2: Flowsheet Synthesis (SX → Precipitator)")
print("=" * 70)

from sep_agents.dsl.schemas import Flowsheet, Stream, UnitOp

feed = Stream(
    name="feed",
    phase="liquid",
    temperature_K=298.15,
    composition_wt={
        "H2O(aq)": 1000.0,
        "Nd+3": 15.0,
        "Ce+3": 20.0,
        "La+3": 10.0,
        "HCl(aq)": 50.0,
    }
)

sx_unit = UnitOp(
    id="sx_1",
    type="solvent_extraction",
    inputs=["feed"],
    outputs=["org_extract", "aq_raffinate"],
    params={
        "distribution_coeff": {"Nd+3": 5.0, "Ce+3": 2.0, "La+3": 1.0},
        "organic_to_aqueous_ratio": 1.5,
    }
)

precip_unit = UnitOp(
    id="precipitator",
    type="precipitator",
    inputs=["aq_raffinate"],
    outputs=["solid_product", "barren_liquor"],
    params={
        "T_C": 25.0,
        "residence_time_s": 3600.0,
        "reagent_dosage_gpl": 10.0,
    }
)

flowsheet = Flowsheet(name="ree_nd_separation", streams=[feed], units=[sx_unit, precip_unit])
fs_yaml = yaml.dump(flowsheet.model_dump(), sort_keys=False)
print("  Generated YAML flowsheet:")
print("  " + fs_yaml.replace("\n", "\n  "))

# ── Step 3: Baseline Simulation ─────────────────────────────────────────────
print("=" * 70)
print("STEP 3: Baseline IDAES Simulation")
print("=" * 70)

from sep_agents.sim.idaes_adapter import run_idaes

result = run_idaes(flowsheet, database="light_ree")

if result["status"] == "ok":
    print("  Simulation SUCCEEDED")
    print("  KPIs:")
    for k, v in result["kpis"].items():
        print(f"    {k}: {v}")
    baseline_kpis = result["kpis"]
else:
    print(f"  Simulation FAILED: {result.get('error')}")
    # Fallback: try precipitator-only
    print("\n  Retrying with precipitator-only flowsheet...")
    precip_unit_solo = UnitOp(
        id="precipitator",
        type="precipitator",
        inputs=["feed"],
        outputs=["solid_product", "barren_liquor"],
        params={
            "T_C": 25.0,
            "residence_time_s": 3600.0,
            "reagent_dosage_gpl": 10.0,
        }
    )
    flowsheet = Flowsheet(name="ree_nd_precip_only", streams=[feed], units=[precip_unit_solo])
    fs_yaml = yaml.dump(flowsheet.model_dump(), sort_keys=False)
    result = run_idaes(flowsheet, database="light_ree")
    if result["status"] == "ok":
        print("  Precipitator-only simulation SUCCEEDED")
        print("  KPIs:")
        for k, v in result["kpis"].items():
            print(f"    {k}: {v}")
        baseline_kpis = result["kpis"]
    else:
        print(f"  Precipitator-only ALSO FAILED: {result.get('error')}")
        sys.exit(1)

# ── Step 4: Bayesian Optimization ───────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 4: BoTorch Bayesian Optimization (Minimize OPEX)")
print("=" * 70)

import torch
from sep_agents.opt.bo import BotorchOptimizer
import copy

# Define design variables and bounds
design_vars = [
    {"unit_id": "sx_1", "param": "organic_to_aqueous_ratio", "bounds": [0.5, 5.0]},
    {"unit_id": "precipitator", "param": "reagent_dosage_gpl", "bounds": [1.0, 50.0]},
]

bounds_list = [v["bounds"] for v in design_vars]
bounds_tensor = torch.tensor(bounds_list, dtype=torch.double).T  # 2 x d

base_data = yaml.safe_load(fs_yaml)

def flowsheet_objective(candidate_x: torch.Tensor) -> float:
    """Evaluate OPEX for a candidate parameter set (normalized [0,1])."""
    current_data = copy.deepcopy(base_data)
    units = [UnitOp(**u) for u in current_data.get("units", [])]
    streams = [Stream(**s) for s in current_data.get("streams", [])]
    
    # Unscale from [0,1] to physical bounds
    x_phys = bounds_tensor[0] + candidate_x * (bounds_tensor[1] - bounds_tensor[0])
    
    for i, var in enumerate(design_vars):
        for u in units:
            if u.id == var["unit_id"]:
                if u.params is None:
                    u.params = {}
                u.params[var["param"]] = x_phys[i].item()
                break
    
    fs = Flowsheet(name="opt_eval", units=units, streams=streams)
    res = run_idaes(fs, database="light_ree")
    if res["status"] == "ok" and "overall.opex_USD" in res["kpis"]:
        return float(res["kpis"]["overall.opex_USD"])
    return 1e9  # Penalty for failed evals

print(f"  Baseline OPEX: ${baseline_kpis.get('overall.opex_USD', 'N/A')}")
print(f"  Optimizing over: {[v['param'] for v in design_vars]}")
print(f"  Running 3 initial + 5 sequential evaluations...")

opt = BotorchOptimizer(maximize=False)
best_x, best_y, history = opt.optimize(
    objective_fn=flowsheet_objective,
    bounds=bounds_tensor,
    n_initial=3,
    n_iters=5
)

# ── Step 5: Final Report ────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 5: FINAL REPORT")
print("=" * 70)

print(f"\n  Baseline OPEX:    ${baseline_kpis.get('overall.opex_USD', 'N/A')}")
print(f"  Optimized OPEX:   ${best_y:.2f}")
print(f"  Optimal O/A Ratio:     {best_x[0].item():.3f}")
print(f"  Optimal Dosage (gpl):  {best_x[1].item():.3f}")
print(f"\n  Optimization History:")
for h in history:
    print(f"    Iter {h['iter']}: best_y = {h['best_y']:.4f}")

print("\n" + "=" * 70)
print("WORKFLOW COMPLETE")
print("=" * 70)
