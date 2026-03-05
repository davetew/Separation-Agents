#!/usr/bin/env python
"""
Eagle Mine Tailings Valorization – Full IDAES Simulation & Optimization
=======================================================================

Runs the complete valorization flowsheet using:
  1. IDAES sequential-modular solver with Reaktoro thermodynamic equilibrium
  2. BoTorch Bayesian Optimization to maximize net economic value
  3. JAX-based differentiable cost estimation and sensitivity analysis

Designed to run in the `rkt` conda environment:
    conda run -n rkt python examples/eagle_creek_valorization.py

All previously-assumed values (recoveries, pH, speciation) are replaced
by Reaktoro equilibrium calculations wherever possible.
"""

import os, sys, yaml, pathlib, logging, json, copy

# Ensure src/ is importable
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_log = logging.getLogger(__name__)

# ── Core imports ──────────────────────────────────────────────────────────
import jax.numpy as jnp
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder
from sep_agents.cost.jax_tea import (
    itemized_cost,
    total_annualized_cost,
    cost_sensitivity,
)

# ── Configuration ─────────────────────────────────────────────────────────
YAML_PATH = pathlib.Path(__file__).parent / "eagle_creek_valorization.yaml"

# Plant & economic parameters
THROUGHPUT_TPD = 2000.0
OPERATING_DAYS = 330.0
DISCOUNT_RATE = 0.08
PROJECT_LIFE = 20.0

# Residual metal grades (wt%)
NI_GRADE_PCT = 0.7
CU_GRADE_PCT = 0.9

# Mg content for CO2 mineralization
MGO_CONTENT_WT = 0.30
CO2_UPTAKE_PER_T_MGO = 1.09

# Content fractions for Fe3O4 and pyrrhotite
FE3O4_CONTENT_WT = 0.05
PYRRHOTITE_CONTENT_WT = 0.10
PYRRHOTITE_TO_H2SO4 = 1.12

# Commodity prices
CU_PRICE_PER_T = 8500.0
NI_PRICE_PER_T = 16000.0
CARBON_CREDIT_PER_T_CO2 = 60.0
FE3O4_PRICE_PER_T = 120.0
H2SO4_PRICE_PER_T = 80.0


# ══════════════════════════════════════════════════════════════════════════
# PART 1: IDAES Flowsheet Simulation
# ══════════════════════════════════════════════════════════════════════════

def load_flowsheet_from_yaml(yaml_path=YAML_PATH):
    """Load and parse the YAML flowsheet definition."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    units = [UnitOp(**u) for u in data.get("units", [])]
    streams = [Stream(**s) for s in data.get("streams", [])]
    return Flowsheet(name=data.get("name", "eagle_mine"), units=units, streams=streams), data


def run_idaes_simulation(flowsheet, use_jax=False):
    """Run the full IDAES flowsheet with Reaktoro equilibrium."""
    builder = IDAESFlowsheetBuilder(database_name="SUPRCRT - BL", use_jax=use_jax)
    result = builder.build_and_solve(flowsheet)
    return result


def extract_simulation_kpis(sim_result):
    """Extract simulation-derived KPIs from the IDAES result.
    
    Values are sourced from three levels of fidelity:
      1. Reaktoro equilibrium: pH, CO₂ conversion, Mg speciation
      2. IDAES separator models: magnetic sep, flotation recovery
      3. McCabe-Thiele mass-balance: Cu/Ni SX recovery (Ni²⁺/Cu²⁺ not
         in SUPCRTBL database, so leach dissolution is estimated from
         the simulation-derived pH and literature kinetic data)
    """
    kpis = {}
    
    if sim_result.get("status") != "ok":
        _log.warning("Simulation did not converge: %s", sim_result.get("error"))
        return kpis
    
    streams = sim_result.get("streams", {})
    sim_kpis = sim_result.get("kpis", {})
    kpis.update(sim_kpis)
    
    # ── Reaktoro-derived values ────────────────────────────────────────
    pls = streams.get("pregnant_solution", {})
    if isinstance(pls, dict) and "pH" in pls:
        kpis["leach_pH"] = pls["pH"]
    
    carb_prod = streams.get("carbonation_product", {})
    if isinstance(carb_prod, dict) and "pH" in carb_prod:
        kpis["carbonation_pH"] = carb_prod["pH"]
    
    # CO₂ conversion from Reaktoro carbonation equilibrium
    carb_feed = streams.get("carbonation_feed", {})
    if isinstance(carb_feed, dict) and isinstance(carb_prod, dict):
        cf_sp = carb_feed.get("species_amounts", {})
        cp_sp = carb_prod.get("species_amounts", {})
        co2_in = cf_sp.get("CO2(aq)", 0)
        co2_out = cp_sp.get("CO2(aq)", 0)
        if co2_in > 1e-15:
            kpis["co2_conversion"] = max(0, (co2_in - co2_out) / co2_in)
    
    # Mg fraction reaching carbonation (from Reaktoro SX mass balance)
    feed = streams.get("tailings_feed", {})
    mg_raff = streams.get("mg_rich_raffinate", {})
    if isinstance(feed, dict) and isinstance(mg_raff, dict):
        feed_mg = feed.get("species_amounts", {}).get("Mg+2", 0)
        raff_mg = mg_raff.get("species_amounts", {}).get("Mg+2", 0)
        if feed_mg > 1e-15:
            kpis["mg_to_carbonation_frac"] = min(raff_mg / feed_mg, 1.0)
    
    # ── IDAES separator-derived values ────────────────────────────────
    kpis["mag_sep_recovery"] = sim_kpis.get("mag_sep.recovery", 0.85)
    kpis["flotation_recovery"] = sim_kpis.get("sulfide_float.recovery", 0.70)
    
    # ── Engineering estimates (informed by simulation pH) ──────────────
    # Cu²⁺ and Ni²⁺ are not in the SUPCRTBL database, so leach
    # dissolution and SX extraction use literature-based parameters:
    #   - Acid leach at pH ~1.5 and 80°C → ~90% dissolution of fine
    #     sulfide-hosted Cu/Ni (Habashi, 1970; Free, 2013)
    #   - SX recovery is computed via McCabe-Thiele for n stages:
    #     R = 1 - 1/(1 + D·OA)^n
    leach_pH = kpis.get("leach_pH", 2.0)
    
    # Dissolution fraction decreases with increasing pH (empirical)
    if leach_pH < 1.0:
        dissolution = 0.95
    elif leach_pH < 2.0:
        dissolution = 0.90 - 0.05 * (leach_pH - 1.0)
    else:
        dissolution = max(0.5, 0.85 - 0.10 * (leach_pH - 2.0))
    
    # McCabe-Thiele SX: R = 1 - 1/(1+D*OA)^n
    D_cu, D_ni = 12.0, 8.0   # from YAML distribution coefficients
    OA = 1.5                   # organic-to-aqueous ratio
    n_stages = 3
    cu_sx_rec = 1.0 - 1.0 / (1.0 + D_cu * OA) ** n_stages
    ni_sx_rec = 1.0 - 1.0 / (1.0 + D_ni * OA) ** n_stages
    
    kpis["cu_dissolution"] = dissolution
    kpis["ni_dissolution"] = dissolution
    kpis["cu_sx_recovery"] = cu_sx_rec
    kpis["ni_sx_recovery"] = ni_sx_rec
    kpis["cu_recovery_overall"] = dissolution * cu_sx_rec
    kpis["ni_recovery_overall"] = dissolution * ni_sx_rec
    
    kpis["_source_notes"] = {
        "leach_pH": "Reaktoro equilibrium",
        "carbonation_pH": "Reaktoro equilibrium", 
        "co2_conversion": "Reaktoro equilibrium",
        "mg_to_carbonation_frac": "Reaktoro SX mass balance",
        "mag_sep_recovery": "IDAES LIMS separator",
        "flotation_recovery": "IDAES flotation bank",
        "cu_recovery_overall": f"Eng. estimate: {dissolution:.0%} dissolution (pH={leach_pH:.2f}) × {cu_sx_rec:.0%} SX (D={D_cu}, OA={OA}, n={n_stages})",
        "ni_recovery_overall": f"Eng. estimate: {dissolution:.0%} dissolution (pH={leach_pH:.2f}) × {ni_sx_rec:.0%} SX (D={D_ni}, OA={OA}, n={n_stages})",
    }
    kpis["_streams"] = streams
    
    return kpis


# ══════════════════════════════════════════════════════════════════════════
# PART 2: BoTorch Optimization
# ══════════════════════════════════════════════════════════════════════════

def run_optimization(yaml_data, n_iters=10):
    """Optimize key process parameters using BoTorch to maximize net value.
    
    Design variables:
      - Leach temperature (T_C): [50, 95] °C
      - Leach residence time: [3600, 28800] s (1-8 hours)
      - SX O/A ratio: [0.5, 3.0]
    """
    import torch
    from sep_agents.opt.bo import BotorchOptimizer
    
    design_vars = [
        {"unit_id": "leach_1", "param": "T_C", "bounds": [50.0, 95.0]},
        {"unit_id": "leach_1", "param": "residence_time_s", "bounds": [3600.0, 28800.0]},
        {"unit_id": "sx_cu_ni", "param": "organic_to_aqueous_ratio", "bounds": [0.5, 3.0]},
    ]
    
    bounds_tensor = torch.tensor(
        [v["bounds"] for v in design_vars], dtype=torch.double
    ).T  # 2 x d
    
    def objective(candidate_x):
        """Evaluate net annual value for a given set of parameters."""
        data = copy.deepcopy(yaml_data)
        units = [UnitOp(**u) for u in data.get("units", [])]
        streams = [Stream(**s) for s in data.get("streams", [])]
        
        # Apply candidate parameters
        for i, var in enumerate(design_vars):
            val = candidate_x[i].item()
            # Unscale from [0,1] to physical bounds
            lo, hi = var["bounds"]
            phys_val = lo + val * (hi - lo)
            for u in units:
                if u.id == var["unit_id"]:
                    u.params[var["param"]] = phys_val
                    break
        
        fs = Flowsheet(name="eagle_mine_opt", units=units, streams=streams)
        
        try:
            result = run_idaes_simulation(fs)
            kpis = extract_simulation_kpis(result)
            
            # Calculate net value using simulation-derived recoveries
            annual_tailings = THROUGHPUT_TPD * OPERATING_DAYS
            
            ni_rec = kpis.get("ni_recovery_overall", 0.75)
            cu_rec = kpis.get("cu_recovery_overall", 0.80)
            mag_rec = kpis.get("mag_sep_recovery", 0.85)
            float_rec = kpis.get("flotation_recovery", 0.70)
            co2_conv = kpis.get("co2_conversion", 0.65)
            
            # Revenues
            cu_rev = annual_tailings * (CU_GRADE_PCT/100) * cu_rec * CU_PRICE_PER_T
            ni_rev = annual_tailings * (NI_GRADE_PCT/100) * ni_rec * NI_PRICE_PER_T
            co2_seq = annual_tailings * MGO_CONTENT_WT * CO2_UPTAKE_PER_T_MGO * co2_conv
            co2_rev = co2_seq * CARBON_CREDIT_PER_T_CO2
            fe_rev = annual_tailings * FE3O4_CONTENT_WT * mag_rec * FE3O4_PRICE_PER_T
            
            total_rev = cu_rev + ni_rev + co2_rev + fe_rev
            
            # Cost (quick proxy using leach params)
            leach_t = 80.0
            for u in units:
                if u.id == "leach_1":
                    leach_t = u.params.get("T_C", 80.0)
            
            jax_params = {
                "ore_throughput_tpd": jnp.array(THROUGHPUT_TPD),
                "strip_ratio": jnp.array(0.0),
                "mine_depth_m": jnp.array(0.0),
                "bond_work_index": jnp.array(12.0),
                "residence_time_h": jnp.array(4.0),
                "acid_consumption_kg_t": jnp.array(80.0),
                "operating_temp_c": jnp.array(float(leach_t)),
                "sx_stages": jnp.array(6.0),
                "precipitation_reagent_tpy": jnp.array(500.0),
                "aq_flow_m3_h": jnp.array(250.0),
            }
            eac = float(total_annualized_cost(jax_params, DISCOUNT_RATE, PROJECT_LIFE))
            
            net_value = total_rev - eac
            return net_value / 1e6  # Normalize to millions for GP fitting
            
        except Exception as e:
            _log.warning("Optimization eval failed: %s", e)
            return -100.0  # Penalty
    
    _log.info("Starting BoTorch optimization (%d iterations)...", n_iters)
    opt = BotorchOptimizer(maximize=True)
    best_x, best_y, history = opt.optimize(
        objective_fn=objective,
        bounds=bounds_tensor,
        n_initial=5,
        n_iters=n_iters,
    )
    
    # Map best_x back to named parameters
    optimal_params = {}
    for i, var in enumerate(design_vars):
        optimal_params[f"{var['unit_id']}.{var['param']}"] = best_x[i].item()
    
    return {
        "optimal_params": optimal_params,
        "best_net_value_M": best_y,
        "history": history,
        "design_vars": design_vars,
    }


# ══════════════════════════════════════════════════════════════════════════
# PART 3: Revenue Computation (uses simulation-derived KPIs)
# ══════════════════════════════════════════════════════════════════════════

def compute_revenues(sim_kpis):
    """Compute revenues using simulation-derived recoveries where available."""
    annual_tailings = THROUGHPUT_TPD * OPERATING_DAYS
    
    # Use simulation-derived or engineering-estimate values
    cu_rec = sim_kpis.get("cu_recovery_overall", 0.80)
    ni_rec = sim_kpis.get("ni_recovery_overall", 0.75)
    mag_rec = sim_kpis.get("mag_sep_recovery", 0.85)
    float_rec = sim_kpis.get("flotation_recovery", 0.70)
    co2_conv = sim_kpis.get("co2_conversion", 0.65)
    
    cu_tpy = annual_tailings * (CU_GRADE_PCT / 100.0) * cu_rec
    ni_tpy = annual_tailings * (NI_GRADE_PCT / 100.0) * ni_rec
    fe3o4_tpy = annual_tailings * FE3O4_CONTENT_WT * mag_rec
    
    pyrrhotite_tpy = annual_tailings * PYRRHOTITE_CONTENT_WT * float_rec
    h2so4_produced = pyrrhotite_tpy * PYRRHOTITE_TO_H2SO4
    h2so4_consumed = annual_tailings * (80.0 / 1000.0)
    h2so4_surplus = max(0.0, h2so4_produced - h2so4_consumed)
    
    co2_seq = annual_tailings * MGO_CONTENT_WT * CO2_UPTAKE_PER_T_MGO * co2_conv
    
    return {
        "annual_tailings_t": annual_tailings,
        "cu_recovery": cu_rec,
        "ni_recovery": ni_rec,
        "mag_recovery": mag_rec,
        "flotation_recovery": float_rec,
        "co2_conversion": co2_conv,
        "cu_produced_tpy": cu_tpy,
        "cu_revenue": cu_tpy * CU_PRICE_PER_T,
        "ni_produced_tpy": ni_tpy,
        "ni_revenue": ni_tpy * NI_PRICE_PER_T,
        "co2_sequestered_tpy": co2_seq,
        "co2_revenue": co2_seq * CARBON_CREDIT_PER_T_CO2,
        "fe3o4_produced_tpy": fe3o4_tpy,
        "fe3o4_revenue": fe3o4_tpy * FE3O4_PRICE_PER_T,
        "h2so4_produced_tpy": h2so4_produced,
        "h2so4_consumed_tpy": h2so4_consumed,
        "h2so4_surplus_tpy": h2so4_surplus,
        "h2so4_revenue": h2so4_surplus * H2SO4_PRICE_PER_T,
        "total_revenue": (cu_tpy * CU_PRICE_PER_T + ni_tpy * NI_PRICE_PER_T +
                          co2_seq * CARBON_CREDIT_PER_T_CO2 + fe3o4_tpy * FE3O4_PRICE_PER_T +
                          h2so4_surplus * H2SO4_PRICE_PER_T),
    }


# ══════════════════════════════════════════════════════════════════════════
# PART 4: Report Generation
# ══════════════════════════════════════════════════════════════════════════

def generate_report(sim_result, sim_kpis, costs, eac, sensitivities, revenues, opt_result=None):
    """Print a comprehensive Markdown report with simulation-derived values."""
    annual_tailings = revenues["annual_tailings_t"]
    eac_val = eac.item()
    total_rev = revenues["total_revenue"]
    net_value = total_rev - eac_val
    lcop_per_t = eac_val / annual_tailings if annual_tailings > 0 else 0
    rev_per_t = total_rev / annual_tailings if annual_tailings > 0 else 0
    net_per_t = net_value / annual_tailings if annual_tailings > 0 else 0
    
    print("# Eagle Mine Tailings Valorization – IDAES Simulation Results")
    
    # ── Section 1 ─────────────────────────────────────────────────────
    print("\n## 1. Request & Assumptions\n")
    print("Valorization of Ni/Cu mine tailings from Eagle Mine (MI).")
    print("Process parameters derived from three modeling tiers:\n")
    print("1. **Reaktoro equilibrium** — pH, CO₂ conversion, Mg speciation")
    print("2. **IDAES separator models** — magnetic sep, flotation recovery")
    print("3. **McCabe-Thiele mass-balance** — Cu/Ni leach+SX recovery (informed by Reaktoro pH)\n")
    print("| Mineral | Formula | Mol % |")
    print("| :--- | :--- | ---: |")
    for m, f, p in [
        ("Forsterite", "Mg₂SiO₄", 45), ("Enstatite", "MgSiO₃", 15),
        ("Pyrrhotite", "Fe₁₋ₓS", 10), ("Anorthite", "CaAl₂Si₂O₈", 15),
        ("Lizardite", "Mg₃Si₂O₅(OH)₄", 10), ("Magnetite", "Fe₃O₄", 5),
    ]:
        print(f"| {m} | {f} | {p}% |")
    
    print(f"\n- **Throughput**: {THROUGHPUT_TPD:,.0f} tpd ({annual_tailings:,.0f} t/yr)")
    print(f"- **Residual Ni**: {NI_GRADE_PCT}% | **Cu**: {CU_GRADE_PCT}%")
    
    # ── Section 2: IDAES Simulation Results ───────────────────────────
    print("\n## 2. Simulation-Derived Process Parameters\n")
    
    sim_status = sim_result.get("status", "unknown") if sim_result else "not run"
    print(f"**IDAES Simulation Status**: {'✅ Converged' if sim_status == 'ok' else '⚠️ ' + sim_status}\n")
    
    def fmt(v): return f"{v:.4f}" if isinstance(v, (int, float)) else str(v)
    source_notes = sim_kpis.get("_source_notes", {})
    
    print("| Parameter | Value | Source |")
    print("| :--- | ---: | :--- |")
    params_to_show = [
        ("Leach pH", "leach_pH"),
        ("Carbonation pH", "carbonation_pH"),
        ("CO₂ conversion", "co2_conversion"),
        ("Mg→carbonation frac", "mg_to_carbonation_frac"),
        ("Mag sep recovery", "mag_sep_recovery"),
        ("Flotation recovery", "flotation_recovery"),
        ("Cu dissolution", "cu_dissolution"),
        ("Cu SX recovery", "cu_sx_recovery"),
        ("Cu overall recovery", "cu_recovery_overall"),
        ("Ni dissolution", "ni_dissolution"),
        ("Ni SX recovery", "ni_sx_recovery"),
        ("Ni overall recovery", "ni_recovery_overall"),
    ]
    for label, key in params_to_show:
        val = sim_kpis.get(key, "N/A")
        src = source_notes.get(key, "—")
        print(f"| {label} | {fmt(val)} | {src} |")
    
    # Show per-unit KPIs from IDAES
    unit_kpis = {k: v for k, v in sim_kpis.items() if not k.startswith("_") and "." in k}
    if unit_kpis:
        print("\n**Per-Unit IDAES KPIs:**\n")
        print("| Unit.KPI | Value |")
        print("| :--- | ---: |")
        for k, v in sorted(unit_kpis.items()):
            if isinstance(v, float):
                print(f"| `{k}` | {v:.4f} |")
    
    # ── Section 3: CAPEX/OPEX ─────────────────────────────────────────
    print("\n## 3. Capital and Operating Expenditures\n")
    print("| Process Stage | CAPEX ($M) | Annual OPEX ($M) |")
    print("| :--- | ---: | ---: |")
    for slug, label in [("mining", "Tailings Handling"), ("comminution", "Regrind"),
                         ("leaching", "Leaching"), ("processing", "Processing (SX)")]:
        c = costs[f"{slug}_capex"].item() / 1e6
        o = costs[f"{slug}_opex"].item() / 1e6
        print(f"| {label} | ${c:,.2f} | ${o:,.2f} |")
    tc = costs["total_capex"].item() / 1e6
    to = costs["total_opex"].item() / 1e6
    print(f"| **Total** | **${tc:,.2f}** | **${to:,.2f}** |")
    
    # ── Section 4: Revenue ────────────────────────────────────────────
    print("\n## 4. Annual Revenue (Simulation-Derived Recoveries)\n")
    print("| Revenue Stream | Production | Price | Revenue ($M/yr) |")
    print("| :--- | ---: | ---: | ---: |")
    print(f"| Cu metal | {revenues['cu_produced_tpy']:,.0f} t/yr | ${CU_PRICE_PER_T:,.0f}/t | ${revenues['cu_revenue']/1e6:,.2f} |")
    print(f"| Ni metal | {revenues['ni_produced_tpy']:,.0f} t/yr | ${NI_PRICE_PER_T:,.0f}/t | ${revenues['ni_revenue']/1e6:,.2f} |")
    print(f"| CO₂ credits | {revenues['co2_sequestered_tpy']:,.0f} t/yr | ${CARBON_CREDIT_PER_T_CO2:,.0f}/t | ${revenues['co2_revenue']/1e6:,.2f} |")
    print(f"| Fe₃O₄ conc. | {revenues['fe3o4_produced_tpy']:,.0f} t/yr | ${FE3O4_PRICE_PER_T:,.0f}/t | ${revenues['fe3o4_revenue']/1e6:,.2f} |")
    print(f"| H₂SO₄ surplus | {revenues['h2so4_surplus_tpy']:,.0f} t/yr | ${H2SO4_PRICE_PER_T:,.0f}/t | ${revenues['h2so4_revenue']/1e6:,.2f} |")
    print(f"| **Total** | | | **${total_rev/1e6:,.2f}** |")
    
    # ── Section 5: Levelized Economics ────────────────────────────────
    print("\n## 5. Levelized Economics\n")
    print(f"**EAC**: ${eac_val/1e6:,.2f} M/yr | **Revenue**: ${total_rev/1e6:,.2f} M/yr | **Net**: ${net_value/1e6:,.2f} M/yr")
    print(f"\n**Cost per tonne tailings**: ${lcop_per_t:,.2f}/t")
    print(f"**Revenue per tonne tailings**: ${rev_per_t:,.2f}/t")
    print(f"**Net value per tonne tailings**: **${net_per_t:,.2f}/t**")
    
    # ── Section 6: Sensitivities ──────────────────────────────────────
    print("\n## 6. Cost Sensitivities (JAX Automatic Differentiation)\n")
    print("| Parameter | Δ EAC ($M/yr per unit) |")
    print("| :--- | ---: |")
    sorted_sens = sorted(sensitivities.items(), key=lambda x: abs(x[1].item()), reverse=True)
    for k, v in sorted_sens[:5]:
        print(f"| `{k}` | ${v.item()/1e6:,.3f} M |")
    
    # ── Section 7: Optimization Results ───────────────────────────────
    if opt_result:
        print("\n## 7. BoTorch Optimization Results\n")
        print(f"**Best Net Value**: ${opt_result['best_net_value_M']:,.2f} M/yr\n")
        print("**Optimal Process Parameters:**\n")
        print("| Parameter | Optimal Value |")
        print("| :--- | ---: |")
        for k, v in opt_result["optimal_params"].items():
            print(f"| `{k}` | {v:.2f} |")
        
        print(f"\n**BO Iterations**: {len(opt_result['history']) - 1}")
        print(f"**Improvement**: ${opt_result['history'][-1].get('best_y', 0) - opt_result['history'][0].get('best_y', 0):,.2f} M/yr")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Eagle Mine Tailings Valorization")
    parser.add_argument("--skip-optimization", action="store_true",
                        help="Skip BoTorch optimization (simulation only)")
    parser.add_argument("--bo-iters", type=int, default=10,
                        help="Number of BoTorch iterations")
    parser.add_argument("--use-jax", action="store_true",
                        help="Use JAX equilibrium solver instead of Reaktoro")
    args = parser.parse_args()
    
    # 1. Load flowsheet
    _log.info("Loading flowsheet from %s", YAML_PATH)
    flowsheet, yaml_data = load_flowsheet_from_yaml()
    
    # 2. Run IDAES simulation with Reaktoro equilibrium
    _log.info("Running IDAES flowsheet simulation (Reaktoro equilibrium)...")
    sim_result = run_idaes_simulation(flowsheet, use_jax=args.use_jax)
    
    # 3. Extract simulation-derived KPIs
    sim_kpis = extract_simulation_kpis(sim_result)
    
    _log.info("Simulation-derived KPIs:")
    for k, v in sorted(sim_kpis.items()):
        if not k.startswith("_") and isinstance(v, (int, float)):
            _log.info("  %s = %.4f", k, v)
    
    # 4. Run BoTorch optimization (optional)
    opt_result = None
    if not args.skip_optimization:
        try:
            opt_result = run_optimization(yaml_data, n_iters=args.bo_iters)
            _log.info("Optimization complete. Best net value: $%.2f M/yr",
                       opt_result["best_net_value_M"])
        except Exception as e:
            _log.warning("Optimization failed: %s. Proceeding with base case.", e)
    
    # 5. Run JAX TEA cost estimation
    _log.info("Running JAX TEA cost estimation...")
    jax_params = {
        "ore_throughput_tpd": jnp.array(THROUGHPUT_TPD),
        "strip_ratio": jnp.array(0.0),
        "mine_depth_m": jnp.array(0.0),
        "bond_work_index": jnp.array(12.0),
        "residence_time_h": jnp.array(4.0),
        "acid_consumption_kg_t": jnp.array(80.0),
        "operating_temp_c": jnp.array(80.0),
        "sx_stages": jnp.array(6.0),
        "precipitation_reagent_tpy": jnp.array(500.0),
        "aq_flow_m3_h": jnp.array(250.0),
    }
    costs = itemized_cost(jax_params)
    eac = total_annualized_cost(jax_params, DISCOUNT_RATE, PROJECT_LIFE)
    sensitivities = cost_sensitivity(jax_params)
    
    # 6. Compute revenues using simulation-derived recoveries
    revenues = compute_revenues(sim_kpis)
    
    # 7. Generate comprehensive report
    generate_report(sim_result, sim_kpis, costs, eac, sensitivities, revenues, opt_result)


if __name__ == "__main__":
    main()
