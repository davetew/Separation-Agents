import jax
import jax.numpy as jnp
from typing import Dict

from sep_agents.cost.jax_tea import (
    itemized_cost,
    total_annualized_cost,
    cost_sensitivity
)

# Preliminary Economic Assessment (PEA) - Round Top Mountain, Texas
# Focuses on 2019/2012 PEA general parameters.

# - Throughput: ~20,000 tpd
# - Strip ratio: Very low for this ryholite dome, approximated at 0.5
# - Mine depth: Surface/near-surface, approximated at 25m
# - Bond Work Index: Ryholite is relatively hard, approximated at 14.5 kWh/t
# - Leaching: Heap leaching with sulfuric acid (using proxy vat/tank parameters here)
# - Acid Consumption: Moderate to high depending on ryholite type, approximated 40 kg/t
# - Recovery: ~72% (reflected downstream in reduced sizing relative to theoretical max)
def execute_round_top_estimate():
    params = {
        "ore_throughput_tpd": jnp.array(20000.0),
        "strip_ratio": jnp.array(0.5),          
        "mine_depth_m": jnp.array(25.0),
        "bond_work_index": jnp.array(14.5),
        "residence_time_h": jnp.array(72.0),     # Extended due to hard rock / simulated heap timeline
        "acid_consumption_kg_t": jnp.array(40.0), 
        "operating_temp_c": jnp.array(25.0),     # Ambient heap
        "sx_stages": jnp.array(40.0),            # Heavy REE fractionation requires many stages
        "precipitation_reagent_tpy": jnp.array(3500.0), 
        "aq_flow_m3_h": jnp.array(1500.0)        # Large flow handling for 20k tpd
    }
    
    costs = itemized_cost(params)
    eac = total_annualized_cost(params, discount_rate=0.08, lifetime_years=20.0)
    
    # Calculate Levelized Cost of Production
    # Assuming 330 operating days/year
    # Head grade proxy: 600 ppm (0.06%), Recovery: 72%
    operating_days = 330.0
    head_grade_pct = 0.06 / 100.0
    recovery = 0.72
    
    annual_ore_tonnes = params["ore_throughput_tpd"].item() * operating_days
    annual_reo_production_tonnes = annual_ore_tonnes * head_grade_pct * recovery
    annual_reo_production_kg = annual_reo_production_tonnes * 1000.0
    
    annual_material_mined_tonnes = annual_ore_tonnes * (1 + params["strip_ratio"].item())
    
    lcop_per_kg = eac.item() / annual_reo_production_kg if annual_reo_production_kg > 0 else 0
    lcop_per_tonne = eac.item() / annual_reo_production_tonnes if annual_reo_production_tonnes > 0 else 0
    cost_per_mined_tonne = eac.item() / annual_material_mined_tonnes if annual_material_mined_tonnes > 0 else 0
    
    # Format and print Markdown Report
    print("# Techno-Economic Analysis Report: Round Top Mountain (PEA Proxy)")
    print("\n## 1. Request & Major Assumptions\n")
    print("This report estimates the lifecycle costs of the Round Top Mountain ryholite deposit based on generalized PEA parameters.")
    print("\n**Key Assumptions:**")
    print(f"- **Throughput**: {params['ore_throughput_tpd'].item():,.0f} tonnes per day")
    print(f"- **Operating Days**: {operating_days} days/year")
    print(f"- **Strip Ratio**: {params['strip_ratio'].item():.2f}")
    print(f"- **Bond Work Index (Hardness)**: {params['bond_work_index'].item():.2f} kWh/t")
    print(f"- **Estimated REO Head Grade**: {head_grade_pct * 100 * 10000:.0f} ppm")
    print(f"- **Estimated Recovery**: {recovery * 100:.0f}%")
    print(f"- **Annual REO Production**: {annual_reo_production_tonnes:,.0f} tonnes")
    
    print("\n## 2. Capital and Operating Expenditures (Scaled)\n")
    print("| Process Stage | CAPEX ($M) | Annual OPEX ($M) |")
    print("| :--- | :--- | :--- |")
    
    for slug in ["mining", "comminution", "leaching", "processing", "total"]:
        c_cap_m = costs[f"{slug}_capex"].item() / 1e6
        c_op_m = costs[f"{slug}_opex"].item() / 1e6
        stage_name = slug.capitalize()
        if slug == "total":
            print(f"| **{stage_name}** | **${c_cap_m:,.2f}** | **${c_op_m:,.2f}** |")
        else:
            print(f"| {stage_name} | ${c_cap_m:,.2f} | ${c_op_m:,.2f} |")
            
    print("\n## 3. Levelized Cost of Production\n")
    print(f"**Equivalent Annualized Cost (EAC)**: ${eac.item() / 1e6:,.2f} Million / year (8% discount over 20 years)")
    print(f"**Levelized Cost (LCOP)**: **${lcop_per_kg:,.2f} / kg REO** (or ${lcop_per_tonne:,.0f} / tonne REO)")
    print(f"**Cost per Tonne Material Mined**: **${cost_per_mined_tonne:,.2f} / tonne** (ore + waste)")
    
    print("\n## 4. Key Sensitivities\n")
    print("Impact of a 1-unit increase in parameter on the Annualized Cost ($M/year):")
    print("\n| Parameter | Sensitivity ($M / yr) |")
    print("| :--- | :--- |")
    
    sensitivities = cost_sensitivity(params)
    sorted_sensitivities = sorted(sensitivities.items(), key=lambda item: abs(item[1].item()), reverse=True)
    for key, grad_val in sorted_sensitivities[:5]:
        print(f"| `{key}` | ${(grad_val.item() / 1e6):,.2f} M |")

if __name__ == "__main__":
    execute_round_top_estimate()
