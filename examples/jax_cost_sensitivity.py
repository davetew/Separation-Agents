import jax
import jax.numpy as jnp
from typing import Dict

from sep_agents.cost.jax_tea import (
    itemized_cost,
    total_annualized_cost,
    cost_sensitivity
)

def display_results(params: Dict[str, jnp.ndarray]):
    print("=" * 60)
    print("REE LEACH SEPARATION - ITEMISED COST & SENSITIVITIES")
    print("=" * 60)
    
    print("\n[Base Design Parameters]")
    for key, val in params.items():
        print(f"  - {key:<30}: {val.item():>8.2f}")
    
    # Run the forward pass (Costs)
    costs = itemized_cost(params)
    eac = total_annualized_cost(params)
    
    print("\n[Itemised Costs (USD)]")
    for key in ["mining_capex", "mining_opex", 
                "comminution_capex", "comminution_opex", 
                "leaching_capex", "leaching_opex", 
                "processing_capex", "processing_opex", 
                "total_capex", "total_opex"]:
        print(f"  - {key:<30}: ${costs[key].item():>11,.2f}")
        
    print("-" * 60)
    print(f"  > EQUIVALENT ANNUALISED COST: ${eac.item():>11,.2f}")
    
    # Run the backwards pass (Sensitivities via AutoDiff)
    sensitivities = cost_sensitivity(params)
    
    print("\n[Parameter Sensitivities (d(EAC)/d(Param))]")
    print("  (Shows how a 1-unit increase in parameter affects total annualised cost)")
    print("-" * 60)
    
    # Sort by absolute magnitude of sensitivity
    sorted_sensitivities = sorted(sensitivities.items(), key=lambda item: abs(item[1].item()), reverse=True)
    
    for key, grad_val in sorted_sensitivities:
        print(f"  - {key:<30}: ${grad_val.item():>11,.2f}")
        
    print("\nConclusion: The results above highlight which parameters drive the highest cost.")
    print("This information is returned by JAX in a single backwards pass and can directly")
    print("be plugged into gradient-descent flow-sheet optimisation strategies.")

if __name__ == "__main__":
    # Simulate a minimal REE flow-sheet with default proxy parameters
    example_params = {
        "ore_throughput_tpd": jnp.array(1500.0),
        "strip_ratio": jnp.array(3.5),
        "mine_depth_m": jnp.array(120.0),
        "bond_work_index": jnp.array(18.0),
        "residence_time_h": jnp.array(36.0),
        "acid_consumption_kg_t": jnp.array(65.0),
        "operating_temp_c": jnp.array(95.0),
        "sx_stages": jnp.array(12.0),
        "precipitation_reagent_tpy": jnp.array(800.0),
        "aq_flow_m3_h": jnp.array(150.0)
    }
    
    display_results(example_params)
