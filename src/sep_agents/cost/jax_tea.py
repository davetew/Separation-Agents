import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple

# Proxy defaults based on industry standards
DEFAULT_CONSTANTS = {
    # Mining
    "mining_capex_factor": 50000.0,  # Base USD per tpd capacity
    "mining_opex_base_per_t": 2.50,  # Base USD/t ore mined
    "strip_ratio_multiplier": 1.50,  # USD/t waste
    "depth_factor": 0.005,           # Additional cost per meter depth
    
    # Comminution
    "bond_work_index_multiplier": 1.2, # kWh/t per unit of Bond Work Index above 10
    "electricity_cost_kwh": 0.08,    # USD/kWh
    "mill_capex_factor": 15000.0,    # Base USD per tpd milling capacity
    
    # Leaching
    "acid_cost_per_t": 150.0,        # USD/t acid
    "leach_residence_time_factor": 200.0, # Capex proxy per hour of residence time per tpd
    "heating_cost_per_deg": 0.05,    # USD/t per degree C above ambient
    
    # Processing (SX / Precipitation)
    "sx_stage_capex": 50000.0,       # Base USD per SX stage
    "precipitation_reagent_cost": 500.0 # USD/t reagent
}

def mining_cost(params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute mining CAPEX and OPEX based on proxy parameters.
    
    Args:
        params: PyTree containing:
            - ore_throughput_tpd: Ore throughput (tons per day)
            - strip_ratio: Waste to ore ratio
            - mine_depth_m: Average depth of the mine (meters)
            
    Returns:
        capex_usd, opex_usd_per_year
    """
    tpd = params.get("ore_throughput_tpd", jnp.array(1000.0))
    strip_ratio = params.get("strip_ratio", jnp.array(2.0))
    depth_m = params.get("mine_depth_m", jnp.array(50.0))
    
    # CAPEX proxy scales sub-linearly with capacity (rule of six-tenths approximation)
    capex = DEFAULT_CONSTANTS["mining_capex_factor"] * (tpd ** 0.6)
    
    # OPEX proxy includes base cost, waste handling, and depth penalty
    opex_per_t = DEFAULT_CONSTANTS["mining_opex_base_per_t"] + \
                 (strip_ratio * DEFAULT_CONSTANTS["strip_ratio_multiplier"]) + \
                 (depth_m * DEFAULT_CONSTANTS["depth_factor"])
                 
    # Assume 330 operating days per year
    opex_per_year = opex_per_t * tpd * 330.0
    
    return capex, opex_per_year

def comminution_cost(params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute comminution (crushing/milling) CAPEX and OPEX.
    
    Args:
        params: PyTree containing:
            - ore_throughput_tpd: Ore throughput (tons per day)
            - bond_work_index: Material hardness (kWh/t)
            
    Returns:
        capex_usd, opex_usd_per_year
    """
    tpd = params.get("ore_throughput_tpd", jnp.array(1000.0))
    bwi = params.get("bond_work_index", jnp.array(15.0))
    
    # Base CAPEX scaling
    capex = DEFAULT_CONSTANTS["mill_capex_factor"] * (tpd ** 0.6)
    
    # Energy requirement based on Bond Work Index (simplified)
    # E = W_i * (10/sqrt(P80) - 10/sqrt(F80)). Proxy this as scaling linearly above a baseline
    energy_kwh_per_t = 10.0 + jnp.maximum(0.0, (bwi - 10.0)) * DEFAULT_CONSTANTS["bond_work_index_multiplier"]
    
    opex_per_t = energy_kwh_per_t * DEFAULT_CONSTANTS["electricity_cost_kwh"]
    opex_per_year = opex_per_t * tpd * 330.0
    
    return capex, opex_per_year

def leaching_cost(params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute leaching block CAPEX and OPEX.
    
    Args:
        params: PyTree containing:
            - ore_throughput_tpd: Ore throughput (tons per day)
            - residence_time_h: Vat/tank residence time
            - acid_consumption_kg_t: Acid used per ton of ore
            - operating_temp_c: Operating temperature (C)
            
    Returns:
        capex_usd, opex_usd_per_year
    """
    tpd = params.get("ore_throughput_tpd", jnp.array(1000.0))
    rt_h = params.get("residence_time_h", jnp.array(24.0))
    acid_kg_t = params.get("acid_consumption_kg_t", jnp.array(50.0))
    temp_c = params.get("operating_temp_c", jnp.array(80.0))
    
    # CAPEX scales with total volume required (throughput * residence time)
    capex = DEFAULT_CONSTANTS["leach_residence_time_factor"] * tpd * rt_h
    
    # Acid cost 
    acid_cost_t = (acid_kg_t / 1000.0) * DEFAULT_CONSTANTS["acid_cost_per_t"]
    
    # Heating OPEX (assume ambient is 25C)
    dt = jnp.maximum(0.0, temp_c - 25.0)
    heating_cost_t = dt * DEFAULT_CONSTANTS["heating_cost_per_deg"]
    
    opex_per_t = acid_cost_t + heating_cost_t
    opex_per_year = opex_per_t * tpd * 330.0
    
    return capex, opex_per_year

def processing_cost(params: Dict[str, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute downstream separation and precipitation CAPEX and OPEX.
    
    Args:
        params: PyTree containing:
            - sx_stages: Number of solvent extraction stages
            - precipitation_reagent_tpy: Tons per year of precipitating agents
            - aq_flow_m3_h: Aqueous flow rate (determines equipment size)
            
    Returns:
        capex_usd, opex_usd_per_year
    """
    n_stages = params.get("sx_stages", jnp.array(10.0))
    precip_reagent_tpy = params.get("precipitation_reagent_tpy", jnp.array(500.0))
    aq_flow = params.get("aq_flow_m3_h", jnp.array(100.0))
    
    # SX Capex proxy
    sx_capex = DEFAULT_CONSTANTS["sx_stage_capex"] * n_stages * (aq_flow / 50.0)**0.6
    
    # Precipitation capex proxy
    precip_capex = 150000.0 * (aq_flow / 50.0)**0.6
    
    total_capex = sx_capex + precip_capex
    
    # OPEX proxy (mostly reagent and base pumping)
    pumping_opex = aq_flow * 24.0 * 330.0 * 0.02 * DEFAULT_CONSTANTS["electricity_cost_kwh"]
    reagent_opex = precip_reagent_tpy * DEFAULT_CONSTANTS["precipitation_reagent_cost"]
    
    total_opex = pumping_opex + reagent_opex
    
    return total_capex, total_opex

def itemized_cost(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Compute total itemized project costs.
    
    Returns a dictionary of named cost components (CAPEX and OPEX).
    All outputs are differentiable w.r.t continuous inputs in params.
    """
    m_cap, m_op = mining_cost(params)
    c_cap, c_op = comminution_cost(params)
    l_cap, l_op = leaching_cost(params)
    p_cap, p_op = processing_cost(params)
    
    total_capex = m_cap + c_cap + l_cap + p_cap
    total_opex = m_op + c_op + l_op + p_op
    
    return {
        "mining_capex": m_cap,
        "mining_opex": m_op,
        "comminution_capex": c_cap,
        "comminution_opex": c_op,
        "leaching_capex": l_cap,
        "leaching_opex": l_op,
        "processing_capex": p_cap,
        "processing_opex": p_op,
        "total_capex": total_capex,
        "total_opex": total_opex
    }

def total_annualized_cost(params: Dict[str, jnp.ndarray], discount_rate: float = 0.08, lifetime_years: float = 20.0) -> jnp.ndarray:
    """
    Computes the total equivalent annualized cost (EAC) for the project.
    EAC = OPEX + CAPEX * (r / (1 - (1+r)^(-n)))
    """
    costs = itemized_cost(params)
    cap_recovery_factor = discount_rate / (1.0 - (1.0 + discount_rate)**(-lifetime_years))
    
    # Compute EAC
    eac = costs["total_opex"] + costs["total_capex"] * cap_recovery_factor
    return eac

def cost_sensitivity(params: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """
    Computes the gradient of the total annualized cost with respect to all continuous parameters.
    Returns a dictionary of the same structure as params, containing d(EAC)/d(param).
    """
    # Define a pure function that returns the scalar EAC
    def eac_fn(p):
        return total_annualized_cost(p)
    
    # Use jax.grad to compute sensitivities automatically
    grad_fn = jax.grad(eac_fn)
    return grad_fn(params)
