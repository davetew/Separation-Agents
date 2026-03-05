from sep_agents.cost.tea import estimate_opex_kwh_reagents
from sep_agents.cost.lca import estimate_co2e

# Future-proofing: If the system has JAX installed and params provided, we can return the TEA
try:
    from sep_agents.cost.jax_tea import itemized_cost, cost_sensitivity
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

async def estimate_cost(params: dict):
    """Estimate OPEX, CO2e, and optional JAX itemized costs from process KPIs.
    
    params = {
      "kpis": {"unit.kpi": value, ...},
      "jax_params": {"ore_throughput_tpd": value, ...}  # optional
    }
    See /cost-analysis workflow for reporting format.
    """
    kpis = params.get("kpis", {}) or {}
    
    # 1. Provide the structural Opex/CO2e proxy
    opex = estimate_opex_kwh_reagents(kpis)
    co2e = estimate_co2e(kpis)
    
    result = {"status": "ok", "OPEX": opex, "CO2e": co2e}
    
    # 2. If 'jax_params' are provided, run the fully differential JAX models
    if JAX_AVAILABLE and "jax_params" in params and params["jax_params"]:
        try:
            # Convert dictionary floats to jax traceables
            j_params = {k: jnp.array(float(v)) for k, v in params["jax_params"].items()}
            
            # Execute JAX model evaluation and gradient
            costs = itemized_cost(j_params)
            sensitivities = cost_sensitivity(j_params)
            
            # Unpack jnp scalars to standard python floats for JSON serialization
            result["jax_itemized_costs"] = {k: float(v) for k, v in costs.items()}
            result["jax_cost_sensitivities"] = {k: float(v) for k, v in sensitivities.items()}
            
        except Exception as e:
            result["jax_error"] = str(e)
            
    return result
