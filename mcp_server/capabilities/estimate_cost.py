from sep_agents.cost.tea import estimate_opex_kwh_reagents
from sep_agents.cost.lca import estimate_co2e

async def estimate_cost(params: dict):
    """
    params = {
      "kpis": {"unit.kpi": value, ...}
    }
    """
    kpis = params.get("kpis", {}) or {}
    opex = estimate_opex_kwh_reagents(kpis)
    co2e = estimate_co2e(kpis)
    return {"status": "ok", "OPEX": opex, "CO2e": co2e}
