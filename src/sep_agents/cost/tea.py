
from typing import Dict

def estimate_opex_kwh_reagents(unit_kpis: Dict[str, float]) -> float:
    # toy estimator: sum of KPI values interpreted as cost contributors
    return float(sum(unit_kpis.values()))
