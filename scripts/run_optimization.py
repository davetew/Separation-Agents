import yaml
import torch
import copy
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.idaes_adapter import run_idaes
from sep_agents.opt.bo import BotorchOptimizer

flowsheet_yaml = """
name: nd_isolation_optimized
streams:
- name: feed
  phase: liquid
  composition_wt:
    H2O(aq): 1000.0
    Nd+3: 2.1636
    Ce+3: 2.8023
    La+3: 1.3891
    HCl(aq): 1.8230
- name: org_extract
  phase: liquid
- name: aq_raffinate
  phase: liquid
- name: nd_crystals
  phase: solid
- name: spent_liquor
  phase: liquid
units:
- id: sx_1
  type: solvent_extraction
  inputs:
  - feed
  outputs:
  - org_extract
  - aq_raffinate
  params:
    distribution_coeff:
      Nd+3: 0.1
      Ce+3: 10.0
      La+3: 50.0
    organic_to_aqueous_ratio: 1.0
- id: nd_precipitator
  type: precipitator
  inputs:
  - aq_raffinate
  outputs:
  - nd_crystals
  - spent_liquor
  params:
    T_C: 25.0
    residence_time_s: 3600.0
    reagent_name: 'NaOH(aq)'
    reagent_dosage_gpl: 4.0
"""

design_variables = [
    {"unit_id": "sx_1", "param": "organic_to_aqueous_ratio", "bounds": [0.5, 5.0]},
    {"unit_id": "nd_precipitator", "param": "reagent_dosage_gpl", "bounds": [3.5, 6.0]}
]

data = yaml.safe_load(flowsheet_yaml)
database = "light_ree"
objective_kpi = "overall.opex_USD"

def flowsheet_objective(candidate_x: torch.Tensor) -> float:
    current_data = copy.deepcopy(data)
    units = [UnitOp(**u) for u in current_data.get("units", [])]
    streams = [Stream(**s) for s in current_data.get("streams", [])]
    
    for i, var in enumerate(design_variables):
        target_unit_id = var["unit_id"]
        param_key = var["param"]
        val = candidate_x[i].item()
        for u in units:
            if u.id == target_unit_id:
                u.params[param_key] = val
                break
    
    fs = Flowsheet(name="optimized", units=units, streams=streams)
    result = run_idaes(fs, database=database)
    
    if result["status"] == "ok" and objective_kpi in result["kpis"]:
        # If recovery is low, we want to penalize
        recovery_nd = 0.0
        crystals = result["streams"].get("nd_crystals")
        if crystals:
            recovery_nd = crystals["species_amounts"].get("Nd(OH)3(s)", 0.0) / 15.0
            
        if recovery_nd < 0.95: # Penalty for low recovery
             return float(result["kpis"][objective_kpi]) + (0.95 - recovery_nd) * 100.0
             
        return float(result["kpis"][objective_kpi])
    return 1e9

bounds_list = [v["bounds"] for v in design_variables]
bounds_tensor = torch.tensor(bounds_list, dtype=torch.double).T

opt = BotorchOptimizer(maximize=False)
best_x, best_y, history = opt.optimize(
    objective_fn=flowsheet_objective,
    bounds=bounds_tensor,
    n_initial=5,
    n_iters=30
)

print(f"Best X: {best_x.tolist()}")
print(f"Best Y (OPEX + Penalty): {best_y}")

# Apply final
best_x_list = best_x.tolist()
units = [UnitOp(**u) for u in data.get("units", [])]
for i, var in enumerate(design_variables):
    for u in units:
        if u.id == var["unit_id"]:
            u.params[var["param"]] = best_x_list[i]
            break
final_fs = Flowsheet(name="Optimized_Flowsheet", units=units, streams=[Stream(**s) for s in data.get("streams", [])])
print("--- OPTIMIZED YAML ---")
print(yaml.dump(final_fs.dict(), sort_keys=False))

# Run final to get KPIs
final_res = run_idaes(final_fs, database=database)
print("--- FINAL KPIs ---")
print(yaml.dump(final_res["kpis"], sort_keys=False))
print("--- FINAL ND CRYSTALS ---")
print(yaml.dump(final_res["streams"]["nd_crystals"]["species_amounts"], sort_keys=False))
