# Tutorial: Building and Optimizing Mineral Separation Flowsheets

This tutorial walks through building a complete multi-stage extraction and precipitation simulation in python, tallying economic factors, and minimizing OPEX using Bayesian Surrogate optimization.

## 1. Flowsheet Configuration
Flowsheets are instantiated using Pydantic schemas in `sep_agents.dsl.schemas`. You begin by defining feed liquids with masses in generic units (like `H2O(aq)` and `La+3`).

```python
from sep_agents.dsl.schemas import Flowsheet, Stream, UnitOp

feed = Stream(
    name="feed_liquor",
    phase="liquid",
    composition_wt={
        "H2O(aq)": 1000.0,
        "Ce+3": 5.0,
        "C2O4-2": 15.0,
    }
)
```

## 2. Unit Operations
Units are connected through simple `inputs` and `outputs` network edges. Parametric settings dictate the runtime physics:

```python
precip = UnitOp(
    id="precipitator",
    type="precipitator",
    inputs=["feed_liquor"],
    outputs=["slurry"],
    params={
        "T_C": 25.0,
        "residence_time_s": 3600.0,
        "reagent_dosage_gpl": 10.0
    }
)
fs = Flowsheet(name="my_plant", streams=[feed], units=[precip])
```

## 3. IDAES Simulation and KPIs
`run_idaes` builds sequential Pyomo unit blocks mapping to Reaktoro Thermodynamic equilibrium constraints.

```python
from sep_agents.sim.idaes_adapter import run_idaes
results = run_idaes(fs, database="light_ree")
print(results["kpis"])
```

## 4. Cost Evaluation
The solver evaluates global `overall.opex_USD` inside the IDAES wrapper by tallying up exact component feed masses and calculating proxy energy costs (e.g., Mixing Energy $= f(\text{residence time, dosage gpl})$). 

## 5. BoTorch Optimization
To explicitly optimize a parameter based on a desired KPI:

```python
from mcp_server.server import optimize_flowsheet
import yaml

optim_result = optimize_flowsheet(
    flowsheet_yaml=yaml.dump(fs.dict()),
    design_variables=[
        {"unit_id": "precipitator", "param": "reagent_dosage_gpl", "bounds": [0.1, 20.0]}
    ],
    objective_kpi="overall.opex_USD",
    maximize=False,
    n_iters=5
)

print(optim_result["optimal_parameters"]) # Found cheapest viable parameter point!
```
The optimizer will generate latin hypercube designs, train a single-task Gaussian Process, test Expected Improvement landscapes, and return the true optimal `Flowsheet` geometry natively.
