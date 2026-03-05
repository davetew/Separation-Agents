# Tutorial: Building and Optimizing Mineral Separation Flowsheets

This tutorial walks through building a complete multi-stage extraction and precipitation simulation, tallying economic factors, and optimizing both parameters and process topology.

## 1. Flowsheet Configuration

Flowsheets are instantiated using Pydantic schemas in `sep_agents.dsl.schemas`. You begin by defining feed liquids with mass or molar compositions:

```python
from sep_agents.dsl.schemas import Flowsheet, Stream, UnitOp

feed = Stream(
    name="feed_liquor",
    phase="liquid",
    composition_wt={
        "H2O(aq)": 1000.0,
        "La+3": 10.0,
        "Ce+3": 10.0,
        "Nd+3": 10.0,
        "HCl(aq)": 50.0,
    }
)
```

## 2. Unit Operations

Units are connected through simple `inputs` and `outputs` network edges. Parametric settings dictate the runtime physics:

```python
sx = UnitOp(
    id="sx_1",
    type="solvent_extraction",
    inputs=["feed_liquor"],
    outputs=["organic", "raffinate"],
    params={
        "distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
        "organic_to_aqueous_ratio": 1.0,
    }
)

precip = UnitOp(
    id="precip",
    type="precipitator",
    inputs=["organic"],
    outputs=["solid", "barren"],
    params={
        "residence_time_s": 3600.0,
        "reagent_dosage_gpl": 10.0,
    }
)

fs = Flowsheet(name="my_plant", streams=[feed], units=[sx, precip])
```

## 3. SM Simulation (IDAES + Reaktoro)

The sequential-modular backend solves units one-by-one using Reaktoro for thermodynamic equilibrium:

```python
from sep_agents.sim.idaes_adapter import run_idaes

results = run_idaes(fs, database="light_ree")
print(results["kpis"])
# {'overall.opex_USD': 0.65, 'overall.lca_kg_CO2e': 3.66, ...}
```

Best for: rigorous thermodynamic speciation, exploratory analysis.

## 4. EO Simulation (Pyomo + IPOPT)

The equation-oriented backend formulates the entire flowsheet as a single Pyomo NLP, solved simultaneously by IPOPT. This is **5–10× faster** than SM and supports gradient-based optimization:

```python
from sep_agents.sim.eo_flowsheet import run_eo

result = run_eo(fs, objective="minimize_opex")
print(f"Status: {result['status']}")       # 'ok'
print(f"KPIs:   {result['kpis']}")          # OPEX, LCA, recovery metrics
print(f"Time:   {result['solve_time_s']}s") # ~1-2s for 3-unit flowsheet
```

Supported objectives: `"none"` (feasibility), `"minimize_opex"`, `"maximize_recovery"`.

### EO Unit Models

| Unit Type | EO Builder | Key Parameters |
|-----------|-----------|----------------|
| `solvent_extraction` | `build_sx_stage` | `distribution_coeff`, `organic_to_aqueous_ratio` |
| `precipitator` | `build_precipitator` | `residence_time_s`, `reagent_dosage_gpl` |
| `ion_exchange` | `build_ix_column` | `selectivity_coeff`, `bed_volume_m3` |

## 5. Superstructure Optimization with GDP

Generalized Disjunctive Programming (GDP) simultaneously optimizes **which units to include** and their **operating parameters**. This replaces exhaustive enumeration of topologies:

```python
from sep_agents.dsl.schemas import Superstructure, DisjunctionDef
from sep_agents.opt.gdp_eo import solve_gdp_eo

# Define a superstructure with an optional precipitator
ss = Superstructure(
    name="optimal_plant",
    base_flowsheet=fs,
    fixed_units=["sx_1"],            # SX is always present
    objective="maximize_recovery",
)

result = solve_gdp_eo(ss)
print(f"Active units:   {result.active_units}")     # e.g., ['sx_1', 'precip']
print(f"Bypassed units: {result.bypassed_units}")    # e.g., []
print(f"Objective:      {result.objective_value}")   # e.g., -0.425
```

### Enumerative GDP Solver

For smaller superstructures, the enumerative solver in `gdp_solver.py` exhaustively evaluates all feasible topologies and ranks them by objective:

```python
from sep_agents.dsl.ree_superstructures import SUPERSTRUCTURE_REGISTRY
from sep_agents.opt.gdp_solver import optimize_superstructure

ss = SUPERSTRUCTURE_REGISTRY["steel_slag_valorization"]()
result = optimize_superstructure(ss, database="light_ree")

for ev in result.all_results:
    print(f"{sorted(ev.config.active_unit_ids)} → obj={ev.objective_value:.4f}")
```

### XOR Disjunctions (Choose-one alternatives)

```python
# Define a flowsheet where either SX or IX is used, but not both
fs_alt = Flowsheet(name="alt_plant", streams=[feed], units=[
    UnitOp(id="leach", type="leach",
           params={"acid_type": "HCl", "acid_molarity": 2.0},
           inputs=["feed_liquor"], outputs=["leachate"]),
    UnitOp(id="sx_alt", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["leachate"], outputs=["org", "raf"]),
    UnitOp(id="ix_alt", type="ion_exchange",
           params={"selectivity_coeff": {"La": 1.0, "Ce": 2.0},
                   "bed_volume_m3": 1.0},
           inputs=["leachate"], outputs=["loaded", "eluate"]),
])

ss_xor = Superstructure(
    name="xor_demo",
    base_flowsheet=fs_alt,
    fixed_units=["leach"],
    disjunctions=[DisjunctionDef(name="sep", unit_ids=["sx_alt", "ix_alt"])],
    objective="minimize_opex",
)

result = solve_gdp_eo(ss_xor)
# Exactly one of sx_alt / ix_alt will be active
```

## 6. Cost Evaluation

Both backends compute global `overall.opex_USD` by tallying component feed masses and calculating proxy energy costs:

```python
# After any sim
print(result["kpis"]["overall.opex_USD"])      # Proxy OPEX in USD
print(result["kpis"]["overall.lca_kg_CO2e"])   # Lifecycle CO2e
```

## 7. BoTorch Parameter Optimization

For black-box parameter optimization (when gradients are unavailable):

```python
from mcp_server.server import optimize_flowsheet
import yaml

optim_result = optimize_flowsheet(
    flowsheet_yaml=yaml.dump(fs.dict()),
    design_variables=[
        {"unit_id": "sx_1", "param": "organic_to_aqueous_ratio", "bounds": [0.5, 3.0]}
    ],
    objective_kpi="overall.opex_USD",
    maximize=False,
    n_iters=5
)
print(optim_result["optimal_parameters"])
```

The optimizer generates Latin Hypercube designs, trains a SingleTaskGP surrogate, optimizes Log Expected Improvement, and returns the best flowsheet configuration.

## 8. JAX Equilibrium Solver

The JAX equilibrium solver (`jax_equilibrium.py`) is a pure-JAX implementation of constrained Gibbs energy minimization. It provides the same speciation outputs as Reaktoro but is fully differentiable via `jax.grad`:

```python
from sep_agents.sim.jax_equilibrium import build_jax_system, JaxEquilibriumSolver

# Build a chemical system for light REEs
system = build_jax_system(preset="light_ree", include_minerals=True)
solver = JaxEquilibriumSolver(system)

# Speciate at 250°C
result = solver.speciate(
    temperature_C=250.0,
    pressure_atm=100.0,
    water_kg=1.0,
    ree_mol={"La": 0.01, "Ce": 0.01, "Nd": 0.01},
    acid_mol={"HCl": 0.1},
)

print(f"pH = {result['pH']:.2f}")
print(f"Eh = {result['Eh_V']:.3f} V")
print(f"Species: {result['species_amounts']}")
```

### HKF-Based System

For expanded databases with 1100+ species from SUPCRTBL:

```python
from sep_agents.sim.jax_equilibrium import build_jax_system_hkf

system = build_jax_system_hkf(
    preset="light_ree",
    include_minerals=True,
    include_gases=False,
)
solver = JaxEquilibriumSolver(system)
```

### Key Advantages

| Feature | Reaktoro | JAX Solver |
|---------|----------|------------|
| Differentiable | No | Yes (`jax.grad`, `jax.jacfwd`) |
| JIT-compilable | No | Yes (`jax.jit`) |
| Custom EOS | Limited | HKF + Holland-Powell + Peng-Robinson |
| GPU acceleration | No | Yes (via JAX backend) |

## 9. JAX Differentiable Cost Analysis

The JAX TEA module (`jax_tea.py`) provides fully differentiable cost functions for techno-economic analysis. All functions are pure JAX, enabling gradient-based sensitivity analysis via `jax.grad`:

```python
import jax.numpy as jnp
from sep_agents.cost.jax_tea import (
    total_annualized_cost,
    cost_sensitivity,
    itemized_cost,
)

params = {
    "ore_throughput_tpd": jnp.array(500.0),
    "strip_ratio": jnp.array(2.0),
    "mine_depth_m": jnp.array(50.0),
    "bond_work_index": jnp.array(15.0),
    "residence_time_h": jnp.array(4.0),
    "acid_consumption_kg_t": jnp.array(50.0),
    "operating_temp_c": jnp.array(250.0),
    "sx_stages": jnp.array(10.0),
    "precipitation_reagent_tpy": jnp.array(500.0),
    "aq_flow_m3_h": jnp.array(100.0),
}

# Compute total EAC
eac = total_annualized_cost(params)
print(f"EAC: ${float(eac):,.0f}/yr")

# Compute itemized costs
costs = itemized_cost(params)
for k, v in costs.items():
    print(f"  {k}: ${float(v):,.0f}")

# Compute sensitivities: d(EAC)/d(param)
grads = cost_sensitivity(params)
for k, v in grads.items():
    print(f"  d(EAC)/d({k}) = {float(v):,.2f}")
```

### Cost Function Hierarchy

| Function | Returns | Differentiable |
|----------|---------|---------------|
| `mining_cost(params)` | (CAPEX, OPEX) | ✅ |
| `comminution_cost(params)` | (CAPEX, OPEX) | ✅ |
| `leaching_cost(params)` | (CAPEX, OPEX) | ✅ |
| `processing_cost(params)` | (CAPEX, OPEX) | ✅ |
| `itemized_cost(params)` | Dict of all components | ✅ |
| `total_annualized_cost(params)` | Scalar EAC | ✅ |
| `cost_sensitivity(params)` | Dict of ∂(EAC)/∂(param) | N/A (returns grads) |

## 10. Agent Workflows

The agent workflow system (`.agent/workflows/`) automates the full valorization pipeline. The top-level `/valorize` command chains six stages:

1. **`/resource-characterization`** — Parse raw material, run equilibrium speciation, identify value streams, assign commodity prices
2. **`/superstructure-selection`** — Query superstructure registry, enumerate topologies, construct new superstructures if needed
3. **`/process-optimization`** — GDP topology optimization, IDAES simulation, BoTorch BO, JAX TEA sensitivity
4. **`/cost-analysis`** — CAPEX/OPEX/EAC/revenue/net value reporting
5. **`/valorization-report`** — LaTeX technical report with TikZ PFDs, tables, and equations
6. **`/valorization-presentation`** — Branded Beamer slide deck using `sf-branding` + `lualatex`

All Python commands run inside the `rkt` conda environment. User gates are inserted at key decision points (after characterization and superstructure selection) to allow review before proceeding.
