---
description: Optimize GDP superstructures via IDAES simulation, BoTorch, and JAX TEA to maximize net economic value
---

# Process Optimization Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given one or more GDP superstructures (from `/superstructure-selection`) and a characterized feed (from `/resource-characterization`), this workflow optimizes each to maximize net economic value and produces ranked trade-study results.

---

## Step 1: GDP Topology Optimization

For each candidate superstructure, run topology optimization using one of two solvers:

### Option A: Enumerative GDP Solver (default for small superstructures ≤ 32 topologies)

Use the `optimize_superstructure_tool` MCP tool:

```
optimize_superstructure_tool(
    superstructure_name="<name>",
    objective="maximize_value_per_kg_ore",
    optimize_continuous=True,
    n_bo_iters=10,
    database="light_ree"  # or appropriate Reaktoro database
)
```

This enumerates all topologies, simulates each via the IDAES sequential-modular solver, and optionally runs BoTorch on continuous parameters for the best topology.

> **Note**: Geo-reactor units (serpentinization, carbonation, heat exchangers, pumps) use **stoichiometric conversion and engineering models** built into the `IDAESFlowsheetBuilder`, not Reaktoro. Reaktoro is used only for REE aqueous equilibrium reactors (leach, precipitation). See `idaes_adapter.py` for the unit solver dispatch.

### Option B: Rigorous EO GDP Solver (for larger superstructures > 32 topologies)

Use the `GDPEOBuilder` directly:

```python
from sep_agents.opt.gdp_eo import GDPEOBuilder
from sep_agents.dsl.ree_superstructures import SUPERSTRUCTURE_REGISTRY

ss = SUPERSTRUCTURE_REGISTRY["<name>"]()
builder = GDPEOBuilder(preset="lree")
result = builder.build_and_solve(
    ss,
    solver_name="ipopt",
    transformation="bigm",
    bigM=1e4
)
```

This formulates the superstructure as a single Pyomo.GDP model and solves with Big-M or Hull reformulation.

Record the result: `active_units`, `bypassed_units`, `kpis`, `objective_value`, `solve_time_s`.

---

## Step 2: Full IDAES Simulation of Top Candidates

For the top 3–5 ranked topologies from Step 1, run full IDAES simulations to obtain detailed stream states:

Use `run_idaes_flowsheet` or `simulate_flowsheet` MCP tool:

```
run_idaes_flowsheet(
    flowsheet_yaml="<yaml_string>",
    database="SUPRCRT - BL"
)
```

Or for programmatic access:

```python
from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder
builder = IDAESFlowsheetBuilder()
result = builder.build_and_solve(flowsheet, database="SUPRCRT - BL")
```

Extract from each simulation:
- Stream states (compositions, temperatures, pressures, pH)
- Unit-level KPIs (recovery, conversion, energy consumption)
- Mass and energy balances

---

## Step 3: Techno-Economic Analysis

For each simulated topology, compute economics using the JAX TEA module.

### 3a. Itemized Costs

```python
from sep_agents.cost.jax_tea import itemized_cost, total_annualized_cost, cost_sensitivity
import jax.numpy as jnp

params = {
    "ore_throughput_tpd": jnp.array(<throughput>),
    "strip_ratio": jnp.array(<ratio>),
    "mine_depth_m": jnp.array(<depth>),
    "bond_work_index": jnp.array(<bwi>),
    "residence_time_h": jnp.array(<rt>),
    "acid_consumption_kg_t": jnp.array(<acid>),
    "operating_temp_c": jnp.array(<temp>),
    "sx_stages": jnp.array(<stages>),
    "precipitation_reagent_tpy": jnp.array(<reagent>),
    "aq_flow_m3_h": jnp.array(<flow>),
}

costs = itemized_cost(params)
eac = total_annualized_cost(params, discount_rate=0.08, lifetime_years=20.0)
```

### 3b. Revenue Estimation

For each product stream, compute annual revenue:

```
Revenue_i = Production_i × Recovery_i × Price_i × Operating_days/yr
```

Use simulation-derived recoveries where available; use the speciation-informed estimates otherwise.

### 3c. Net Economic Value

```
Net Value = Total Revenue − EAC
Net Value per tonne = Net Value / (throughput_tpd × operating_days)
```

### 3d. Cost Sensitivity (JAX Autodiff)

```python
sensitivities = cost_sensitivity(params)
```

Rank the top-5 cost drivers by |∂EAC/∂parameter| and present in a table.

---

## Step 4: BoTorch Continuous Optimization (Optional)

If `optimize_continuous=True` was enabled in Step 1, or if the user requests it, run BoTorch Bayesian optimization over continuous design variables.

Typical design variables and bounds:
- Leach temperature: [50, 95] °C
- Leach residence time: [1, 8] hours
- SX organic-to-aqueous ratio: [0.5, 3.0]
- Precipitator reagent dosage: [5, 50] g/L

Each BoTorch iteration runs a full IDAES simulation as the objective function. Use 10–20 iterations with 5 initial Latin Hypercube samples.

Record the optimal design variable values and the improvement in net value.

---

## Step 5: Trade-Study Comparison Table

Assemble results into a ranked comparison table:

| Rank | Topology | Active Units | CAPEX ($M) | OPEX ($M/yr) | EAC ($M/yr) | Revenue ($M/yr) | Net Value ($M/yr) | Net $/t |
|------|----------|-------------|------------|--------------|-------------|-----------------|-------------------|---------|
| 1 | Config-3 | SX+Precip+Carb | ... | ... | ... | ... | ... | ... |
| 2 | Config-1 | IX+Precip | ... | ... | ... | ... | ... | ... |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Include a brief commentary on the economic drivers for the top-ranked configuration.

---

## Step 6: Generate Intermediate Reports

Generate Markdown reports using the built-in report generators:

For GDP results:
```python
from sep_agents.report import generate_gdp_report
md, path = generate_gdp_report(gdp_result, superstructure_name="<name>", output_dir="reports")
```

For individual flowsheet results:
```python
from sep_agents.report import generate_report
md, path = generate_report(request_text, flowsheet, result, states, baseline_kpis, ...)
```

These generate process flow diagrams (PNG/SVG), stream state tables, and economic summaries.

---

## Step 7: Output

Produce:
- Trade-study comparison table (Markdown)
- Detailed simulation results for top-3 topologies
- Cost sensitivity analysis for the optimal topology
- BoTorch optimization results (if applicable)
- PFD images for each evaluated topology

Save all outputs to the `reports/` directory.

---

## Handoff

After completing this workflow, recommend that the user proceed with:
```
/valorization-report
```
to generate the formal LaTeX technical report, and/or:
```
/valorization-presentation
```
to generate a Beamer slide deck. Pass the trade-study results and optimal configuration as context.
