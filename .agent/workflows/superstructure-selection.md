---
description: Select or construct GDP superstructures for valorization of a characterized raw material
---

# Superstructure Selection Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given a characterized raw material (from `/resource-characterization`), this workflow identifies or constructs GDP superstructures targeting the identified value streams.

---

## Step 1: Query the Superstructure Registry

Use the `list_superstructures` MCP tool to retrieve all pre-built superstructures:

```
list_superstructures_tool()
```

The current registry includes:
- `lree_acid_leach` — LREE recovery from acid-leach liquor (SX + precipitation, with alternatives)
- `simple_sx_precip` — Minimal SX → Precipitator with optional scrubber
- `steel_slag_h2_co2` — H₂ production + CO₂ sequestration from steel slag

Evaluate each against the value streams identified in the previous step.

---

## Step 2: Evaluate Applicability

For each registered superstructure, assess:

| Criterion | Check |
|---|---|
| **Feed compatibility** | Does the superstructure accept the feed mineralogy / chemistry? |
| **Value streams covered** | Which of the identified value streams does it address? |
| **Missing pathways** | Are there value streams with no superstructure coverage? |
| **Scale appropriateness** | Is the superstructure designed for the throughput range? |

Produce a compatibility matrix. If a registered superstructure covers ≥80% of the value by potential revenue, recommend using it (possibly with minor modifications).

---

## Step 3: Construct New Superstructures (if needed)

If no pre-built superstructure adequately covers the identified value streams, construct a new one using the Separation-Agents DSL.

### 3a. Define the base flowsheet

Build a `Flowsheet` containing the **superset** of all candidate unit operations. Use the available unit types from `dsl/schemas.py`:

- `mill`, `cyclone` — Comminution / classification
- `lims` — Magnetic separation
- `flotation_bank` — Froth flotation
- `leach_reactor` — Acid/alkaline leaching
- `precipitator` — Chemical precipitation
- `solvent_extraction` — SX with distribution coefficients
- `ion_exchange` — IX columns
- `crystallizer` — Product crystallization
- `thickener` — Solid/liquid separation
- `mixer` — Stream merging
- `heat_exchanger`, `pump` — Utilities
- `carbonation_reactor` — CO₂ mineralization
- `serpentinization_reactor` — H₂ generation
- `separator` — Generic split

For each unit type, ensure required params are specified per `UNIT_PARAM_SPEC` in `schemas.py`.

### 3b. Define disjunctions

Identify mutually exclusive process alternatives (e.g., SX vs. IX for rare earth separation; oxalate vs. hydroxide precipitation). Define each as a `DisjunctionDef`:

```python
DisjunctionDef(
    name="separation_method",
    unit_ids=["sx_d2ehpa", "sx_pc88a", "ix_resin_a"],
    description="Choose one separation technology"
)
```

Mark standalone optional units with `optional=True` (e.g., scrubbing stages, polishing steps).

For multi-stage units, use `stage_range` (e.g., `stage_range=(2, 6)` for SX cascades).

### 3c. Set the objective

Choose from:
- `maximize_value_per_kg_ore` — **Default**. Net revenue minus annualized cost per unit feedstock
- `minimize_opex` — Lowest operating cost (if revenue is fixed or external)
- `maximize_recovery` — Maximum extraction of target species
- `minimize_lca` — Lowest environmental footprint

If the user has a specific objective, use it. Otherwise default to `maximize_value_per_kg_ore`.

### 3d. Define continuous bounds

For parameters amenable to BoTorch continuous optimization, specify bounds:

```python
continuous_bounds={
    "sx_1.organic_to_aqueous_ratio": (0.5, 3.0),
    "leach_1.T_C": (50.0, 95.0),
    "precip_1.reagent_dosage_gpl": (5.0, 50.0),
}
```

### 3e. Validate the superstructure

```python
from sep_agents.dsl.schemas import Superstructure
ss = Superstructure(name=..., base_flowsheet=..., disjunctions=..., ...)
ss.base_flowsheet.validate_graph()  # Check stream connectivity
```

---

## Step 4: Enumerate Topologies

For each superstructure, enumerate all valid configurations:

```python
from sep_agents.opt.gdp_builder import enumerate_configurations
configs = enumerate_configurations(superstructure)
```

Report:
- Total number of feasible topologies
- Brief description of each configuration (active/bypassed units)
- Rough expected performance ranking (if heuristic is available)

---

## Step 5: Save New Superstructures to Registry

If a new superstructure was created in Step 3, persist it for future reuse:

1. Create a new Python file or add a factory function to an existing file in `src/sep_agents/dsl/`:
   - For REE-focused: add to `ree_superstructures.py` and update `SUPERSTRUCTURE_REGISTRY`
   - For geological (H₂/CO₂): add to `geo_superstructures.py` and update `GEO_SUPERSTRUCTURE_REGISTRY`
   - For other categories: create a new file (e.g., `brine_superstructures.py`) following the same pattern
2. The factory function should return a `Superstructure` object
3. Register it in the appropriate `*_REGISTRY` dict

---

## Step 6: Output

Produce:
- A YAML representation of each candidate superstructure
- A topology enumeration summary table
- Recommendation for which superstructures to optimize

---

## Handoff

After completing this workflow, recommend that the user proceed with:
```
/process-optimization
```
passing the superstructure name(s) and feed characterization as context.
