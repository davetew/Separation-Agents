# Separation Agents

LLM-guided, physics-grounded **multi-agent** system that can propose, simulate, and optimize
mineral/metal separation flowsheets (specifically targeting Rare Earth Elements).

## Key ideas
- **Flowsheet DSL** (YAML/JSON) → validated into Pydantic models.
- **Unit op stubs** (comminution, cyclone, magnetic, flotation, hydromet, precipitator, solvent extraction, ion exchange).
- **Two simulation backends**:
  - **Sequential-Modular (SM)** via IDAES/Pyomo + Reaktoro equilibrium speciation
  - **Equation-Oriented (EO)** via native Pyomo (simultaneous NLP solved by IPOPT)
- **Generalized Disjunctive Programming (GDP)** for superstructure optimization — simultaneously selects the best topology *and* optimal continuous parameters.
- **Techno-Economic Analysis (TEA) & Life Cycle Assessment (LCA)** embedded into the sequences, allowing for agent-driven OPEX and $CO_2$e evaluation.
- **Optimizers**: BoTorch Bayesian Optimization (parameter tuning) + Pyomo.GDP/IPOPT (topology + parameter co-optimization).
- **Orchestrator** loop that ties proposal → simulate → critique → optimize.

## Quickstart
```bash
# Create and activate a virtual env
python -m venv .venv && source .venv/bin/activate

# Install (core)
pip install -e .

# Optional heavy deps
pip install 'sep-agents[thermo]'

# Start notebook or run CLI flowsheet
python scripts/run_loop.py examples/steel_slag_minimal.yaml
```

### EO Quick Demo
```python
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
from sep_agents.sim.eo_flowsheet import run_eo

fs = Flowsheet(name="demo", units=[
    UnitOp(id="sx_1", type="solvent_extraction",
           params={"distribution_coeff": {"La": 2.0, "Ce": 5.0, "Nd": 3.5},
                   "organic_to_aqueous_ratio": 1.0},
           inputs=["feed"], outputs=["org", "raf"]),
], streams=[
    Stream(name="feed", phase="liquid",
           composition_wt={"H2O(aq)": 1000, "La+3": 10, "Ce+3": 10,
                           "Nd+3": 10, "HCl(aq)": 50}),
])
result = run_eo(fs, objective="minimize_opex")
print(result["kpis"])
```

## Repo layout
```
src/sep_agents/
  dsl/                # schemas & loaders
  units/              # unit op models (SM stubs + EO Pyomo blocks)
  sim/                # adapters: IDAES (SM), eo_flowsheet (EO)
  opt/                # optimization: BoTorch (BO), gdp_eo (GDP)
  orchestrator/       # planner loop
  critic/             # feasibility & sanity checks
  properties/         # custom thermodynamic element databases
  cost/               # TEA/LCA capabilities
docs/                 # Comprehensive documentation and tutorials
examples/
scripts/
tests/
```

## Simulation Backends

| | Sequential-Modular (SM) | Equation-Oriented (EO) |
|---|---|---|
| **Entry point** | `run_idaes(flowsheet)` | `run_eo(flowsheet)` |
| **Solver** | Unit-by-unit Reaktoro equilibrium | Simultaneous IPOPT NLP |
| **Strength** | Rigorous thermodynamics | Speed, gradient-based optimization |
| **Recycles** | Not yet supported | Not yet supported |
| **GDP capable** | Exhaustive enumeration only | Native Pyomo.GDP (Big-M → MINLP) |

## MCP server (experimental)

This repo includes an **MCP server** under `mcp_server/` that exposes tools:

- `simulate_flowsheet` – run an IDAES-backed sim and return KPIs
- `simulate_eo` – run an EO-backed sim (IPOPT) and return KPIs
- `run_speciation` – Reaktoro speciation for a stream
- `estimate_cost` – TEA/LCA computations
- `optimize_flowsheet` – Rigorous Bayesian Optimization (SingleTaskGP) of unit parameters against target KPIs
- `optimize_topology` – GDP superstructure optimization via `solve_gdp_eo()`

### Run locally (stdio)

```bash
# Activate your env
conda activate rkt   # Ensure reaktoro and ipopt are installed

# Install the package (and dev tools)
python -m pip install -e ".[dev]"

# Install BoTorch for Bayesian Optimizer
conda install botorch -c conda-forge

# Install IPOPT for EO solver
conda install ipopt -c conda-forge

# Install an MCP SDK
python -m pip install mcp

# Start the MCP server (stdio transport)
python mcp_server/server.py
```

## What's next
- Full integration of AI Agents to holistically synthesize multi-unit operation pathways natively.
- Deep Learning Kinetics to explicitly model rate-dependent non-equilibrium precipitation logic.
- Multi-objective GDP optimization (Pareto front of OPEX vs purity).
- Recycle convergence for both SM and EO solvers.
- Expand TEA/LCA parameter databases beyond baseline hydrometallurgical proxies.
