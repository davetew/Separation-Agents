# Separation Agents

LLM-guided, physics-grounded **multi-agent** system that can propose, simulate, and optimize
mineral/metal separation and valorization flowsheets. Originally targeting Rare Earth Elements (REEs), the framework now supports general mineral valorization workflows including steel slag, mine tailings, and industrial byproducts.

## Key Ideas

- **Flowsheet DSL** (YAML/JSON) → validated into Pydantic models.
- **Unit op stubs** (comminution, cyclone, magnetic, flotation, hydromet, precipitator, solvent extraction, ion exchange, leach reactor, mixer, separator).
- **Three simulation backends**:
  - **Sequential-Modular (SM)** via IDAES/Pyomo + Reaktoro equilibrium speciation
  - **Equation-Oriented (EO)** via native Pyomo (simultaneous NLP solved by IPOPT)
  - **JAX Differentiable** — pure-JAX Gibbs energy minimization solver with HKF, Holland-Powell, and Peng-Robinson EOS support
- **Generalized Disjunctive Programming (GDP)** for superstructure optimization — simultaneously selects the best topology *and* optimal continuous parameters.
- **JAX-Based Differentiable TEA** — fully differentiable CAPEX/OPEX/EAC cost model with `jax.grad` autodiff sensitivity analysis.
- **TEA & LCA** embedded into simulation sequences, enabling agent-driven OPEX and CO₂e evaluation.
- **Optimizers**: BoTorch Bayesian Optimization (parameter tuning) + Pyomo.GDP/IPOPT (topology + parameter co-optimization).
- **Agent Workflows** — seven slash-command workflows (`/valorize`, `/resource-characterization`, `/superstructure-selection`, `/process-optimization`, `/cost-analysis`, `/valorization-report`, `/valorization-presentation`) that chain the full pipeline from raw material to polished LaTeX report and Beamer presentation.
- **Orchestrator** loop that ties proposal → simulate → critique → optimize.

## Quickstart

```bash
# Create and activate the conda environment
conda create -n rkt python=3.11
conda activate rkt

# Install core package
pip install -e .

# Install Reaktoro, IPOPT, and BoTorch via conda-forge
conda install reaktoro ipopt botorch -c conda-forge

# Optional: JAX for differentiable solvers
pip install jax jaxlib jaxopt

# Run a demo flowsheet
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

### JAX Cost Sensitivity Demo

```python
import jax.numpy as jnp
from sep_agents.cost.jax_tea import total_annualized_cost, cost_sensitivity

params = {
    "ore_throughput_tpd": jnp.array(500.0),
    "residence_time_h": jnp.array(4.0),
    "acid_consumption_kg_t": jnp.array(50.0),
    "operating_temp_c": jnp.array(250.0),
}
eac = total_annualized_cost(params)
grads = cost_sensitivity(params)  # d(EAC)/d(param) via jax.grad
print(f"EAC: ${float(eac):,.0f}/yr")
for k, v in grads.items():
    print(f"  d(EAC)/d({k}) = {float(v):,.2f}")
```

## Repo Layout

```
src/sep_agents/
  dsl/                # Schemas & loaders (Flowsheet, Stream, UnitOp, Superstructure)
  units/              # Unit op models (SM stubs + EO Pyomo blocks)
  sim/                # Simulation backends
    idaes_adapter.py       # Sequential-Modular (SM) solver
    eo_flowsheet.py        # Equation-Oriented (EO) solver
    jax_equilibrium.py     # JAX Gibbs energy minimization solver
    jax_hkf.py             # HKF EOS for aqueous species
    jax_holland_powell.py  # Holland-Powell EOS for minerals
    jax_peng_robinson.py   # Peng-Robinson EOS for gases
    equilibrium_agent.py   # Parameter sweep agent
    reaktoro_adapter.py    # Reaktoro bridge
  opt/                # Optimization
    bo.py                  # BoTorch Bayesian Optimization
    gdp_eo.py              # Pyomo.GDP MINLP (Big-M transformation)
    gdp_builder.py         # Superstructure → Configuration enumerator
    gdp_solver.py          # Enumerative GDP topology solver
  cost/               # Economics & sustainability
    tea.py                 # Proxy OPEX estimation
    lca.py                 # Lifecycle CO₂e estimation
    jax_tea.py             # Differentiable TEA (JAX autodiff)
    economics_agent.py     # Revenue & net value agent
  properties/         # Custom thermodynamic element databases
  orchestrator/       # Planner loop
  critic/             # Feasibility & sanity checks
  report.py           # Auto-generated Markdown analysis reports
mcp_server/           # MCP server exposing 14 tools
.agent/workflows/     # Agent slash-command workflows
examples/             # Example flowsheets and scripts
scripts/              # Benchmarks, validation, and utilities
docs/                 # Technical report, tutorial, and references
reports/              # Auto-generated analysis reports
presentations/        # Branded Beamer presentations (sf-branding)
tests/
```

## Simulation Backends

| | Sequential-Modular (SM) | Equation-Oriented (EO) | JAX Differentiable |
|---|---|---|---|
| **Entry point** | `run_idaes(flowsheet)` | `run_eo(flowsheet)` | `JaxEquilibriumSolver.speciate()` |
| **Solver** | Unit-by-unit Reaktoro equilibrium | Simultaneous IPOPT NLP | Constrained GEM via `jaxopt.LBFGSB` |
| **Strength** | Rigorous thermodynamics | Speed, gradient-based opt | End-to-end differentiability |
| **EOS** | Reaktoro (SUPCRTBL) | Proxy models | HKF + Holland-Powell + Peng-Robinson |
| **Recycles** | Not yet supported | Not yet supported | N/A |
| **GDP capable** | Exhaustive enumeration | Native Pyomo.GDP (Big-M → MINLP) | N/A |
| **Typical speed** | ~10–20s (3-unit) | ~1–2s (3-unit) | ~0.1s (speciation) |

## Agent Workflows

The `.agent/workflows/` directory contains seven slash-command workflows that automate the full valorization pipeline. The top-level `/valorize` workflow chains all stages:

```
/valorize
  ├── /resource-characterization   → Feed speciation & value stream identification
  ├── /superstructure-selection     → GDP superstructure construction & topology screening
  ├── /process-optimization         → IDAES simulation, BoTorch BO, JAX TEA sensitivity
  ├── /cost-analysis                → CAPEX/OPEX/EAC/revenue/net value reporting
  ├── /valorization-report          → LaTeX technical report generation
  └── /valorization-presentation    → Branded Beamer slide deck (sf-branding + lualatex)
```

All Python commands in these workflows run inside the `rkt` conda environment.

## MCP Server

This repo includes an **MCP server** under `mcp_server/` that exposes 14 tools organized by category:

### Speciation & Thermodynamics
| Tool | Description |
|------|-------------|
| `run_speciation` | Run Reaktoro speciation for a stream |
| `speciate_ree_stream` | Compute REE speciation at equilibrium (pH, Eh, species distribution) |
| `evaluate_separation_factor` | Calculate separation factor β(A/B) between two REE elements |
| `get_stream_analysis` | Analyse a solved stream state (species ranking, pH, Eh) |
| `perform_sweep` | Sweep a parameter over a range and return speciation results |

### Flowsheet Simulation
| Tool | Description |
|------|-------------|
| `simulate_flowsheet` | Run a full SM flowsheet via the orchestrator |
| `run_idaes_flowsheet` | Build and solve an IDAES flowsheet with Reaktoro equilibrium |
| `build_ree_flowsheet` | Generate a YAML flowsheet with feed → SX → precipitation |
| `simulate_sx_cascade` | Simulate a multi-stage SX cascade via IDAES |

### GDP Superstructure Optimization
| Tool | Description |
|------|-------------|
| `list_superstructures_tool` | List available superstructure templates |
| `evaluate_topology` | Evaluate a specific topology from a superstructure |
| `optimize_superstructure_tool` | Enumerate + rank topologies, optional BO on continuous params |

### Optimization & Economics
| Tool | Description |
|------|-------------|
| `optimize_flowsheet` | BoTorch Bayesian Optimization of flowsheet parameters |
| `estimate_cost` | Estimate OPEX and CO₂e from unit KPIs |

### Run Locally (stdio)

```bash
conda activate rkt
pip install -e ".[dev]"
pip install mcp
python mcp_server/server.py
```

## What's Next

- Full integration of AI Agents to holistically synthesize multi-unit operation pathways natively.
- Deep Learning Kinetics to explicitly model rate-dependent non-equilibrium precipitation logic.
- Multi-objective GDP optimization (Pareto front of OPEX vs purity).
- Recycle convergence for both SM and EO solvers.
- Expand TEA/LCA parameter databases beyond baseline hydrometallurgical proxies.
- Expand JAX EOS coverage (additional aqueous species, organic-phase models).
