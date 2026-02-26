# Separation Agents (skeleton)

LLM-guided, physics-grounded **multi-agent** system that can propose, simulate, and optimize
mineral/metal separation flowsheets (specifically targeting Rare Earth Elements).

## Key ideas
- **Flowsheet DSL** (YAML/JSON) → validated into Pydantic models.
- **Unit op stubs** (comminution, cyclone, magnetic, flotation, hydromet, precipitator, solvent extraction).
- **Sim adapters** for IDAES/Pyomo and Reaktoro (equilbrium speciation with custom REE databases).
- **Techno-Economic Analysis (TEA) & Life Cycle Assessment (LCA)** embedded into the sequences, allowing for agent-driven OPEX and $CO_2$e evaluation.
- **Optimizer** (BoTorch Bayesian Optimization) for rapid multi-variable parameter optimization (e.g., minimizing OPEX via continuous unit parameter adjustments).
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

## Repo layout
```
src/sep_agents/
  dsl/                # schemas & loaders
  units/              # unit op stubs
  sim/                # adapters to physics engines (IDAES, Reaktoro)
  opt/                # optimization strategies (BoTorch)
  orchestrator/       # planner loop
  critic/             # feasibility & sanity checks
  properties/         # custom thermodynamic element databases
  cost/               # TEA/LCA capabilities
docs/                 # Comprehensive documentation and tutorials
examples/
scripts/
tests/
```
## MCP server (experimental)

This repo includes an **MCP server** under `mcp_server/` that exposes tools:

- `simulate_flowsheet` – run an IDAES-backed sim and return KPIs
- `run_speciation` – Reaktoro speciation for a stream
- `estimate_cost` – TEA/LCA computations
- `optimize_flowsheet` – Rigorous Bayesian Optimization (SingleTaskGP) of unit parameters against target KPIs

### Run locally (stdio)

```bash
# Activate your env
conda activate rkt   # Ensure reaktoro is installed in this environment

# Install the package (and dev tools)
python -m pip install -e ".[dev]"

# Install Botrch for Bayesian Optimizer
conda install botorch -c conda-forge

# Install an MCP SDK
python -m pip install mcp

# Start the MCP server (stdio transport)
python mcp_server/server.py
```

## What’s next
- Full integration of AI Agents to holistically synthesize multi-unit operation pathways natively.
- Deep Learning Kinetics to explicitly model rate-dependent non-equilibrium precipitation logic.
- Expand TEA/LCA parameter databases beyond baseline hydrometallurgical proxies.
