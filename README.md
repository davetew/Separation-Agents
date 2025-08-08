
# Separation Agents (skeleton)

LLM-guided, physics-grounded **multi-agent** system that can propose, simulate, and optimize
mineral/metal separation flowsheets. This is a *skeleton repo* to get you started—plug in your
MCP server, high-fidelity models, and data as you go.

## Key ideas
- **Flowsheet DSL** (YAML/JSON) → validated into Pydantic models.
- **Unit op stubs** (comminution, cyclone, magnetic, flotation, hydromet, thickener).
- **Sim adapters** for IDAES/Pyomo and Reaktoro (hooks only here).
- **Optimizer** (Optuna/Bayesian) with multi-objective placeholders.
- **Orchestrator** loop that ties proposal → simulate → critique → optimize.
- **Critic** for balance/compatibility checks.
- **TEA/LCA** stubs with simple cost hooks.

## Quickstart
```bash
# Create and activate a virtual env
python -m venv .venv && source .venv/bin/activate

# Install (core)
pip install -e .

# Optional heavy deps
pip install 'sep-agents[thermo]'

# Run the tiny example loop
python scripts/run_loop.py examples/steel_slag_minimal.yaml
```

## Repo layout
```
src/sep_agents/
  dsl/                # schemas & loaders
  units/              # unit op stubs
  sim/                # adapters to physics engines
  opt/                # optimization strategies
  orchestrator/       # planner loop
  critic/             # feasibility & sanity checks
  cost/               # TEA/LCA stubs
examples/
scripts/
tests/
```

## What’s next
- Wire your **MCP** planner to call: flowsheet_synthesizer, simulator, optimizer.
- Replace unit stubs with calibrated physics/empirical hybrids.
- Introduce multi-fidelity surrogates (BoTorch) for slow sims.
- Add robust optimization across feed variability scenarios.
