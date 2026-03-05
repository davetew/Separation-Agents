# Separation-Agents — Agent Instructions

## Environment

> [!IMPORTANT]
> **All Python commands for this repository MUST be run inside the `rkt` conda environment.**
> Use `conda run --no-capture-output -n rkt` as the command prefix for all Python executions.

Example:
```bash
conda run --no-capture-output -n rkt python3 script.py
conda run --no-capture-output -n rkt pip install -e .
conda run --no-capture-output -n rkt pytest tests/
```

The `rkt` environment contains all required dependencies:
- `reaktoro` — thermodynamic equilibrium solver
- `pyomo` + `idaes-pse` — equation-oriented process modeling
- `jax` / `jaxlib` — differentiable equilibrium solver
- `botorch` — Bayesian optimization
- `numpy`, `scipy`, `networkx`, `pydantic`

### Source Path

The package source is in `src/`. When running scripts that import `sep_agents`, either:
- Install in dev mode: `conda run -n rkt pip install -e .`
- Or prepend: `PYTHONPATH=src` / `sys.path.insert(0, 'src')`

## Workflows

All workflows are located in `.agent/workflows/`. See the [README](.agent/workflows/README.md) for the full list.

## Lint Errors

Pyre2 (the IDE static type checker) will report many false-positive import errors because it cannot resolve packages installed in the `rkt` conda environment (e.g., `reaktoro`, `pyomo`, `idaes`, `numpy`). These are safe to ignore — they resolve at runtime within the `rkt` environment.
