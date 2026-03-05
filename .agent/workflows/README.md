# Workflows

All workflows for this repository are located in `.agent/workflows/`.

> [!IMPORTANT]
> All commands must be run in the **`rkt` conda environment**: `conda run --no-capture-output -n rkt <command>`

## Available Workflows

| Slash Command | Description |
|---|---|
| `/valorize` | End-to-end valorization analysis — chains all sub-workflows from raw material to final report |
| `/resource-characterization` | Characterize a raw material feed, compute equilibrium speciation, and identify value streams |
| `/superstructure-selection` | Select or construct GDP superstructures for valorization of a characterized raw material |
| `/process-optimization` | Optimize GDP superstructures via IDAES simulation, BoTorch, and JAX TEA |
| `/cost-analysis` | Present cost analysis results to the user |
| `/valorization-report` | Generate a comprehensive LaTeX technical report summarizing the valorization analysis |
| `/valorization-presentation` | Generate a LaTeX Beamer presentation summarizing a valorization study |

## Workflow Chain

The `/valorize` command chains the sub-workflows in order:

1. `/resource-characterization` → Feed analysis & speciation
2. `/superstructure-selection` → GDP superstructure construction
3. `/process-optimization` → IDAES simulation & BoTorch optimization
4. `/cost-analysis` → TEA/LCA metrics
5. `/valorization-report` → LaTeX report
6. `/valorization-presentation` → LaTeX Beamer slides
