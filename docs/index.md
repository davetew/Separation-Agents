# Mineral Separation Agents: Documentation

Welcome to the **Mineral Separation Agents** documentation. This project is a multi-agent, LLM-guided framework for simulating, evaluating, and optimizing continuous mineral separation and hydrometallurgical flowsheets. It supports three simulation backends (**Sequential-Modular** via IDAES/Reaktoro, **Equation-Oriented** via native Pyomo/IPOPT, and **JAX Differentiable** via pure-JAX Gibbs energy minimization), and can perform **superstructure optimization** via Generalized Disjunctive Programming (GDP).

## Table of Contents

1. [Technical Report: REE Separation Process Modeling](technical_report.md)
   - REE Separation Process Overview and Chemistry
   - Sequential-Modular Flowsheet Solver Architecture
   - Equation-Oriented Flowsheet Solver (EO)
   - JAX-Based Equilibrium Solver (GEM + HKF/HP/PR EOS)
   - Generalized Disjunctive Programming (GDP) for Superstructure Optimization
   - Thermodynamic Property Models (Reaktoro / SUPCRTBL / Custom Species)
   - Techno-Economic Analysis (TEA) and Life Cycle Assessment (LCA)
   - JAX-Based Differentiable Cost Estimation (Autodiff Sensitivity)
   - Bayesian Optimization with BoTorch (SingleTaskGP + LogEI)
2. [Tutorial: Building and Optimizing Flowsheets](tutorial.md)
   - Defining Feeds and Units
   - SM Simulation (IDAES + Reaktoro)
   - EO Simulation (Pyomo + IPOPT)
   - Superstructure Optimization with GDP
   - BoTorch Parameter Optimization
   - JAX Equilibrium Solver
   - JAX Differentiable Cost Analysis
   - Agent Workflows (`/valorize` Pipeline)
3. [Gaps and Limitations](gaps_limitations.md)

## Core Architecture

This framework exposes physics-grounded tools via a Model Context Protocol (MCP) server with 14 tools. LLM agents can query thermodynamics, construct multi-stage flowsheets, tally proxy operating expenses (OPEX) and emissions (LCA), and orchestrate formal optimization. Seven agent workflows automate the full valorization pipeline from raw material characterization to polished LaTeX reports and Beamer presentations.

### Three-Tier Compute Hierarchy

```
Tier 1: Reaktoro Speciation        → Rigorous Gibbs energy minimization (SUPCRTBL)
Tier 2: IDAES/GDP Process Synthesis → Flowsheet simulation + topology optimization
Tier 3: JAX TEA                     → Differentiable cost model with autodiff sensitivity
```

### Simulation Backends

- **Sequential-Modular (SM)**: Each unit is solved independently in topological order using **Reaktoro** for rigorous Gibbs-energy-minimization equilibrium speciation and **IDAES/Pyomo** for block-based stream propagation.
- **Equation-Oriented (EO)**: The entire flowsheet is formulated as a single simultaneous NLP using native Pyomo expressions, solved by **IPOPT**. This enables gradient-based optimization of continuous design variables.
- **JAX Differentiable**: A pure-JAX implementation of constrained Gibbs energy minimization with extended Debye-Hückel activity coefficients, HKF EOS for aqueous species, Holland-Powell for minerals, and Peng-Robinson for gases. Enables end-to-end differentiable speciation and cost analysis via `jax.grad`.

### Optimization

- **BoTorch Bayesian Optimization**: Gaussian Process (SingleTaskGP) surrogate with Log Expected Improvement for black-box optimization of continuous parameters.
- **GDP Superstructure Optimization**: Pyomo.GDP with Big-M transformation converts topology selection (optional/alternative units) into a mixed-integer NLP, enabling simultaneous optimization of process topology and operating parameters.
- **Enumerative GDP Solver**: `gdp_builder.py` + `gdp_solver.py` enumerate all feasible topologies from a superstructure definition and rank them by objective (OPEX, recovery, net value).
- **JAX Autodiff Cost Sensitivity**: `jax_tea.cost_sensitivity()` computes ∂(EAC)/∂(parameter) for all continuous design variables in a single backward pass.

### Agent Workflows

Seven slash-command workflows under `.agent/workflows/` automate the full valorization pipeline:

| Workflow | Description |
|----------|-------------|
| `/valorize` | End-to-end orchestrator chaining all sub-workflows |
| `/resource-characterization` | Feed speciation, value stream identification, commodity pricing |
| `/superstructure-selection` | GDP superstructure construction and topology screening |
| `/process-optimization` | IDAES simulation, BoTorch BO, JAX TEA sensitivity analysis |
| `/cost-analysis` | CAPEX/OPEX/EAC/revenue/net value reporting |
| `/valorization-report` | LaTeX technical report generation |
| `/valorization-presentation` | Branded Beamer slide deck (sf-branding + lualatex) |
