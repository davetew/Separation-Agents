# Mineral Separation Agents: Documentation

Welcome to the **Mineral Separation Agents** documentation. This project is a multi-agent, LLM-guided framework for simulating, evaluating, and optimizing continuous mineral separation and hydrometallurgical flowsheets. It supports two simulation backends (**Sequential-Modular** via IDAES/Reaktoro and **Equation-Oriented** via native Pyomo/IPOPT) and can perform **superstructure optimization** via Generalized Disjunctive Programming (GDP).

## Table of Contents

1. [Technical Report: REE Separation Process Modeling](technical_report.md)
   - REE Separation Process Overview and Chemistry
   - Sequential-Modular Flowsheet Solver Architecture
   - Equation-Oriented Flowsheet Solver (EO)
   - Generalized Disjunctive Programming (GDP) for Superstructure Optimization
   - Thermodynamic Property Models (Reaktoro / SUPCRTBL / Custom Species)
   - Techno-Economic Analysis (TEA) and Life Cycle Assessment (LCA)
   - Bayesian Optimization with BoTorch (SingleTaskGP + LogEI)
2. [Tutorial: Building and Optimizing Flowsheets](tutorial.md)
   - Defining Feeds and Units
   - SM Simulation (IDAES + Reaktoro)
   - EO Simulation (Pyomo + IPOPT)
   - Superstructure Optimization with GDP
   - BoTorch Parameter Optimization
3. [Gaps and Limitations](gaps_limitations.md)

## Core Architecture

This framework exposes physics-grounded tools via a Model Context Protocol (MCP) server. LLM agents can query thermodynamics, construct multi-stage flowsheets, tally proxy operating expenses (OPEX) and emissions (LCA), and orchestrate formal optimization.

### Simulation Backends

- **Sequential-Modular (SM)**: Each unit is solved independently in topological order using **Reaktoro** for rigorous Gibbs-energy-minimization equilibrium speciation and **IDAES/Pyomo** for block-based stream propagation.
- **Equation-Oriented (EO)**: The entire flowsheet is formulated as a single simultaneous NLP using native Pyomo expressions, solved by **IPOPT**. This enables gradient-based optimization of continuous design variables.

### Optimization

- **BoTorch Bayesian Optimization**: Gaussian Process (SingleTaskGP) surrogate with Log Expected Improvement for black-box optimization of continuous parameters.
- **GDP Superstructure Optimization**: Pyomo.GDP with Big-M transformation converts topology selection (optional/alternative units) into a mixed-integer NLP, enabling simultaneous optimization of process topology and operating parameters.
