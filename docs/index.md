# Mineral Separation Agents: Documentation

Welcome to the **Mineral Separation Agents** documentation. This project is a multi-agent, LLM-guided framework for simulating, evaluating, and optimizing continuous mineral separation and hydrometallurgical flowsheets, built on top of high-fidelity simulators (IDAES/Pyomo and Reaktoro).

## Table of Contents

1. [Tutorial: Building and Optimizing Flowsheets](tutorial.md)
   - Defining Feeds and Units
   - Integrating TEA and LCA
   - BoTorch Parameter Optimization
2. [Gaps and Limitations](gaps_limitations.md)
   - Deep BoTorch Parameter Optimization
   - Deep Learning Kinetics
   - Database and Cost Model Fidelity

## Core Architecture

This framework exposes physics-grounded tools via a Model Context Protocol (MCP) server. LLM agents can query thermodynamics, construct multi-stage flowsheets, tally proxy operating expenses (OPEX) and emissions (LCA), and orchestrate formal Bayesian Optimization.

- **Reaktoro:** Used for high-speed, rigorous equilibrium speciation calculations, natively embedded with custom sets of elements (e.g., Rare Earth Elements, Oxalates).
- **IDAES (Pyomo):** Used as the core flowsheet block sequence solver. Streams and unit states are modeled sequentially to simulate separation efficiencies, split fractions, cascading recycle loops, etc.
- **BoTorch:** Utilizes Gaussian Process (SingleTaskGP) surrogate models and Expected Improvement (EI) to tune unit operation block parameters against exact target KPIs (e.g., minimizing dollars per kg or maximizing extraction percentage).
