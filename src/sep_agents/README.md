# Separation Agents Package

The `sep_agents` package provides a suite of AI-driven agents and tools for designing, simulating, and optimizing separation processes.

## Module Structure

*   **`sim/`**: **Simulation**. Agents for calculating chemical equilibrium, kinetics, and reactor design.
*   **`units/`**: **Unit Operations**. Models for specific industrial separation units (flotation, hydrometallurgy, comminution).
*   **`cost/`**: **Economics**. Techno-Economic Analysis (TEA) and Life Cycle Assessment (LCA) agents.
*   **`opt/`**: **Optimization**. Bayesian optimization and other strategies for process improvement.
*   **`orchestrator/`**: **Planning**. High-level agents that coordinate the design workflow.
*   **`properties/`**: **Thermodynamics**. Interfaces to backend property packages (Reaktoro, IDAES).
*   **`dsl/`**: **Data Structures**. Shared schemas and definitions for process data.
*   **`critic/`**: **Validation**. Agents that check constraints and validate simulation outputs.
