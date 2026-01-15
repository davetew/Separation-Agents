# Orchestrator (`orchestrator`)

This module provides the high-level control logic for the agentic workflow.

## Components

*   **`orchestrator_agent.py`**: The "Manager" agent. It takes a high-level user request (e.g., "Design a process to extract Lithium") and decomposes it into tasks for subordinate agents (Simulation, Cost, etc.).
*   **`planner.py`**: Logic for generating execution plans and dependency graphs for complex workflows.
