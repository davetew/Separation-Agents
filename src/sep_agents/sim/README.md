# Simulation Agents (`sim`)

This module contains agents responsible for the core physical and chemical simulations of the process.

## Components

*   **`equilibrium_agent.py`**:
    *   **Class**: `EquilibriumAgent`
    *   **Function**: Performs chemical equilibrium calculations using Reaktoro. It supports parameter sweeps (e.g., varying Temperature or Pressure) and handles database-specific species naming.
    *   **Key Features**: Robust failure handling, multi-database support (SUPCRT, PHREEQC).

*   **`kinetics_agent.py`**:
    *   **Class**: `KineticsAgent`
    *   **Function**: Simulates reaction kinetics and time-dependent processes.

*   **`reactor_design_agent.py`**:
    *   **Class**: `ReactorDesignAgent`
    *   **Function**: Sizes reactors based on required residence times and throughputs.

*   **`reaktoro_adapter.py`**:
    *   **Function**: Helper functions and adapters for interfacing with the Reaktoro library.
