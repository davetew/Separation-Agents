# Tests

This directory contains the test suite for the `sep_agents` package.

## Running Tests

You can run the full test suite using `pytest`:

```bash
pytest
```

## Key Files

*   **`test_equilibrium_agent.py`**: Unit tests for the `EquilibriumAgent`. It validates:
    *   Species mapping across different thermodynamic databases (SUPCRT, PHREEQC, etc.).
    *   Robust handling of convergence failures during parameter sweeps.
    *   Setup and solution of equilibrium problems.

*   **`test_architecture.py`**: Integration tests verifying the structural integrity of the agent architecture.

*   **`test_reaktoro_prop.py`**: specific tests for the Reaktoro property interface.

*   **`test_smoke.py`**: Basic smoke tests to ensure the package handles imports and basic initializations correctly.
