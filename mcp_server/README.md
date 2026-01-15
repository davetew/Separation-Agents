# MCP Server

This directory contains the implementation of the Model Context Protocol (MCP) server for the Separation Agents. This server exposes the functionalities of the python package as "tools" that can be consumed by AI assistants (like Claude or Gemini).

## Structure

*   **`server.py`**: The entry point for the MCP server. It instantiates the `FastMCP` server and registers the available tools.
*   **`capabilities/`**: Contains the actual implementation of the exposed tools.
    *   `run_speciation.py`: Tool for performing chemical speciation checks.
    *   `simulate_flowsheet.py`: Tool for running full flowsheet simulations.
    *   `estimate_cost.py`: Tool for performing cost estimation.
    *   `optimize_flowsheet.py`: Tool for optimizing process parameters.
