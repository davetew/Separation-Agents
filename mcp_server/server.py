import asyncio
try:
    # Replace with your MCP SDK import. This is a placeholder import.
    from mcp.server import Server
except Exception as e:
    raise RuntimeError(
        "MCP server SDK not installed. Install an MCP SDK (e.g., `pip install mcp`) "
        "or adjust imports to your chosen MCP library."
    ) from e

from capabilities.simulate_flowsheet import simulate_flowsheet
from capabilities.run_speciation import run_speciation
from capabilities.estimate_cost import estimate_cost
from capabilities.optimize_flowsheet import optimize_flowsheet

server = Server(name="MineralSeparationMCP", version="0.1.0")

server.register_tool(
    name="simulate_flowsheet",
    description="Run a mineral separation flowsheet in IDAES and return KPIs.",
    handler=simulate_flowsheet
)

server.register_tool(
    name="run_speciation",
    description="Run thermodynamic speciation using Reaktoro for a given stream.",
    handler=run_speciation
)

server.register_tool(
    name="estimate_cost",
    description="Estimate TEA/LCA costs for a flowsheet simulation result.",
    handler=estimate_cost
)

server.register_tool(
    name="optimize_flowsheet",
    description="Run a BoTorch (or simple) optimization on a flowsheet design.",
    handler=optimize_flowsheet
)

if __name__ == "__main__":
    # stdio is convenient for local MCP clients (including ChatGPT MCP)
    asyncio.run(server.serve_stdio())
