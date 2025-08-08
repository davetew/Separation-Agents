# mcp_server/server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MineralSeparationMCP")

# Tools
@mcp.tool()
def simulate_flowsheet(flowsheet_yaml: str) -> dict:
    from sep_agents.orchestrator.planner import load_flowsheet, Orchestrator
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkstemp(suffix=".yaml")[1])
    p.write_text(flowsheet_yaml)
    fs = load_flowsheet(str(p))
    return Orchestrator().run_once(fs)

@mcp.tool()
def run_speciation(stream: dict) -> dict:
    try:
        import reaktoro  # optional
    except ImportError:
        return {"status": "error", "error": "Reaktoro not installed"}
    from sep_agents.dsl.schemas import Stream
    s = Stream(**stream)
    # TODO: Reaktoro call
    return {"status": "ok", "stream_out": s.dict()}

@mcp.tool()
def estimate_cost(kpis: dict) -> dict:
    from sep_agents.cost.tea import estimate_opex_kwh_reagents
    from sep_agents.cost.lca import estimate_co2e
    return {"status": "ok", "OPEX": estimate_opex_kwh_reagents(kpis), "CO2e": estimate_co2e(kpis)}

@mcp.tool()
def optimize_flowsheet(flowsheet: dict) -> dict:
    from sep_agents.opt.bo import SimpleOptimizer
    from sep_agents.dsl.schemas import Flowsheet
    fs = Flowsheet(**flowsheet)
    if fs.units:
        fs.units[0].params = SimpleOptimizer().suggest_edit(fs.units[0].params)
    return {"status": "ok", "flowsheet": fs.dict()}

if __name__ == "__main__":
    mcp.run()  # stdio by default; see docs for other transports
