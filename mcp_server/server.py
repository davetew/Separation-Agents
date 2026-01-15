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
        from sep_agents.sim.reaktoro_adapter import run_reaktoro
        from sep_agents.dsl.schemas import Stream
    except ImportError:
         return {"status": "error", "error": "Internal module import failed"}

    try:
        s = Stream(**stream)
        s_out = run_reaktoro(s)
        return {"status": "ok", "stream_out": s_out.dict()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def perform_sweep(initial_conditions: dict, param_name: str, values: list) -> dict:
    try:
        from sep_agents.sim.equilibrium_agent import EquilibriumAgent
        agent = EquilibriumAgent()
        df = agent.sweep(initial_conditions, param_name, values)
        return {"status": "ok", "results": df.to_dict(orient="records")}
    except Exception as e:
        return {"status": "error", "error": str(e)}

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
