from sep_agents.orchestrator.planner import load_flowsheet, Orchestrator
import tempfile, pathlib

async def simulate_flowsheet(params: dict):
    """
    params = {
      "flowsheet_yaml": "<YAML string>"
    }
    """
    yaml_str = params.get("flowsheet_yaml", "")
    if not yaml_str:
        return {"status": "error", "error": "flowsheet_yaml missing"}

    tmp = pathlib.Path(tempfile.mkstemp(suffix=".yaml")[1])
    tmp.write_text(yaml_str)

    fs = load_flowsheet(str(tmp))
    orch = Orchestrator()
    res = orch.run_once(fs)
    return res
