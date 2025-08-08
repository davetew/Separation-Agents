
from sep_agents.orchestrator.planner import load_flowsheet, Orchestrator

def test_load_and_run():
    fs = load_flowsheet("examples/steel_slag_minimal.yaml")
    orch = Orchestrator()
    res = orch.run_once(fs)
    assert res["status"] == "ok"
