
import sys
from sep_agents.orchestrator.planner import Orchestrator, load_flowsheet

def main(path: str):
    fs = load_flowsheet(path)
    orch = Orchestrator()
    res = orch.run_once(fs)
    print(res)
    fs2 = orch.suggest(fs)
    res2 = orch.run_once(fs2)
    print(res2)

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "examples/steel_slag_minimal.yaml")
