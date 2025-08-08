
"""Adapters to call IDAES/Pyomo models.

These are stubs. Implement:
- build_flowsheet(fs: Flowsheet) -> ConcreteModel
- solve(model) -> results
- extract_streams(results) -> Dict[str, Stream]
"""
from typing import Dict
from ..dsl.schemas import Flowsheet, Stream

def run_idaes(flowsheet: Flowsheet) -> Dict[str, Stream]:
    # TODO: translate Flowsheet -> Pyomo/IDAES model and solve
    return {s.name: s for s in flowsheet.streams}
