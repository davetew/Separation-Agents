from sep_agents.opt.bo import SimpleOptimizer
from sep_agents.dsl.schemas import Flowsheet

async def optimize_flowsheet(params: dict):
    """
    params = {
      "flowsheet": { ... Flowsheet JSON ... }
    }
    """
    fs = Flowsheet(**params["flowsheet"])
    opt = SimpleOptimizer()
    if fs.units:
        fs.units[0].params = opt.suggest_edit(fs.units[0].params)
    return {"status": "ok", "flowsheet": fs.dict()}
