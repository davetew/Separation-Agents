
from __future__ import annotations
from typing import Dict, Any, List
import yaml
from ..dsl.schemas import Flowsheet, UnitOp, Stream
from ..units.comminution import Mill
from ..units.cyclone import Cyclone
from ..units.magnetic import LIMS
from ..units.flotation import Flotation
from ..units.hydromet import Leach
from ..units.thickener import Thickener
from ..critic.checks import Critic
from ..opt.bo import SimpleOptimizer
from ..cost.tea import estimate_opex_kwh_reagents

UNIT_REGISTRY = {
    "mill": Mill,
    "cyclone": Cyclone,
    "lims": LIMS,
    "flotation": Flotation,
    "leach": Leach,
    "thickener": Thickener,
}

class Orchestrator:
    def __init__(self):
        self.critic = Critic()
        self.optimizer = SimpleOptimizer()

    def run_once(self, fs: Flowsheet) -> Dict[str, Any]:
        ok, msg = self.critic.check(fs)
        if not ok:
            return {"status": "invalid", "reason": msg}

        # Simulate each unit in order of declaration (no true topology yet)
        kpis_agg = {}
        for u in fs.units:
            cls = UNIT_REGISTRY.get(u.type)
            if not cls:
                continue
            unit = cls(u.id, u.params)
            # For now, assume single 'feed' stream exists and 'product' or similar exits.
            feed_name = u.inputs[0] if u.inputs else None
            feed_stream = next((s for s in fs.streams if s.name == feed_name), None)
            if not feed_stream:
                continue
            res = unit.simulate(feed=feed_stream)
            kpis_agg.update({f"{u.id}.{k}": v for k, v in res.kpis.items()})
        opex = estimate_opex_kwh_reagents(kpis_agg)
        return {"status": "ok", "kpis": kpis_agg, "OPEX_score": opex}

    def suggest(self, fs: Flowsheet) -> Flowsheet:
        # Apply a simple param tweak to the first unit
        if fs.units:
            u0 = fs.units[0]
            u0.params = self.optimizer.suggest_edit(u0.params)
        return fs

def load_flowsheet(path: str) -> Flowsheet:
    data = yaml.safe_load(open(path))
    units = [UnitOp(**u) for u in data["units"]]
    streams = [Stream(**s) for s in data.get("streams", [])]
    return Flowsheet(name=data.get("name", "fs"), units=units, streams=streams)
