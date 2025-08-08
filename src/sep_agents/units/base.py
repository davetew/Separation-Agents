
from __future__ import annotations
from typing import Dict, Any, Tuple
from pydantic import BaseModel
from ..dsl.schemas import Stream

class UnitResult(BaseModel):
    outputs: Dict[str, Stream]
    kpis: Dict[str, float]  # local KPIs for this unit

class BaseUnit:
    def __init__(self, unit_id: str, params: Dict[str, float] | None = None):
        self.unit_id = unit_id
        self.params = params or {}

    def simulate(self, **streams: Stream) -> UnitResult:
        raise NotImplementedError
