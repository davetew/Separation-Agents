
from .base import BaseUnit, UnitResult
from ..dsl.schemas import Stream
import numpy as np

class Cyclone(BaseUnit):
    def simulate(self, feed: Stream) -> UnitResult:
        # Placeholder: split by fixed cut to underflow/overflow ratios
        split = float(self.params.get("underflow_split", 0.6))
        of = feed.copy()
        uf = feed.copy()
        # Track simple solids split KPI
        return UnitResult(outputs={"overflow": of, "underflow": uf}, kpis={"UF_split": split})
