
from .base import BaseUnit, UnitResult
from ..dsl.schemas import Stream

class Flotation(BaseUnit):
    def simulate(self, feed: Stream) -> UnitResult:
        # Placeholder: split to froth/tail
        k = float(self.params.get("k", 0.5))
        return UnitResult(outputs={"froth": feed.copy(), "tail": feed.copy()}, kpis={"k": k})
