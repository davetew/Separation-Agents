
from .base import BaseUnit, UnitResult
from ..dsl.schemas import Stream

class Leach(BaseUnit):
    def simulate(self, feed: Stream) -> UnitResult:
        # Placeholder: no chemistry yet
        ext = float(self.params.get("extraction", 0.6))
        return UnitResult(outputs={"pregnant": feed.copy(), "residue": feed.copy()}, kpis={"ext": ext})
