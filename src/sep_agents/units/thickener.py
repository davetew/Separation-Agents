
from .base import BaseUnit, UnitResult
from ..dsl.schemas import Stream

class Thickener(BaseUnit):
    def simulate(self, feed: Stream) -> UnitResult:
        under = feed.copy()
        overflow = feed.copy()
        return UnitResult(outputs={"underflow": under, "overflow": overflow}, kpis={"flux": 1.0})
