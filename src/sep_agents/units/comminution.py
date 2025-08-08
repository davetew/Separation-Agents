
from __future__ import annotations
from typing import Dict
from .base import BaseUnit, UnitResult
from ..dsl.schemas import Stream, PSD
import numpy as np

class Mill(BaseUnit):
    """Very simplified mill: shifts PSD to finer sizes; energy ~ Bond-like placeholder."""
    def simulate(self, feed: Stream) -> UnitResult:
        if not feed.psd:
            return UnitResult(outputs={"product": feed}, kpis={"E_specific_kWhpt": 0.0})
        bins = np.array(feed.psd.bins_um, dtype=float)
        mf = np.array(feed.psd.mass_frac, dtype=float)
        factor = self.params.get("fineness_factor", 0.7)  # <1 makes finer
        new_bins = bins * factor
        new_psd = PSD(bins_um=new_bins.tolist(), mass_frac=mf.tolist())
        prod = feed.copy(update={"psd": new_psd})
        e = float(self.params.get("E_specific_kWhpt", 8.0))
        return UnitResult(outputs={"product": prod}, kpis={"E_specific_kWhpt": e})
