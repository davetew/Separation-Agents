
from __future__ import annotations
from typing import Dict, Any
import random

class SimpleOptimizer:
    """Placeholder that perturbs numeric params to explore design space."""
    def suggest_edit(self, unit_params: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in unit_params.items():
            if isinstance(v, (int, float)):
                out[k] = float(v) * (0.9 + 0.2 * random.random())
            else:
                out[k] = v
        return out
