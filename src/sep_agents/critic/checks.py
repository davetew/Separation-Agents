
from __future__ import annotations
from typing import Tuple
from ..dsl.schemas import Flowsheet

class Critic:
    def check(self, fs: Flowsheet) -> Tuple[bool, str]:
        # Minimal: ensure every unit has at least one input and one output id
        for u in fs.units:
            if not u.inputs or not u.outputs:
                return False, f"Unit {u.id} missing inputs/outputs"
        return True, "ok"
