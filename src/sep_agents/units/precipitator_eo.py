"""
EO Precipitator Unit Model
============================

Simplified equilibrium precipitation formulated as Pyomo constraints
for equation-oriented optimisation.

Two modelling modes are available:

1. **Recovery-fraction mode** (default, numerically robust):
   Each REE component has a recovery fraction ``R[j]`` ∈ [0, 1] that
   can be fixed (simulation) or optimised.  This avoids log/exp
   numerical issues entirely.

2. **Ksp mode** (optional, use with caution):
   Uses a solubility-product with smooth ``max(0, ...)`` approximation.
   Requires careful scaling of Ksp and eps parameters.

Design variables (Var, optimisable)
------------------------------------
* ``recovery[j]``  — precipitation recovery per REE component [0–1]
* ``reagent_flow`` — reagent molar flow [mol/s]

Constraints
-----------
* solid_j = feed_j × recovery_j
* barren_j = feed_j × (1 − recovery_j)
* Mass balance: feed = solid + barren
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

from pyomo.environ import (
    Block,
    Constraint,
    Expression,
    Param,
    Set,
    Var,
    NonNegativeReals,
    PositiveReals,
    value,
)

_log = logging.getLogger(__name__)

# Default recovery fractions (high for hydroxide precipitation at high pH)
_DEFAULT_RECOVERY = {
    "La": 0.95,
    "Ce": 0.97,
    "Pr": 0.96,
    "Nd": 0.96,
    "Sm": 0.97,
    "Y":  0.98,
    "Dy": 0.97,
}


def build_precipitator(
    parent_block,
    name: str,
    component_list,
    ree_components: Optional[set] = None,
    recovery_init: Optional[Dict[str, float]] = None,
    reagent_flow_init: float = 0.1,
    fix_recovery: bool = False,
):
    """Attach a precipitator block to *parent_block*.

    Creates Vars, Constraints and Expressions for a recovery-fraction
    based solid/liquid separation.

    Parameters
    ----------
    parent_block : Pyomo Block
        Parent to attach to.
    name : str
        Block name.
    component_list : iterable
        All components (REE + background).
    ree_components : set, optional
        Which components are REE (subject to precipitation).
    recovery_init : dict, optional
        Initial recovery fractions per REE component.
    reagent_flow_init : float
        Reagent molar flow [mol/s].
    fix_recovery : bool
        If True, fix recovery fractions (simulation mode).

    Returns
    -------
    Pyomo Block
    """
    if ree_components is None:
        ree_components = set(_DEFAULT_RECOVERY.keys())
    if recovery_init is None:
        recovery_init = dict(_DEFAULT_RECOVERY)

    blk = Block(concrete=True)
    parent_block.add_component(name, blk)

    comp_list = list(component_list)
    ree_list = [c for c in comp_list if c in ree_components]
    bg_list = [c for c in comp_list if c not in ree_components]

    blk.component_list = Set(initialize=comp_list)
    blk.ree_set = Set(initialize=ree_list)
    blk.background_set = Set(initialize=bg_list)

    # ── Design variables ────────────────────────────────────────
    blk.recovery = Var(
        blk.ree_set,
        initialize={c: recovery_init.get(c, 0.95) for c in ree_list},
        bounds=(0.0, 1.0),
        domain=NonNegativeReals,
        doc="Precipitation recovery fraction [0-1]",
    )
    blk.reagent_flow = Var(
        initialize=reagent_flow_init,
        bounds=(0.001, 10.0),
        domain=PositiveReals,
        doc="Reagent molar flow [mol/s]",
    )

    if fix_recovery:
        for j in blk.ree_set:
            blk.recovery[j].fix()
    # Fix reagent flow (physical input, not optimised by default)
    blk.reagent_flow.fix()

    # ── Stream variables ────────────────────────────────────────
    blk.feed = Var(
        blk.component_list,
        initialize=0.01,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Inlet component flow [mol/s]",
    )
    blk.solid = Var(
        blk.component_list,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Solid (precipitate) outlet [mol/s]",
    )
    blk.barren = Var(
        blk.component_list,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Barren liquor outlet [mol/s]",
    )

    # ── Constraints ─────────────────────────────────────────────

    # REE solid = feed × recovery
    def _ree_solid_rule(b, j):
        return b.solid[j] == b.feed[j] * b.recovery[j]
    blk.ree_solid_eq = Constraint(
        blk.ree_set,
        rule=_ree_solid_rule,
        doc="REE precipitation: solid_j = feed_j * R_j",
    )

    # REE barren = feed × (1 - recovery)
    def _ree_barren_rule(b, j):
        return b.barren[j] == b.feed[j] * (1.0 - b.recovery[j])
    blk.ree_barren_eq = Constraint(
        blk.ree_set,
        rule=_ree_barren_rule,
        doc="REE remaining in solution",
    )

    # Background species pass through to barren (no precipitation)
    def _bg_barren_rule(b, j):
        return b.barren[j] == b.feed[j]
    blk.bg_barren_eq = Constraint(
        blk.background_set,
        rule=_bg_barren_rule,
        doc="Background species pass through",
    )

    def _bg_solid_zero_rule(b, j):
        return b.solid[j] == 0.0
    blk.bg_solid_zero = Constraint(
        blk.background_set,
        rule=_bg_solid_zero_rule,
        doc="No precipitation of background species",
    )

    # ── Convenience expressions ─────────────────────────────────
    blk.feed_total = Expression(
        expr=sum(blk.feed[j] for j in blk.component_list))
    blk.solid_total = Expression(
        expr=sum(blk.solid[j] for j in blk.component_list))
    blk.barren_total = Expression(
        expr=sum(blk.barren[j] for j in blk.component_list))

    # Overall REE recovery = solid_ree / feed_ree
    blk.ree_recovery = Expression(
        expr=sum(blk.solid[j] for j in blk.ree_set)
             / (sum(blk.feed[j] for j in blk.ree_set) + 1e-12),
        doc="Overall REE recovery to solid",
    )

    return blk
