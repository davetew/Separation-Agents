"""
EO Solvent Extraction Unit Model
==================================

Single- and multi-stage McCabe-Thiele SX formulated as Pyomo
``Constraint`` objects for equation-oriented optimisation.

Each stage applies the distribution equilibrium::

    E_j = D_j · R / (1 + D_j · R)

where *D_j* is the distribution coefficient for component *j* and *R*
is the organic-to-aqueous phase ratio (O/A).

Design variables (``Var``, optimisable)
---------------------------------------
* ``D[j]`` — distribution coefficient per component
* ``OA_ratio`` — organic-to-aqueous volumetric ratio

Constraints
-----------
* Component mass balance on each stage
* Overall mass balance (sum of all components)
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
    units as pyunits,
)

_log = logging.getLogger(__name__)


class SXStageEO(Block):
    """A single SX stage as a Pyomo Block.

    Builds mass-balance constraints expressing McCabe-Thiele
    equilibrium for every component in the property package.

    Parameters
    ----------
    component_list : Pyomo Set
        The set of component names.
    D_init : dict, optional
        Initial distribution coefficients {component: float}.
    OA_init : float
        Initial organic-to-aqueous ratio.
    aqueous_species : set
        Species that remain entirely in the aqueous phase (e.g. H2O, HCl).
    """

    pass  # We use a factory function below for clarity


def build_sx_stage(
    parent_block,
    name: str,
    component_list,
    D_init: Optional[Dict[str, float]] = None,
    OA_init: float = 1.0,
    aqueous_background: Optional[set] = None,
):
    """Attach a single SX stage to *parent_block* as a sub-Block.

    Creates:
        ``parent_block.<name>.D[j]``            — distribution coefficients (Var)
        ``parent_block.<name>.OA_ratio``         — O/A ratio (Var)
        ``parent_block.<name>.feed[j]``          — inlet component flows (Var)
        ``parent_block.<name>.organic[j]``       — organic outlet flows (Var)
        ``parent_block.<name>.raffinate[j]``     — raffinate outlet flows (Var)
        ``parent_block.<name>.extraction[j]``    — extraction fraction (Expression)
        ``parent_block.<name>.mass_balance[j]``  — constraint: feed = org + raf
        ``parent_block.<name>.equilibrium[j]``   — constraint: org = feed * E

    Returns the sub-Block.
    """
    if D_init is None:
        D_init = {}
    if aqueous_background is None:
        aqueous_background = {"H2O", "HCl", "NaOH", "H2C2O4"}

    blk = Block(concrete=True)
    parent_block.add_component(name, blk)

    blk.component_list = Set(initialize=list(component_list))

    # ── Design variables ────────────────────────────────────────
    blk.D = Var(
        blk.component_list,
        initialize={c: D_init.get(c, 0.0) for c in component_list},
        bounds=(0, 100),
        domain=NonNegativeReals,
        doc="Distribution coefficient D_j",
    )
    blk.OA_ratio = Var(
        initialize=OA_init,
        bounds=(0.1, 10.0),
        domain=PositiveReals,
        doc="Organic-to-aqueous phase ratio",
    )

    # Fix D for aqueous background to zero
    for sp in aqueous_background:
        if sp in blk.D:
            blk.D[sp].fix(0.0)

    # ── Stream variables ────────────────────────────────────────
    blk.feed = Var(
        blk.component_list,
        initialize=1.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Inlet component flow [mol/s]",
    )
    blk.organic = Var(
        blk.component_list,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Organic outlet component flow [mol/s]",
    )
    blk.raffinate = Var(
        blk.component_list,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Raffinate outlet component flow [mol/s]",
    )

    # ── Extraction fraction (Expression) ────────────────────────
    def _extraction_rule(b, j):
        # E_j = D_j * R / (1 + D_j * R), handle D=0 gracefully
        return b.D[j] * b.OA_ratio / (1.0 + b.D[j] * b.OA_ratio)
    blk.extraction = Expression(
        blk.component_list,
        rule=_extraction_rule,
        doc="Extraction fraction E_j = D·R/(1+D·R)",
    )

    # ── Constraints ─────────────────────────────────────────────
    def _mass_balance_rule(b, j):
        return b.feed[j] == b.organic[j] + b.raffinate[j]
    blk.mass_balance = Constraint(
        blk.component_list,
        rule=_mass_balance_rule,
        doc="Component mass balance: feed = org + raf",
    )

    def _equilibrium_rule(b, j):
        return b.organic[j] == b.feed[j] * b.extraction[j]
    blk.equilibrium = Constraint(
        blk.component_list,
        rule=_equilibrium_rule,
        doc="Equilibrium: org_j = feed_j * E_j",
    )

    # ── Convenience expressions ─────────────────────────────────
    blk.feed_total = Expression(
        expr=sum(blk.feed[j] for j in blk.component_list))
    blk.organic_total = Expression(
        expr=sum(blk.organic[j] for j in blk.component_list))
    blk.raffinate_total = Expression(
        expr=sum(blk.raffinate[j] for j in blk.component_list))

    return blk


def build_sx_cascade(
    parent_block,
    name: str,
    n_stages: int,
    component_list,
    D_init: Optional[Dict[str, float]] = None,
    OA_init: float = 1.0,
    aqueous_background: Optional[set] = None,
):
    """Build an N-stage SX cascade with inter-stage linking constraints.

    The cascade arranges stages so that the *raffinate* of stage k feeds
    stage k+1 (cross-current contacting).  Overall feed enters stage 1.

    Returns the cascade Block containing ``stage_1``, ``stage_2``, ...
    and inter-stage constraints.
    """
    cascade = Block(concrete=True)
    parent_block.add_component(name, cascade)

    cascade.n_stages = n_stages
    cascade.stages = {}

    for k in range(1, n_stages + 1):
        stg = build_sx_stage(
            cascade, f"stage_{k}", component_list,
            D_init=D_init, OA_init=OA_init,
            aqueous_background=aqueous_background,
        )
        cascade.stages[k] = stg

    # Feed of cascade = feed of stage 1 (linked externally)
    cascade.feed = cascade.stage_1.feed
    # Final organic = sum of all stage organics
    cascade.organic_total = Expression(
        expr=sum(
            cascade.stages[k].organic_total
            for k in range(1, n_stages + 1)
        ))
    # Final raffinate = raffinate of last stage
    cascade.raffinate = getattr(cascade, f"stage_{n_stages}").raffinate

    # Inter-stage linking: raf_k → feed_{k+1}
    def _make_link(cascade_blk, k):
        src = getattr(cascade_blk, f"stage_{k}").raffinate
        dst = getattr(cascade_blk, f"stage_{k + 1}").feed

        def _link_rule(b, j):
            return src[j] == dst[j]
        setattr(cascade_blk, f"link_{k}_{k + 1}",
                Constraint(component_list, rule=_link_rule,
                           doc=f"Inter-stage link {k}→{k+1}"))

    for k in range(1, n_stages):
        _make_link(cascade, k)

    return cascade
