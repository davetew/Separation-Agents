"""
EO Ion Exchange Unit Model
============================

Langmuir-isotherm IX formulated as Pyomo constraints for
equation-oriented optimisation.

The model uses a competitive Langmuir isotherm::

    q_j = q_max · K_j · C_j / (1 + Σ_i K_i · C_i)

where *K_j* is the selectivity constant, *C_j* is the feed
concentration, and *q_max* is the maximum resin capacity.

Design variables (Var, optimisable)
------------------------------------
* ``K[j]``     — selectivity coefficients per component
* ``q_max``    — maximum resin loading [mol/kg_resin]
* ``resin_mass`` — mass of resin [kg]

Constraints
-----------
* Langmuir loading for each REE component
* Mass balance: feed = eluate + loaded
* Background species pass through
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

# Default selectivity coefficients (relative to La)
_DEFAULT_K = {
    "La": 1.0,
    "Ce": 1.5,
    "Pr": 2.0,
    "Nd": 3.0,
    "Sm": 5.0,
    "Y":  8.0,
    "Dy": 10.0,
}


def build_ix_column(
    parent_block,
    name: str,
    component_list,
    ree_components: Optional[set] = None,
    K_init: Optional[Dict[str, float]] = None,
    q_max_init: float = 2.0,
    resin_mass_init: float = 1.0,
):
    """Attach an ion-exchange column block to *parent_block*.

    Parameters
    ----------
    parent_block : Pyomo Block
        Parent to attach to.
    name : str
        Block name.
    component_list : iterable
        All components (REE + background).
    ree_components : set, optional
        Which components participate in IX.
    K_init : dict, optional
        Initial selectivity coefficients.
    q_max_init : float
        Maximum resin capacity [mol/kg_resin].
    resin_mass_init : float
        Resin mass [kg].

    Returns
    -------
    Pyomo Block
    """
    if ree_components is None:
        ree_components = set(_DEFAULT_K.keys())
    if K_init is None:
        K_init = dict(_DEFAULT_K)

    blk = Block(concrete=True)
    parent_block.add_component(name, blk)

    comp_list = list(component_list)
    ree_list = [c for c in comp_list if c in ree_components]
    bg_list = [c for c in comp_list if c not in ree_components]

    blk.component_list = Set(initialize=comp_list)
    blk.ree_set = Set(initialize=ree_list)
    blk.background_set = Set(initialize=bg_list)

    # ── Design variables ────────────────────────────────────────
    blk.K = Var(
        blk.ree_set,
        initialize={c: K_init.get(c, 1.0) for c in ree_list},
        bounds=(0.01, 100),
        domain=PositiveReals,
        doc="Selectivity coefficient",
    )
    blk.q_max = Var(
        initialize=q_max_init,
        bounds=(0.01, 20.0),
        domain=PositiveReals,
        doc="Max resin capacity [mol/kg]",
    )
    blk.resin_mass = Var(
        initialize=resin_mass_init,
        bounds=(0.01, 100.0),
        domain=PositiveReals,
        doc="Resin mass [kg]",
    )

    # ── Stream variables ────────────────────────────────────────
    blk.feed = Var(
        blk.component_list,
        initialize=1.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Inlet component flow [mol/s]",
    )
    blk.loaded = Var(
        blk.component_list,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Loaded resin outlet [mol/s]",
    )
    blk.eluate = Var(
        blk.component_list,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Eluate (barren) outlet [mol/s]",
    )

    # ── Langmuir denominator (Expression) ───────────────────────
    blk.langmuir_denom = Expression(
        expr=1.0 + sum(blk.K[j] * blk.feed[j] for j in blk.ree_set),
        doc="1 + Σ K_j · C_j",
    )

    # ── Recovery fraction from Langmuir (capped at 1.0) ─────────
    # raw_frac = q_max * resin_mass * K_j / denom
    # This is per unit of feed: loaded_j = feed_j * min(raw_frac, 1)
    # We use a smooth min: min(x, 1) ≈ 1 - softplus(1-x, eps)
    # But simpler: just clamp via bounds on recovery Var
    def _raw_frac_rule(b, j):
        return b.q_max * b.resin_mass * b.K[j] / b.langmuir_denom
    blk.raw_recovery_frac = Expression(
        blk.ree_set,
        rule=_raw_frac_rule,
        doc="Uncapped Langmuir recovery fraction",
    )

    # Capped recovery: a Var bounded [0, 1] constrained to equal
    # min(raw_frac, 1). We implement this as:
    #   recovery_j = raw_frac_j    when raw_frac_j ≤ 1
    #   recovery_j = 1.0           when raw_frac_j > 1
    # Smooth implementation: recovery_j ≤ 1 (bound), recovery_j ≤ raw_frac_j
    blk.recovery = Var(
        blk.ree_set,
        initialize={c: min(K_init.get(c, 1.0) * q_max_init * resin_mass_init / 2.0, 0.9)
                    for c in ree_list},
        bounds=(0.0, 1.0),
        domain=NonNegativeReals,
        doc="Capped recovery fraction [0-1]",
    )

    # Constraint: recovery ≤ raw_frac
    def _recovery_cap_rule(b, j):
        return b.recovery[j] == b.raw_recovery_frac[j] - b.recovery_slack[j]
    # Use a slack variable for the capping
    blk.recovery_slack = Var(
        blk.ree_set,
        initialize=0.0,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Slack for recovery cap",
    )

    # Simple approach: just constrain loaded_j = feed_j * min(raw_frac, 1)
    # Since recovery is bounded [0, 1], we set:
    #   recovery = min(raw_frac, 1) via: raw_frac = recovery + slack, slack ≥ 0
    def _recovery_decomp_rule(b, j):
        return b.raw_recovery_frac[j] == b.recovery[j] + b.recovery_slack[j]
    blk.recovery_decomp = Constraint(
        blk.ree_set,
        rule=_recovery_decomp_rule,
        doc="Decompose raw_frac into capped recovery + slack",
    )

    # ── Constraints ─────────────────────────────────────────────

    # REE mass balance
    def _ree_mass_balance_rule(b, j):
        return b.feed[j] == b.loaded[j] + b.eluate[j]
    blk.ree_mass_balance = Constraint(
        blk.ree_set,
        rule=_ree_mass_balance_rule,
    )

    # Loaded = feed × recovery (capped)
    def _ree_loading_rule(b, j):
        return b.loaded[j] == b.feed[j] * b.recovery[j]
    blk.ree_loading = Constraint(
        blk.ree_set,
        rule=_ree_loading_rule,
    )

    # Background species pass through
    def _bg_mass_balance_rule(b, j):
        return b.eluate[j] == b.feed[j]
    blk.bg_mass_balance = Constraint(
        blk.background_set,
        rule=_bg_mass_balance_rule,
    )

    def _bg_loaded_zero_rule(b, j):
        return b.loaded[j] == 0.0
    blk.bg_loaded_zero = Constraint(
        blk.background_set,
        rule=_bg_loaded_zero_rule,
    )

    # ── Convenience expressions ─────────────────────────────────
    blk.feed_total = Expression(
        expr=sum(blk.feed[j] for j in blk.component_list))
    blk.loaded_total = Expression(
        expr=sum(blk.loaded[j] for j in blk.component_list))
    blk.eluate_total = Expression(
        expr=sum(blk.eluate[j] for j in blk.component_list))

    blk.ree_recovery = Expression(
        expr=sum(blk.loaded[j] for j in blk.ree_set)
             / (sum(blk.feed[j] for j in blk.ree_set) + 1e-12),
        doc="Overall REE recovery to loaded resin",
    )

    return blk
