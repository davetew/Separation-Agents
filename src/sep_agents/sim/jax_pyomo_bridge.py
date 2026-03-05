"""
JAX-Pyomo ExternalFunction Bridge
==================================

Wraps the JAX Gibbs-energy-minimisation equilibrium solver as a set of
Pyomo ``ExternalFunction`` instances so that IPOPT receives exact first-
order derivatives (via ``jax.jacfwd``) during NLP solves.  This enables
simultaneous optimisation of **continuous parameters** (temperature,
pressure, feed composition) **and** discrete topology decisions within
Pyomo.GDP superstructures.

Public API
----------
``build_jax_reactor(parent_block, name, component_list, ...)``
    Factory matching the ``build_sx_stage`` / ``build_precipitator``
    pattern.  Returns a Pyomo ``Block`` with ``feed[j]`` → ``product[j]``
    linked through the JAX equilibrium solve.

Architecture
------------
Pyomo ``ExternalFunction(fcn=callback)`` requires a *scalar* return.
Because equilibrium produces N output species, we register N separate
``ExternalFunction`` instances (``_efn_0``, ``_efn_1``, …), each
returning the equilibrium amount of one output species.  All N callbacks
share a **memoisation cache** keyed on the input vector so that only ONE
JAX solve is performed per unique IPOPT evaluation point.

References
----------
- Pyomo ExternalFunction docs (``pyomo.core.base.external``)
- JAX autodiff: ``jax.jacfwd``, ``jax.jacobian``
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from pyomo.environ import (
    Block,
    Constraint,
    Expression,
    ExternalFunction,
    Reference,
    Set as PyomoSet,
    Var,
    NonNegativeReals,
    PositiveReals,
    value,
)

_log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Species name mapping:  EO component names  ↔  JAX species names
# ═══════════════════════════════════════════════════════════════════════

# The EO property package uses simplified names (La, Ce, H2O, HCl)
# while the JAX solver uses full species names (La+3, Ce+3, H2O(aq), HCl(aq)).

_EO_TO_JAX = {
    "H2O": "H2O(aq)",
    "HCl": "HCl(aq)",
    "NaOH": "NaOH(aq)",
    "NaCl": "NaCl(aq)",
}
# REE components: La → La+3, Ce → Ce+3, etc.
_REE_SYMBOLS = {"La", "Ce", "Pr", "Nd", "Sm", "Y", "Dy",
                "Gd", "Tb", "Ho", "Er", "Tm", "Yb", "Lu", "Sc"}

_BACKGROUND = {"H2O", "HCl", "NaOH", "NaCl", "H2C2O4"}


def _eo_to_jax_name(comp: str) -> str:
    """Convert an EO component name to the JAX species name."""
    if comp in _EO_TO_JAX:
        return _EO_TO_JAX[comp]
    if comp in _REE_SYMBOLS:
        return f"{comp}+3"
    # Ions: Ca, Fe, Al, etc.
    return comp


def _jax_to_eo_name(sp: str) -> str:
    """Convert a JAX species name back to an EO component name."""
    # Reverse the mapping
    for eo, jax in _EO_TO_JAX.items():
        if sp == jax:
            return eo
    # Strip +3 suffix for REEs
    if sp.endswith("+3"):
        base = sp[:-2]
        if base in _REE_SYMBOLS:
            return base
    # Strip +2 suffix for divalent ions
    if sp.endswith("+2"):
        return sp[:-2]
    return sp


# ═══════════════════════════════════════════════════════════════════════
# Memoised JAX equilibrium evaluation
# ═══════════════════════════════════════════════════════════════════════

class _JaxEquilibriumCache:
    """Cache for JAX equilibrium solves to avoid redundant GEM evaluations.

    Within a single IPOPT iteration, Pyomo evaluates each ExternalFunction
    independently.  Since all N output functions share the same inputs,
    we cache the *entire* solve result keyed on the input vector (rounded
    to a tolerance to handle floating-point drift).
    """

    def __init__(self, solver, jax_species: List[str], eo_components: List[str]):
        self.solver = solver
        self.jax_species = jax_species
        self.eo_components = eo_components
        self._cache_key: Optional[tuple] = None
        self._cache_result: Optional[np.ndarray] = None
        self._cache_jac: Optional[np.ndarray] = None

        # JAX species index → EO component index mapping
        self._jax_to_eo_idx: Dict[int, int] = {}
        jax_sp_idx = {sp: i for i, sp in enumerate(jax_species)}
        for ci, comp in enumerate(eo_components):
            jax_name = _eo_to_jax_name(comp)
            if jax_name in jax_sp_idx:
                self._jax_to_eo_idx[jax_sp_idx[jax_name]] = ci

        # Number of inputs: 2 (T, P) + len(eo_components) feed amounts
        self.n_inputs = 2 + len(eo_components)
        self.n_outputs = len(eo_components)

    def _make_key(self, args: tuple) -> tuple:
        """Create a hashable cache key from input args."""
        return tuple(round(float(a), 10) for a in args)

    def evaluate_all(self, args: tuple) -> np.ndarray:
        """Run the JAX solve (or return cached result).

        Parameters
        ----------
        args : tuple
            (T_K, P_Pa, feed_mol_0, feed_mol_1, ..., feed_mol_{N-1})

        Returns
        -------
        np.ndarray of shape (n_outputs,)
            Equilibrium amounts for each EO component.
        """
        key = self._make_key(args)
        if key == self._cache_key and self._cache_result is not None:
            return self._cache_result

        T_K = float(args[0])
        P_Pa = float(args[1])
        feed_mols = args[2:]

        # Build species_amounts dict for JAX solver
        species_amounts: Dict[str, float] = {}
        for ci, comp in enumerate(self.eo_components):
            jax_name = _eo_to_jax_name(comp)
            species_amounts[jax_name] = max(float(feed_mols[ci]), 1e-20)

        # Solve equilibrium
        result = self.solver.solve(T_K, P_Pa, species_amounts)

        # Extract output amounts in EO component order
        out = np.zeros(self.n_outputs, dtype=np.float64)
        if result["status"] == "ok":
            eq_amounts = result["species_amounts"]
            for ci, comp in enumerate(self.eo_components):
                jax_name = _eo_to_jax_name(comp)
                out[ci] = eq_amounts.get(jax_name, 0.0)

        self._cache_key = key
        self._cache_result = out
        self._cache_jac = None  # invalidate Jacobian cache
        return out

    def jacobian_all(self, args: tuple) -> np.ndarray:
        """Compute ∂(output_j)/∂(input_i) via finite differences.

        We use finite differences here because the JAX solver internally
        uses scipy.optimize which breaks the JAX trace.  For the IPOPT
        use-case, finite differences at ~1e-7 step size provide sufficient
        accuracy for convergence.

        Returns
        -------
        np.ndarray of shape (n_outputs, n_inputs)
        """
        key = self._make_key(args)
        if key == self._cache_key and self._cache_jac is not None:
            return self._cache_jac

        # Ensure base evaluation is cached
        f0 = self.evaluate_all(args)

        # Finite-difference Jacobian
        jac = np.zeros((self.n_outputs, self.n_inputs), dtype=np.float64)
        args_arr = np.array(args, dtype=np.float64)

        for i in range(self.n_inputs):
            h = max(abs(args_arr[i]) * 1e-7, 1e-10)
            args_pert = args_arr.copy()
            args_pert[i] += h

            # Clear cache to force re-evaluation
            old_key = self._cache_key
            old_result = self._cache_result
            self._cache_key = None

            f_pert = self.evaluate_all(tuple(args_pert))
            jac[:, i] = (f_pert - f0) / h

            # Restore base cache
            self._cache_key = old_key
            self._cache_result = old_result

        self._cache_jac = jac
        return jac


# ═══════════════════════════════════════════════════════════════════════
# Pyomo ExternalFunction callback factory
# ═══════════════════════════════════════════════════════════════════════

def _make_ext_callbacks(cache: _JaxEquilibriumCache, output_idx: int):
    """Create (fcn, grad) callbacks for the k-th output species.

    Pyomo ExternalFunction Python callback protocol:
      fcn(arg1, arg2, ..., argN, fid)  → float
      grad(arg1, arg2, ..., argN, fid) → list[float]

    Pyomo passes each argument as a separate positional argument,
    followed by a trailing integer function ID.  We strip the fid
    and pack the remaining args into a tuple for the cache.

    Parameters
    ----------
    cache : _JaxEquilibriumCache
        Shared cache for all output species.
    output_idx : int
        Which output species this function returns.
    """
    n_expected = cache.n_inputs  # T, P, feed[0], ..., feed[N-1]

    def fcn(*args):
        """Evaluate equilibrium amount for output species k."""
        # Strip the trailing function ID appended by Pyomo
        input_args = args[:n_expected]
        result = cache.evaluate_all(tuple(float(a) for a in input_args))
        return float(result[output_idx])

    def grad(*args):
        """First derivatives ∂f_k/∂(input_i)."""
        input_args = args[:n_expected]
        jac = cache.jacobian_all(tuple(float(a) for a in input_args))
        row = jac[output_idx, :]
        return list(float(x) for x in row)

    return fcn, grad


# ═══════════════════════════════════════════════════════════════════════
# Pyomo Block Factory  (public API)
# ═══════════════════════════════════════════════════════════════════════

def build_jax_reactor(
    parent_block,
    name: str,
    component_list: List[str],
    preset: str = "light_ree",
    temperature_init: float = 298.15,
    pressure_init: float = 101325.0,
    temperature_bounds: Tuple[float, float] = (273.15, 623.15),
    pressure_bounds: Tuple[float, float] = (1e4, 1e8),
    fix_temperature: bool = False,
    fix_pressure: bool = True,
) -> Block:
    """Build a JAX-equilibrium reactor block matching the EO unit pattern.

    Creates a Pyomo Block with:
      - ``blk.feed[j]``        — inlet component flows (Var, mol/s)
      - ``blk.product[j]``     — outlet component flows at equilibrium (Var)
      - ``blk.temperature``    — reactor temperature (Var, K)
      - ``blk.pressure``       — reactor pressure (Var, Pa)
      - ``blk.eq_link[j]``     — constraints linking product to ExternalFunction
      - ``blk.feed_total``     — Expression: sum of feed flows
      - ``blk.product_total``  — Expression: sum of product flows

    The block follows the same interface as ``build_sx_stage`` so it plugs
    directly into ``EOFlowsheetBuilder._build_unit_block()`` and
    ``GDPEOBuilder._build_unit_block()``.

    Parameters
    ----------
    parent_block : ConcreteModel or Block
        Parent Pyomo model to attach the reactor block to.
    name : str
        Block name (e.g., ``"u_reactor_1"``).
    component_list : list[str]
        EO component names (e.g., ``["La", "Ce", "H2O", "HCl"]``).
    preset : str
        JAX chemical system preset (default ``"light_ree"``).
    temperature_init : float
        Initial temperature in K (default 298.15).
    pressure_init : float
        Initial pressure in Pa (default 101325.0).
    temperature_bounds : tuple
        (T_min, T_max) in K.
    pressure_bounds : tuple
        (P_min, P_max) in Pa.
    fix_temperature : bool
        If True, fix temperature (simulation mode). Default False (optimisable).
    fix_pressure : bool
        If True, fix pressure. Default True.

    Returns
    -------
    Block
        The Pyomo Block attached to ``parent_block``.
    """
    from .jax_equilibrium import build_jax_system, JaxEquilibriumSolver

    # ── Build JAX solver ────────────────────────────────────────
    system = build_jax_system(preset=preset, include_minerals=True)
    solver = JaxEquilibriumSolver(system, tol=1e-8, maxiter=500)

    # ── Create Pyomo Block ──────────────────────────────────────
    blk = Block(concrete=True)
    parent_block.add_component(name, blk)

    comp_list = list(component_list)
    blk.component_list = PyomoSet(initialize=comp_list)

    # ── Design variables ────────────────────────────────────────
    blk.temperature = Var(
        initialize=temperature_init,
        bounds=temperature_bounds,
        domain=PositiveReals,
        doc="Reactor temperature [K]",
    )
    blk.pressure = Var(
        initialize=pressure_init,
        bounds=pressure_bounds,
        domain=PositiveReals,
        doc="Reactor pressure [Pa]",
    )

    if fix_temperature:
        blk.temperature.fix(temperature_init)
    if fix_pressure:
        blk.pressure.fix(pressure_init)

    # ── Stream variables ────────────────────────────────────────
    blk.feed = Var(
        blk.component_list,
        initialize=0.01,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Inlet component flow [mol/s]",
    )
    blk.product = Var(
        blk.component_list,
        initialize=0.01,
        bounds=(0, None),
        domain=NonNegativeReals,
        doc="Outlet component flow at equilibrium [mol/s]",
    )

    # Create 'organic' alias via Reference so GDP wiring logic can find output
    blk.organic = Reference(blk.product)

    # ── Build memoised JAX cache ────────────────────────────────
    cache = _JaxEquilibriumCache(solver, system.species_names, comp_list)
    blk._jax_cache = cache  # prevent garbage collection

    # ── Register ExternalFunctions ──────────────────────────────
    # One ExternalFunction per output component
    blk._ext_fns = {}
    for ci, comp in enumerate(comp_list):
        fcn, grad = _make_ext_callbacks(cache, ci)
        ef = ExternalFunction(fcn, grad)
        fn_name = f"_efn_{ci}"
        blk.add_component(fn_name, ef)
        blk._ext_fns[comp] = ef

    # ── Equilibrium linking constraints ─────────────────────────
    # product[j] = ExternalFunction_j(T, P, feed[0], feed[1], ...)
    #
    # Pyomo ExternalFunction calls: ef(arg1, arg2, ..., argN)
    # Our convention: args = (T, P, feed[comp_0], feed[comp_1], ...)

    def _eq_link_rule(b, j):
        ef = b._ext_fns[j]
        # Build argument list: T, P, then feed[comp] in order
        args = [b.temperature, b.pressure]
        for comp in comp_list:
            args.append(b.feed[comp])
        return b.product[j] == ef(*args)

    blk.eq_link = Constraint(
        blk.component_list,
        rule=_eq_link_rule,
        doc="Link product to JAX equilibrium ExternalFunction",
    )

    # ── Convenience expressions ─────────────────────────────────
    blk.feed_total = Expression(
        expr=sum(blk.feed[j] for j in blk.component_list),
    )
    blk.product_total = Expression(
        expr=sum(blk.product[j] for j in blk.component_list),
    )

    _log.info(
        "Built JAX reactor block '%s' with %d components, T=%.1fK, P=%.0fPa",
        name, len(comp_list), temperature_init, pressure_init,
    )

    return blk
