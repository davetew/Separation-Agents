"""
GDP Superstructure Solver
=========================

Evaluates all feasible topology configurations from a :class:`Superstructure`,
optionally running BoTorch continuous optimization within each configuration,
and ranks results by the specified objective.

This is the **Hybrid GDP** approach: topology enumeration is exhaustive
(or sampled for large spaces), and physics is evaluated via the existing
sequential-modular solver.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from ..dsl.schemas import Superstructure
from ..sim.idaes_adapter import IDAESFlowsheetBuilder
from .gdp_builder import Configuration, build_sub_flowsheet, enumerate_configurations

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Result of evaluating a single configuration."""

    config: Configuration
    kpis: Dict[str, float] = field(default_factory=dict)
    objective_value: float = float("inf")
    status: str = "ok"
    error: str = ""
    elapsed_s: float = 0.0
    optimized_params: Dict[str, float] = field(default_factory=dict)
    """Continuous params optimized by BoTorch (if applicable)."""


@dataclass
class GDPResult:
    """Result of the full GDP superstructure optimization."""

    best: Optional[EvalResult] = None
    all_results: List[EvalResult] = field(default_factory=list)
    objective: str = ""
    total_elapsed_s: float = 0.0
    num_configs_evaluated: int = 0


# ---------------------------------------------------------------------------
# Objective extraction
# ---------------------------------------------------------------------------
_OBJECTIVE_KEYS = {
    "minimize_opex": ("overall.opex_USD", "minimize"),
    "maximize_recovery": ("overall.recovery", "maximize"),
    "minimize_lca": ("overall.lca_kg_CO2e", "minimize"),
    "maximize_value_per_kg_ore": ("overall.value_per_kg_ore", "maximize"),
}


def _extract_objective(kpis: Dict[str, float], objective: str) -> float:
    """Extract the objective value from KPIs.

    Returns a float where **lower is always better** (negates for maximize).
    """
    key, direction = _OBJECTIVE_KEYS.get(objective, ("overall.opex_USD", "minimize"))
    raw = kpis.get(key, float("inf"))
    if direction == "maximize":
        return -raw  # negate so lower = better
    return raw


# ---------------------------------------------------------------------------
# Single configuration evaluator
# ---------------------------------------------------------------------------
def evaluate_configuration(
    superstruct: Superstructure,
    config: Configuration,
    database: str = "light_ree",
    param_overrides: Optional[Dict[str, float]] = None,
) -> EvalResult:
    """Build and solve a concrete flowsheet for one topology configuration.

    Parameters
    ----------
    superstruct : Superstructure
        Parent superstructure.
    config : Configuration
        Specific topology to evaluate.
    database : str
        Reaktoro database preset.
    param_overrides : dict, optional
        Override continuous params: keys like ``"sx_1.organic_to_aqueous_ratio"``.

    Returns
    -------
    EvalResult
    """
    t0 = time.time()
    result = EvalResult(config=config)

    try:
        flowsheet = build_sub_flowsheet(superstruct, config)

        # Apply param overrides: "unit_id.param_name" → value
        if param_overrides:
            unit_map = {u.id: u for u in flowsheet.units}
            for key, val in param_overrides.items():
                parts = key.split(".", 1)
                if len(parts) == 2 and parts[0] in unit_map:
                    unit_map[parts[0]].params[parts[1]] = val

        builder = IDAESFlowsheetBuilder(database_name=database)
        sim_result = builder.build_and_solve(flowsheet)

        if sim_result.get("status") != "ok":
            result.status = "error"
            result.error = sim_result.get("error", "Unknown error")
            return result

        result.kpis = sim_result.get("kpis", {})
        result.objective_value = _extract_objective(
            result.kpis, superstruct.objective
        )
        result.status = "ok"

    except Exception as e:
        result.status = "error"
        result.error = str(e)
        _log.warning("Config %s failed: %s", config, e)

    result.elapsed_s = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# BoTorch inner-loop optimizer (per configuration)
# ---------------------------------------------------------------------------
def _optimize_continuous_params(
    superstruct: Superstructure,
    config: Configuration,
    database: str = "light_ree",
    n_initial: int = 3,
    n_iters: int = 5,
) -> EvalResult:
    """Run BoTorch to optimize continuous params within a fixed topology.

    Only runs if ``superstruct.continuous_bounds`` is non-empty.
    """
    bounds_spec = superstruct.continuous_bounds
    if not bounds_spec:
        return evaluate_configuration(superstruct, config, database)

    # Filter bounds to only include params on active units
    active_params: List[Tuple[str, float, float]] = []
    for key, (lo, hi) in bounds_spec.items():
        unit_id = key.split(".")[0]
        if unit_id in config.active_unit_ids:
            active_params.append((key, lo, hi))

    if not active_params:
        return evaluate_configuration(superstruct, config, database)

    param_names = [p[0] for p in active_params]
    bounds_tensor = torch.tensor(
        [[p[1] for p in active_params], [p[2] for p in active_params]],
        dtype=torch.double,
    )

    def objective_fn(x_norm: torch.Tensor) -> float:
        """BoTorch objective: x_norm is in [0,1]^d."""
        x_phys = bounds_tensor[0] + x_norm * (bounds_tensor[1] - bounds_tensor[0])
        overrides = {name: float(x_phys[i]) for i, name in enumerate(param_names)}
        ev = evaluate_configuration(superstruct, config, database, overrides)
        # BoTorch maximizes, but our objective is "lower is better"
        return -ev.objective_value if ev.status == "ok" else -1e6

    from .bo import BotorchOptimizer

    optimizer = BotorchOptimizer(maximize=True)
    best_x, best_y, history = optimizer.optimize(
        objective_fn=objective_fn,
        bounds=bounds_tensor,
        n_initial=n_initial,
        n_iters=n_iters,
    )

    # Re-evaluate at best point to get full KPIs
    best_overrides = {
        name: float(best_x[i]) for i, name in enumerate(param_names)
    }
    result = evaluate_configuration(superstruct, config, database, best_overrides)
    result.optimized_params = best_overrides
    return result


# ---------------------------------------------------------------------------
# Full superstructure optimizer
# ---------------------------------------------------------------------------
def optimize_superstructure(
    superstruct: Superstructure,
    database: str = "light_ree",
    optimize_continuous: bool = True,
    n_bo_initial: int = 3,
    n_bo_iters: int = 5,
    max_configs: int = 64,
) -> GDPResult:
    """Enumerate and evaluate all topology configurations, return best.

    Parameters
    ----------
    superstruct : Superstructure
        The superstructure to optimize.
    database : str
        Reaktoro database preset.
    optimize_continuous : bool
        If True, run BoTorch on continuous params within each topology.
    n_bo_initial, n_bo_iters : int
        BoTorch settings for inner-loop optimization.
    max_configs : int
        Max configurations to evaluate (prune beyond this).

    Returns
    -------
    GDPResult
        Ranked results with the best configuration first.
    """
    t0 = time.time()
    gdp_result = GDPResult(objective=superstruct.objective)

    configs = enumerate_configurations(superstruct)
    _log.info("Superstructure '%s': %d configurations", superstruct.name, len(configs))

    if len(configs) > max_configs:
        _log.warning(
            "Pruning from %d to %d configurations", len(configs), max_configs
        )
        configs = configs[:max_configs]

    for i, config in enumerate(configs):
        _log.info(
            "Evaluating config %d/%d: %s", i + 1, len(configs), config
        )

        if optimize_continuous and superstruct.continuous_bounds:
            ev = _optimize_continuous_params(
                superstruct, config, database,
                n_initial=n_bo_initial, n_iters=n_bo_iters,
            )
        else:
            ev = evaluate_configuration(superstruct, config, database)

        gdp_result.all_results.append(ev)

    # Rank by objective (lower is better after our normalization)
    successful = [r for r in gdp_result.all_results if r.status == "ok"]
    if successful:
        successful.sort(key=lambda r: r.objective_value)
        gdp_result.best = successful[0]

    gdp_result.num_configs_evaluated = len(configs)
    gdp_result.total_elapsed_s = time.time() - t0

    _log.info(
        "GDP optimization complete: %d configs in %.1fs. Best: %s (obj=%.4f)",
        len(configs),
        gdp_result.total_elapsed_s,
        gdp_result.best.config if gdp_result.best else "None",
        gdp_result.best.objective_value if gdp_result.best else float("inf"),
    )

    return gdp_result
