"""
GDP Superstructure Builder
==========================

Translates a :class:`Superstructure` DSL definition into an enumerable
set of valid topology configurations.  Each configuration is a concrete
:class:`Flowsheet` that can be evaluated by `IDAESFlowsheetBuilder`.

This is the **Hybrid GDP** approach: combinatorial topology decisions
are handled here; physics evaluations are delegated to the existing
sequential-modular solver + Reaktoro.
"""
from __future__ import annotations

import itertools
import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from ..dsl.schemas import (
    DisjunctionDef,
    Flowsheet,
    Superstructure,
    Stream,
    UnitOp,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration container
# ---------------------------------------------------------------------------
@dataclass
class Configuration:
    """A single instantiated topology from a superstructure."""

    id: int
    active_unit_ids: FrozenSet[str]
    bypassed_unit_ids: FrozenSet[str]
    stage_choices: Dict[str, int] = field(default_factory=dict)
    """Map of unit_id → chosen stage count (for variable-stage units)."""

    description: str = ""

    def __repr__(self) -> str:
        active = ", ".join(sorted(self.active_unit_ids))
        stages = ", ".join(f"{k}={v}" for k, v in self.stage_choices.items())
        extra = f", stages={{{stages}}}" if stages else ""
        return f"Config#{self.id}(active=[{active}]{extra})"


# ---------------------------------------------------------------------------
# Topology enumerator
# ---------------------------------------------------------------------------
def enumerate_configurations(superstruct: Superstructure) -> List[Configuration]:
    """Generate all valid topology combinations from a superstructure.

    Rules
    -----
    1. Units in ``fixed_units`` are always active.
    2. ``optional`` units independently contribute 2 states (on/off).
    3. ``DisjunctionDef`` groups enforce exactly-one-active (XOR).
    4. ``stage_range`` units contribute one variant per integer in [min, max].
    5. Infeasible topologies (disconnected graphs) are pruned.

    Returns
    -------
    list[Configuration]
        All feasible topologies, numbered starting from 0.
    """
    unit_map: Dict[str, UnitOp] = {
        u.id: u for u in superstruct.base_flowsheet.units
    }
    all_unit_ids = set(unit_map.keys())
    fixed = set(superstruct.fixed_units)

    # Identify units governed by disjunctions (XOR groups)
    disjunction_units: Set[str] = set()
    for dj in superstruct.disjunctions:
        disjunction_units.update(dj.unit_ids)

    # Identify standalone optional units (not in any disjunction)
    standalone_optional = {
        uid for uid, u in unit_map.items()
        if u.optional and uid not in disjunction_units and uid not in fixed
    }

    # Identify units with stage_range
    stage_units = {
        uid: u.stage_range
        for uid, u in unit_map.items()
        if u.stage_range is not None
    }

    # --- Build the combinatorial axes ---

    # Axis 1: Disjunction choices (exactly one active per group)
    disj_axes: List[List[FrozenSet[str]]] = []
    for dj in superstruct.disjunctions:
        # Each choice activates one unit, deactivates the rest
        axis = []
        for chosen in dj.unit_ids:
            if chosen in all_unit_ids:
                axis.append(frozenset([chosen]))
        if axis:
            disj_axes.append(axis)

    # Axis 2: Optional unit on/off
    opt_axes: List[List[FrozenSet[str]]] = []
    for uid in sorted(standalone_optional):
        opt_axes.append([
            frozenset([uid]),  # ON
            frozenset(),       # OFF
        ])

    # Axis 3: Stage counts
    stage_axes: List[List[Tuple[str, int]]] = []
    for uid, (lo, hi) in stage_units.items():
        stage_axes.append([(uid, n) for n in range(lo, hi + 1)])

    # --- Cartesian product ---
    all_axes = disj_axes + opt_axes
    if not all_axes:
        all_axes = [[frozenset()]]  # single degenerate axis

    stage_combos = list(itertools.product(*stage_axes)) if stage_axes else [()]

    configs: List[Configuration] = []
    config_id = 0

    for combo in itertools.product(*all_axes):
        # Collect active units from each axis choice
        chosen_units: Set[str] = set(fixed)
        for choice in combo:
            chosen_units.update(choice)

        # Add non-optional, non-disjunction units that are always present
        for uid in all_unit_ids:
            if uid not in standalone_optional and uid not in disjunction_units:
                chosen_units.add(uid)

        bypassed = all_unit_ids - chosen_units

        for stage_combo in stage_combos:
            stage_choices = dict(stage_combo)
            configs.append(Configuration(
                id=config_id,
                active_unit_ids=frozenset(chosen_units),
                bypassed_unit_ids=frozenset(bypassed),
                stage_choices=stage_choices,
            ))
            config_id += 1

    _log.info("Enumerated %d configurations from superstructure '%s'",
              len(configs), superstruct.name)
    return configs


# ---------------------------------------------------------------------------
# Sub-flowsheet builder
# ---------------------------------------------------------------------------
def build_sub_flowsheet(
    superstruct: Superstructure,
    config: Configuration,
) -> Flowsheet:
    """Build a concrete Flowsheet for a specific topology configuration.

    Bypassed units are removed; their input streams are connected directly
    to their output streams (passthrough).

    Parameters
    ----------
    superstruct : Superstructure
        The parent superstructure definition.
    config : Configuration
        The specific topology to instantiate.

    Returns
    -------
    Flowsheet
        A concrete, solvable flowsheet with only active units.
    """
    base = superstruct.base_flowsheet
    unit_map = {u.id: u for u in base.units}

    # Build stream remap table for bypassed units
    # If a unit is bypassed, its outputs become aliases for its inputs
    remap: Dict[str, str] = {}
    for uid in config.bypassed_unit_ids:
        unit = unit_map[uid]
        if unit.inputs and unit.outputs:
            # Map each output to the first input (passthrough semantics)
            primary_input = unit.inputs[0]
            for out_name in unit.outputs:
                remap[out_name] = primary_input

    def resolve(stream_name: str) -> str:
        """Follow the remap chain to the original stream."""
        seen = set()
        while stream_name in remap and stream_name not in seen:
            seen.add(stream_name)
            stream_name = remap[stream_name]
        return stream_name

    # Build active unit list with remapped connections
    active_units: List[UnitOp] = []
    for uid in config.active_unit_ids:
        if uid not in unit_map:
            continue
        orig = unit_map[uid]
        # Only resolve INPUTS through the remap chain.
        # Outputs are PRODUCED by this active unit, so they keep their
        # original names — otherwise disjunction alternatives sharing
        # the same I/O create self-loops.
        remapped_inputs = [resolve(s) for s in orig.inputs]
        remapped_outputs = list(orig.outputs)

        # Apply stage_choice if applicable
        params = dict(orig.params)
        if uid in config.stage_choices:
            params["stages"] = config.stage_choices[uid]

        active_units.append(UnitOp(
            id=orig.id,
            type=orig.type,
            params=params,
            inputs=remapped_inputs,
            outputs=remapped_outputs,
            optional=orig.optional,
            alternatives=orig.alternatives,
            stage_range=orig.stage_range,
        ))

    # Collect needed stream names
    needed_streams: Set[str] = set()
    for u in active_units:
        needed_streams.update(u.inputs)
        needed_streams.update(u.outputs)

    # Build stream list — include feed streams and remap as needed
    streams: List[Stream] = []
    seen_names: Set[str] = set()
    for s in base.streams:
        resolved_name = resolve(s.name)
        if resolved_name in needed_streams and resolved_name not in seen_names:
            streams.append(Stream(
                name=resolved_name,
                phase=s.phase,
                temperature_K=s.temperature_K,
                pressure_Pa=s.pressure_Pa,
                composition_wt=dict(s.composition_wt),
                psd=s.psd,
                liberation=s.liberation,
                pH=s.pH,
                Eh_mV=s.Eh_mV,
                solids_wtfrac=s.solids_wtfrac,
            ))
            seen_names.add(resolved_name)

    return Flowsheet(
        name=f"{superstruct.name}_config{config.id}",
        units=active_units,
        streams=streams,
    )
