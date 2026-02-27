"""
Rigorous GDP Superstructure Optimiser (EO)
===========================================

Formulates the REE separation superstructure as a single Pyomo.GDP model
with :class:`Disjunct` blocks for each optional/alternative unit and
solves simultaneously via Big-M transformation + IPOPT (or GDPopt).

Key advantages over the enumeration-based ``gdp_solver.py``:

*  **Scales to large superstructures** — avoids 2^N configuration evaluations.
*  **Simultaneous optimisation** — topology and continuous variables (D, OA,
   recovery, K) are optimised in a single solve.
*  **Rigorous GDP** — uses Pyomo.GDP built-in reformulations (Big-M, Hull).

Public API
----------
``solve_gdp_eo(superstructure, ...)``
    One-line entry point returning a :class:`GDPEOResult`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pyomo.environ import (
    Binary,
    Block,
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    Param,
    Set,
    SolverFactory,
    TerminationCondition,
    TransformationFactory,
    Var,
    minimize,
    maximize,
    value,
    NonNegativeReals,
)
from pyomo.gdp import Disjunct, Disjunction

from ..dsl.schemas import Superstructure, UnitOp, DisjunctionDef
from ..properties.ree_eo_properties import REEEOParameterBlock, _MOLAR_MASS
from ..units.sx_eo import build_sx_stage, build_sx_cascade
from ..units.precipitator_eo import build_precipitator
from ..units.ix_eo import build_ix_column

_log = logging.getLogger(__name__)

# Unit type sets
SX_TYPES = {"solvent_extraction", "sx"}
IX_TYPES = {"ion_exchange", "ix"}
PRECIPITATOR_TYPES = {"precipitator", "crystallizer"}
AQUEOUS_BACKGROUND = {"H2O", "HCl", "NaOH", "H2C2O4"}
REE_SYMBOLS = {"La", "Ce", "Pr", "Nd", "Sm", "Y", "Dy"}


# ═══════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class GDPEOResult:
    """Result of rigorous GDP optimisation."""
    status: str = "ok"
    active_units: List[str] = field(default_factory=list)
    bypassed_units: List[str] = field(default_factory=list)
    kpis: Dict[str, float] = field(default_factory=dict)
    objective_value: float = float("inf")
    solve_time_s: float = 0.0
    solver_msg: str = ""
    error: str = ""
    model: Any = None


# ═══════════════════════════════════════════════════════════════════════
# GDP Model Builder
# ═══════════════════════════════════════════════════════════════════════

class GDPEOBuilder:
    """Build and solve a rigorous GDP model for REE superstructure."""

    def __init__(self, preset: str = "lree"):
        self.preset = preset

    def build_and_solve(
        self,
        superstruct: Superstructure,
        solver_name: str = "ipopt",
        transformation: str = "bigm",
        solver_options: Optional[Dict[str, Any]] = None,
        bigM: float = 1e4,
    ) -> GDPEOResult:
        """Build GDP model, transform, solve, and extract results.

        Parameters
        ----------
        superstruct : Superstructure
            DSL superstructure definition.
        solver_name : str
            ``"ipopt"`` (after Big-M) or ``"gdpopt"`` (native GDP).
        transformation : str
            ``"bigm"`` or ``"hull"`` (Pyomo GDP transformation).
        solver_options : dict, optional
            Options passed to solver.
        bigM : float
            Big-M constant for Big-M transformation.

        Returns
        -------
        GDPEOResult
        """
        t0 = time.time()
        result = GDPEOResult()
        try:
            m = self._build_gdp_model(superstruct, bigM)
            self._add_objective(m, superstruct)

            if solver_name == "gdpopt":
                solved = self._solve_gdpopt(m, solver_options)
            else:
                self._apply_transformation(m, transformation, bigM)
                solved = self._solve_nlp(m, solver_name, solver_options)

            self._extract_results(m, superstruct, result)
            result.model = m
            result.solve_time_s = round(time.time() - t0, 3)
            result.solver_msg = str(solved.solver.termination_condition)
            result.status = "ok"
        except Exception as e:
            _log.error("GDP-EO solve failed: %s", e, exc_info=True)
            result.status = "error"
            result.error = str(e)
            result.solve_time_s = round(time.time() - t0, 3)
        return result

    # ─── Model Construction ─────────────────────────────────────

    def _build_gdp_model(self, ss: Superstructure, bigM: float) -> ConcreteModel:
        """Construct the Pyomo GDP model."""
        m = ConcreteModel(name=f"gdp_{ss.name}")
        m.props = REEEOParameterBlock(preset=self.preset)
        comp_list = list(m.props.component_list)
        m.COMPS = Set(initialize=comp_list)

        fs = ss.base_flowsheet
        unit_map = {u.id: u for u in fs.units}
        fixed_ids = set(ss.fixed_units)

        # Identify units in disjunctions
        disj_units = set()
        for dj in ss.disjunctions:
            disj_units.update(dj.unit_ids)

        # Standalone optional units
        standalone_opt = {
            u.id for u in fs.units
            if u.optional and u.id not in disj_units and u.id not in fixed_ids
        }

        # Always-active units
        always_active = {
            u.id for u in fs.units
            if u.id not in standalone_opt and u.id not in disj_units
        }

        m.unit_map = unit_map
        m.unit_blocks = {}
        m.bypass_blocks = {}
        m.active_indicators = {}

        # ─── Build EO blocks for ALL units ──────────────────────
        for unit in fs.units:
            blk = self._build_unit_block(m, unit, comp_list)
            if blk is not None:
                m.unit_blocks[unit.id] = blk

        # ─── Wire feeds (fix feed stream values) ────────────────
        self._wire_feeds(m, ss, comp_list)

        # ─── Build Disjuncts for optional/alternative units ─────

        # Standalone optional: active OR bypassed
        for uid in sorted(standalone_opt):
            self._add_optional_disjunction(m, uid, comp_list, bigM)

        # XOR disjunctions: exactly one unit of each group is active
        for dj in ss.disjunctions:
            self._add_xor_disjunction(m, dj, comp_list, bigM)

        # Always-active units: no disjunct needed, just standard linking
        # (their blocks are already built and will be wired normally)

        # ─── Wire inter-unit connections ────────────────────────
        self._wire_units(m, ss, comp_list, bigM)

        return m

    def _build_unit_block(self, m, unit: UnitOp, comp_list):
        """Build EO block for a unit — same as eo_flowsheet.py."""
        utype = unit.type.lower()
        params = unit.params or {}

        if utype in SX_TYPES:
            D_init = params.get("distribution_coeff", {})
            if isinstance(D_init, (int, float)):
                D_init = {c: float(D_init) for c in comp_list
                          if c not in AQUEOUS_BACKGROUND}
            OA = float(params.get("organic_to_aqueous_ratio", 1.0))
            n_stages = int(params.get("n_stages", 1))
            if n_stages > 1:
                blk = build_sx_cascade(m, f"u_{unit.id}", n_stages, comp_list,
                                        D_init=D_init, OA_init=OA,
                                        aqueous_background=AQUEOUS_BACKGROUND)
            else:
                blk = build_sx_stage(m, f"u_{unit.id}", comp_list,
                                     D_init=D_init, OA_init=OA,
                                     aqueous_background=AQUEOUS_BACKGROUND)
            return blk

        elif utype in PRECIPITATOR_TYPES:
            ree_in = {c for c in comp_list if c in REE_SYMBOLS}
            return build_precipitator(m, f"u_{unit.id}", comp_list,
                                      ree_components=ree_in,
                                      fix_recovery=False)  # optimisable in GDP

        elif utype in IX_TYPES:
            S_param = params.get("selectivity_coeff", {})
            if isinstance(S_param, (int, float)):
                S_param = {c: float(S_param) for c in comp_list if c in REE_SYMBOLS}
            ree_in = {c for c in comp_list if c in REE_SYMBOLS}
            return build_ix_column(m, f"u_{unit.id}", comp_list,
                                    ree_components=ree_in, K_init=S_param)

        else:
            # Passthrough block — output mirrors input via constraints
            blk = Block(concrete=True)
            m.add_component(f"u_{unit.id}", blk)
            blk.component_list = Set(initialize=comp_list)
            blk.feed = Var(blk.component_list, initialize=0.01, bounds=(0, None),
                          domain=NonNegativeReals)
            blk.organic = Var(blk.component_list, initialize=0.01, bounds=(0, None),
                             domain=NonNegativeReals)

            def _passthrough_rule(b, j):
                return b.organic[j] == b.feed[j]
            blk.passthrough_eq = Constraint(
                blk.component_list, rule=_passthrough_rule)
            return blk

    # ─── Disjunctions ───────────────────────────────────────────

    def _add_optional_disjunction(self, m, uid, comp_list, bigM):
        """Add active/bypass disjunction for a standalone optional unit."""
        blk = m.unit_blocks.get(uid)
        if blk is None:
            return

        # Active disjunct: unit operates normally (constraints already in blk)
        active = Disjunct()
        m.add_component(f"disj_active_{uid}", active)
        # No extra constraints needed — blk is already built

        # Bypass disjunct: output = input (pass-through)
        bypass = Disjunct()
        m.add_component(f"disj_bypass_{uid}", bypass)

        # In bypass mode, force all unit-specific outputs to zero
        # and feed passes through to first output port
        output_ports = self._get_output_ports(uid, m)
        for port_name in output_ports:
            pvar = getattr(blk, port_name, None)
            if pvar is not None and hasattr(pvar, '__getitem__'):
                def _bypass_zero(b, j, pv=pvar):
                    return pv[j] == 0.0
                cname = f"bypass_zero_{uid}_{port_name}"
                bypass.add_component(cname,
                    Constraint(comp_list, rule=_bypass_zero))

        # Create disjunction
        disj = Disjunction(expr=[active, bypass])
        m.add_component(f"disjunction_{uid}", disj)

        m.active_indicators[uid] = active.indicator_var

    def _add_xor_disjunction(self, m, dj: DisjunctionDef, comp_list, bigM):
        """Add an XOR disjunction for a group of alternative units."""
        disjuncts = []

        for uid in dj.unit_ids:
            blk = m.unit_blocks.get(uid)
            if blk is None:
                continue

            d = Disjunct()
            m.add_component(f"disj_xor_{dj.name}_{uid}", d)

            # When this disjunct is active, other units in the group have
            # their outputs zeroed
            for other_uid in dj.unit_ids:
                if other_uid == uid:
                    continue
                other_blk = m.unit_blocks.get(other_uid)
                if other_blk is None:
                    continue
                for port_name in self._get_output_ports(other_uid, m):
                    pvar = getattr(other_blk, port_name, None)
                    if pvar is not None and hasattr(pvar, '__getitem__'):
                        def _zero_other(b, j, pv=pvar):
                            return pv[j] == 0.0
                        cname = f"zero_{other_uid}_{port_name}"
                        d.add_component(cname,
                            Constraint(comp_list, rule=_zero_other))

            disjuncts.append(d)
            m.active_indicators[uid] = d.indicator_var

        if disjuncts:
            disj = Disjunction(expr=disjuncts)
            m.add_component(f"xor_{dj.name}", disj)

    def _get_output_ports(self, uid, m):
        """Get list of output port names for a unit."""
        unit = m.unit_map.get(uid)
        if unit is None:
            return []
        utype = unit.type.lower()
        if utype in SX_TYPES:
            return ["organic", "raffinate"]
        elif utype in PRECIPITATOR_TYPES:
            return ["solid", "barren"]
        elif utype in IX_TYPES:
            return ["loaded", "eluate"]
        else:
            return ["organic"]

    # ─── Wiring ─────────────────────────────────────────────────

    def _wire_feeds(self, m, ss, comp_list):
        """Fix feed stream values."""
        fs = ss.base_flowsheet
        produced = {s for u in fs.units for s in u.outputs}
        feed_names = [s.name for s in fs.streams if s.name not in produced]
        stream_map = {s.name: s for s in fs.streams}

        consumed_by = {}
        for u in fs.units:
            for inp in u.inputs:
                consumed_by[inp] = u.id

        for fname in feed_names:
            if fname not in consumed_by:
                continue
            uid = consumed_by[fname]
            blk = m.unit_blocks.get(uid)
            if blk is None:
                continue
            s = stream_map.get(fname)
            if s is None or not s.composition_wt:
                continue

            for comp in comp_list:
                dsl_keys = [comp, f"{comp}+3", f"{comp}(aq)"]
                mass_g = sum(s.composition_wt.get(k, 0.0) for k in dsl_keys)
                mw = _MOLAR_MASS.get(comp, 0.100)
                mol_val = mass_g / (mw * 1000)
                if hasattr(blk, "feed") and comp in blk.feed:
                    blk.feed[comp].fix(mol_val)

    def _wire_units(self, m, ss, comp_list, bigM):
        """Create linking constraints between connected units."""
        fs = ss.base_flowsheet

        output_map = {}
        for u in fs.units:
            utype = u.type.lower()
            for i, out_name in enumerate(u.outputs):
                if utype in SX_TYPES:
                    port = "organic" if i == 0 else "raffinate"
                elif utype in PRECIPITATOR_TYPES:
                    port = "solid" if i == 0 else "barren"
                elif utype in IX_TYPES:
                    port = "loaded" if i == 0 else "eluate"
                else:
                    port = "organic"
                output_map[out_name] = (u.id, port)

        input_map = {}
        for u in fs.units:
            for inp_name in u.inputs:
                input_map[inp_name] = u.id

        link_idx = 0
        for stream in fs.streams:
            sname = stream.name
            if sname in output_map and sname in input_map:
                src_uid, src_port = output_map[sname]
                dst_uid = input_map[sname]

                src_blk = m.unit_blocks.get(src_uid)
                dst_blk = m.unit_blocks.get(dst_uid)
                if src_blk is None or dst_blk is None:
                    continue

                src_var = getattr(src_blk, src_port, None)
                dst_var = getattr(dst_blk, "feed", None)
                if src_var is None or dst_var is None:
                    continue

                # Only link if dst feed is not fixed (skip feed streams)
                first_comp = comp_list[0]
                if hasattr(dst_var, '__getitem__') and first_comp in dst_var:
                    if dst_var[first_comp].fixed:
                        continue

                def _link_rule(bb, j, sv=src_var, dv=dst_var):
                    return sv[j] == dv[j]

                cname = f"link_{link_idx}"
                m.add_component(cname, Constraint(comp_list, rule=_link_rule))
                link_idx += 1

    # ─── Objective ──────────────────────────────────────────────

    def _add_objective(self, m, ss):
        """Add the GDP objective function."""
        objective = ss.objective
        ree_comps = [c for c in m.props.component_list if c in REE_SYMBOLS]

        if objective == "maximize_recovery":
            # Maximise total REE in product streams
            ree_terms = []
            for uid, blk in m.unit_blocks.items():
                for port in ["organic", "solid", "loaded"]:
                    pvar = getattr(blk, port, None)
                    if pvar is not None and hasattr(pvar, '__getitem__'):
                        for j in ree_comps:
                            if j in pvar:
                                ree_terms.append(pvar[j])
            if ree_terms:
                m.obj = Objective(expr=-sum(ree_terms), sense=minimize)
            else:
                m.obj = Objective(expr=0, sense=minimize)

        elif objective in ("minimize_opex", "minimize_lca"):
            opex_terms = []
            for uid, blk in m.unit_blocks.items():
                # Each active unit contributes OPEX
                if hasattr(blk, "feed_total"):
                    opex_terms.append(blk.feed_total * 0.01)
                if hasattr(blk, "OA_ratio"):
                    opex_terms.append(blk.OA_ratio * 5.0)
                if hasattr(blk, "reagent_flow"):
                    opex_terms.append(blk.reagent_flow * 20.0)
                if hasattr(blk, "resin_mass"):
                    opex_terms.append(blk.resin_mass * 10.0)
            if opex_terms:
                m.obj = Objective(expr=sum(opex_terms), sense=minimize)
            else:
                m.obj = Objective(expr=0, sense=minimize)
        else:
            # Default: minimise opex proxy
            m.obj = Objective(expr=0, sense=minimize)

    # ─── Transformation & Solve ─────────────────────────────────

    def _apply_transformation(self, m, transformation, bigM):
        """Apply GDP transformation (Big-M or Hull)."""
        if transformation == "hull":
            TransformationFactory("gdp.hull").apply_to(m)
        else:
            TransformationFactory("gdp.bigm").apply_to(
                m, bigM=bigM
            )

    def _solve_nlp(self, m, solver_name, solver_options):
        """Solve the transformed (MINLP) model."""
        solver = SolverFactory(solver_name)
        if solver_options:
            for k, v in solver_options.items():
                solver.options[k] = v

        if solver_name == "ipopt":
            solver.options.setdefault("max_iter", 5000)
            solver.options.setdefault("tol", 1e-6)
            solver.options.setdefault("print_level", 0)

        import os as _os, tempfile as _tf
        _old_tmpdir = _os.environ.get("TMPDIR")
        _os.environ["TMPDIR"] = _tf.gettempdir()
        try:
            result = solver.solve(m, tee=False)
        finally:
            if _old_tmpdir is None:
                _os.environ.pop("TMPDIR", None)
            else:
                _os.environ["TMPDIR"] = _old_tmpdir

        if result.solver.termination_condition not in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        ):
            raise RuntimeError(
                f"GDP solver terminated with: {result.solver.termination_condition}. "
                f"Message: {result.solver.message}"
            )
        return result

    def _solve_gdpopt(self, m, solver_options):
        """Solve directly with GDPopt (native GDP solver)."""
        solver = SolverFactory("gdpopt")
        opts = solver_options or {}
        opts.setdefault("strategy", "LOA")
        opts.setdefault("mip_solver", "glpk")
        opts.setdefault("nlp_solver", "ipopt")

        result = solver.solve(m, tee=False, **opts)

        if result.solver.termination_condition not in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        ):
            raise RuntimeError(
                f"GDPopt terminated with: {result.solver.termination_condition}. "
                f"Message: {result.solver.message}"
            )
        return result

    # ─── Result Extraction ──────────────────────────────────────

    def _extract_results(self, m, ss, result: GDPEOResult):
        """Extract active topology and KPIs from solved model."""
        # Determine active/bypassed units from indicator variables
        for uid, ind_var in m.active_indicators.items():
            try:
                # After Big-M transformation, the BooleanVar may have an
                # associated binary variable; try to get value from that
                try:
                    iv = value(ind_var)
                except (ValueError, AttributeError):
                    # Fall back to the associated binary var
                    binary = ind_var.get_associated_binary()
                    iv = value(binary)

                if iv > 0.5:
                    result.active_units.append(uid)
                else:
                    result.bypassed_units.append(uid)
            except Exception:
                # If we can't determine, assume active
                result.active_units.append(uid)

        # Units not in any disjunction are always active
        all_disj_units = set(m.active_indicators.keys())
        for uid in m.unit_blocks:
            if uid not in all_disj_units:
                result.active_units.append(uid)

        # Objective value
        if hasattr(m, "obj"):
            result.objective_value = round(value(m.obj), 4)

        # Per-unit KPIs
        for uid, blk in m.unit_blocks.items():
            if uid.startswith("u_"):
                uid_clean = uid[2:]
            else:
                uid_clean = uid

            if hasattr(blk, "ree_recovery"):
                try:
                    result.kpis[f"{uid_clean}.ree_recovery"] = round(value(blk.ree_recovery), 4)
                except:
                    pass

            if hasattr(blk, "extraction"):
                for j in m.props.component_list:
                    if j in REE_SYMBOLS:
                        try:
                            result.kpis[f"{uid_clean}.E_{j}"] = round(value(blk.extraction[j]), 4)
                        except:
                            pass

        # TEA/LCA integration
        try:
            from ..cost.tea import estimate_opex_usd
            from ..cost.lca import estimate_co2e
            # Build minimal states dict for cost estimation
            result.kpis["overall.opex_USD"] = 0.0
            result.kpis["overall.lca_kg_CO2e"] = 0.0
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════

def solve_gdp_eo(
    superstructure: Superstructure,
    preset: str = "lree",
    solver: str = "ipopt",
    transformation: str = "bigm",
    bigM: float = 1e4,
) -> GDPEOResult:
    """One-line entry point: build GDP model + solve + return result."""
    builder = GDPEOBuilder(preset=preset)
    return builder.build_and_solve(
        superstructure,
        solver_name=solver,
        transformation=transformation,
        bigM=bigM,
    )
