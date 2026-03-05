"""
EO Flowsheet Builder
====================

Constructs and solves equation-oriented IDAES flowsheets from the
DSL ``Flowsheet`` schema.  Unlike the sequential-modular adapter,
this builder formulates the entire process as a simultaneous system
of Pyomo constraints and solves with IPOPT.

Public API
----------
``build_and_solve_eo(flowsheet, objective, ...)``
    One-line entry point returning the same result format as ``run_idaes``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Objective,
    Set,
    SolverFactory,
    SolverStatus,
    TerminationCondition,
    Var,
    minimize,
    maximize,
    value,
)

from ..dsl.schemas import Flowsheet, UnitOp, Stream
from ..properties.ree_eo_properties import REEEOParameterBlock
from ..units.sx_eo import build_sx_stage, build_sx_cascade
from ..units.precipitator_eo import build_precipitator
from ..units.ix_eo import build_ix_column
from ..sim.jax_pyomo_bridge import build_jax_reactor

_log = logging.getLogger(__name__)

# Unit type sets (matching idaes_adapter.py conventions)
SX_TYPES = {"solvent_extraction", "sx"}
IX_TYPES = {"ion_exchange", "ix"}
PRECIPITATOR_TYPES = {"precipitator", "crystallizer"}
PASSTHROUGH_TYPES = {"mixer", "mill", "separator", "magnetic_separator",
                     "flotation", "leach", "filter"}

# Reactor types backed by JAX equilibrium solver
REACTOR_TYPES = {"reactor", "jax_reactor", "leach_reactor",
                 "serpentinization_reactor", "carbonation_reactor"}
# Equipment types — pass-through until full Pyomo models exist
GEO_EQUIPMENT_TYPES = {"heat_exchanger", "pump"}

# Background species that stay in aqueous phase during SX
AQUEOUS_BACKGROUND = {"H2O", "HCl", "NaOH", "H2C2O4"}

# REE element symbols
REE_SYMBOLS = {"La", "Ce", "Pr", "Nd", "Sm", "Y", "Dy"}


class EOFlowsheetBuilder:
    """Build and solve equation-oriented REE separation flowsheets.

    Parameters
    ----------
    preset : str
        Property-package component preset (e.g. ``"lree"``).
    """

    def __init__(self, preset: str = "lree"):
        self.preset = preset

    def build_and_solve(
        self,
        flowsheet: Flowsheet,
        objective: str = "minimize_opex",
        solver_name: str = "ipopt",
        solver_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build the EO model, solve, and return results.

        Parameters
        ----------
        flowsheet : Flowsheet
            DSL flowsheet definition.
        objective : str
            ``"minimize_opex"`` | ``"maximize_recovery"`` | ``"none"``.
        solver_name : str
            Pyomo solver name (default ``"ipopt"``).
        solver_options : dict, optional
            Options passed to solver.

        Returns
        -------
        dict
            ``status``, ``states``, ``kpis``, ``solve_time_s``, ``solver_msg``.
        """
        t0 = time.time()
        try:
            m = self._build_model(flowsheet)
            self._add_objective(m, flowsheet, objective)
            result = self._solve(m, solver_name, solver_options)
            states = self._extract_states(m, flowsheet)
            kpis = self._compute_kpis(m, flowsheet, states)
            elapsed = time.time() - t0
            return {
                "status": "ok",
                "states": states,
                "kpis": kpis,
                "solve_time_s": round(elapsed, 3),
                "solver_msg": str(result.solver.termination_condition),
                "model": m,
            }
        except Exception as e:
            _log.error("EO solve failed: %s", e)
            return {
                "status": "error",
                "error": str(e),
                "solve_time_s": round(time.time() - t0, 3),
            }

    # ─── Internal: model construction ───────────────────────────

    def _build_model(self, flowsheet: Flowsheet) -> ConcreteModel:
        """Construct the Pyomo ConcreteModel."""
        m = ConcreteModel(name=f"eo_{flowsheet.name}")
        m.props = REEEOParameterBlock(preset=self.preset)
        comp_list = list(m.props.component_list)

        # Identify feed streams (not produced by any unit)
        produced = {s for u in flowsheet.units for s in u.outputs}
        feed_stream_names = [s.name for s in flowsheet.streams
                             if s.name not in produced]

        # Build unit blocks
        m.unit_blocks = {}
        for unit in flowsheet.units:
            blk = self._build_unit_block(m, unit, comp_list)
            if blk is not None:
                m.unit_blocks[unit.id] = blk

        # Wire feed streams → first unit inputs
        self._wire_feeds(m, flowsheet, feed_stream_names, comp_list)

        # Wire inter-unit connections
        self._wire_units(m, flowsheet, comp_list)

        return m

    def _build_unit_block(
        self, m: ConcreteModel, unit: UnitOp, comp_list: List[str]
    ):
        """Create the appropriate EO block for a unit operation."""
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
                blk = build_sx_cascade(
                    m, f"u_{unit.id}", n_stages, comp_list,
                    D_init=D_init, OA_init=OA,
                    aqueous_background=AQUEOUS_BACKGROUND,
                )
            else:
                blk = build_sx_stage(
                    m, f"u_{unit.id}", comp_list,
                    D_init=D_init, OA_init=OA,
                    aqueous_background=AQUEOUS_BACKGROUND,
                )

            # Fix distribution coefficients and O/A ratio as simulation inputs
            # (unfix explicitly for optimisation problems)
            if hasattr(blk, 'D'):
                for j in blk.D:
                    blk.D[j].fix()
            if hasattr(blk, 'OA_ratio'):
                blk.OA_ratio.fix()

            return blk

        elif utype in PRECIPITATOR_TYPES:
            recovery = params.get("recovery", None)
            dosage = float(params.get("reagent_dosage_gpl", 10.0)) / 1000.0
            ree_in_list = {c for c in comp_list if c in REE_SYMBOLS}
            return build_precipitator(
                m, f"u_{unit.id}", comp_list,
                ree_components=ree_in_list,
                recovery_init=recovery,
                reagent_flow_init=max(dosage, 0.001),
                fix_recovery=True,  # simulation mode by default
            )

        elif utype in IX_TYPES:
            S_param = params.get("selectivity_coeff", {})
            if isinstance(S_param, (int, float)):
                S_param = {c: float(S_param) for c in comp_list
                           if c in REE_SYMBOLS}
            ree_in_list = {c for c in comp_list if c in REE_SYMBOLS}
            blk = build_ix_column(
                m, f"u_{unit.id}", comp_list,
                ree_components=ree_in_list,
                K_init=S_param,
            )

            # Fix design parameters for simulation mode
            if hasattr(blk, 'K'):
                for j in blk.K:
                    blk.K[j].fix()
            if hasattr(blk, 'q_max'):
                blk.q_max.fix()
            if hasattr(blk, 'resin_mass'):
                blk.resin_mass.fix()

            return blk

        elif utype in REACTOR_TYPES:
            # JAX-backed equilibrium reactor with differentiable gradients
            params_r = unit.params or {}
            T_init = float(params_r.get("temperature_K", 298.15))
            P_init = float(params_r.get("pressure_Pa", 101325.0))
            preset = params_r.get("preset", "light_ree")
            fix_T = params_r.get("fix_temperature", True)  # simulation mode default
            fix_P = params_r.get("fix_pressure", True)
            return build_jax_reactor(
                m, f"u_{unit.id}", comp_list,
                preset=preset,
                temperature_init=T_init,
                pressure_init=P_init,
                fix_temperature=fix_T,
                fix_pressure=fix_P,
            )

        elif utype in PASSTHROUGH_TYPES | GEO_EQUIPMENT_TYPES:
            # Simple pass-through: create trivial block with feed = output
            return self._build_passthrough(m, unit, comp_list)

        else:
            _log.warning("Unknown unit type '%s', treating as passthrough", utype)
            return self._build_passthrough(m, unit, comp_list)

    def _build_passthrough(self, m, unit, comp_list):
        """Build a trivial pass-through block."""
        from pyomo.environ import Block
        blk = Block(concrete=True)
        m.add_component(f"u_{unit.id}", blk)
        blk.component_list = Set(initialize=comp_list)
        blk.feed = Var(blk.component_list, initialize=1.0, bounds=(0, None))

        # For pass-through, output = input
        # Create named outputs matching unit.outputs
        if len(unit.outputs) >= 1:
            blk.organic = blk.feed  # alias
        if len(unit.outputs) >= 2:
            # Second output gets zero (or could split — simplified)
            blk.raffinate = Var(blk.component_list, initialize=0.0, bounds=(0, None))

            def _passthrough_rule(b, j):
                return b.raffinate[j] == 0.0
            blk.passthrough_zero = Constraint(blk.component_list, rule=_passthrough_rule)

        return blk

    # ─── Internal: wiring ───────────────────────────────────────

    def _wire_feeds(self, m, flowsheet, feed_names, comp_list):
        """Fix feed stream values onto first units' feed variables."""
        stream_map = {s.name: s for s in flowsheet.streams}
        consumed_by = {}  # stream_name → unit_id
        for u in flowsheet.units:
            for inp in u.inputs:
                consumed_by[inp] = u.id

        m.feed_constraints = {}
        for fname in feed_names:
            if fname not in consumed_by:
                continue
            uid = consumed_by[fname]
            if uid not in m.unit_blocks:
                continue

            blk = m.unit_blocks[uid]
            s = stream_map.get(fname)
            if s is None or not s.composition_wt:
                continue

            # Convert composition_wt to mol amounts
            from ..properties.ree_eo_properties import _MOLAR_MASS
            for comp in comp_list:
                # Map DSL species names to component names
                dsl_keys = [comp, f"{comp}+3", f"{comp}(aq)"]
                mass_g = 0.0
                for k in dsl_keys:
                    mass_g += s.composition_wt.get(k, 0.0)
                mw = _MOLAR_MASS.get(comp, 0.100)
                mol_val = mass_g / (mw * 1000)  # g → kg → mol

                # Fix the feed variable
                if hasattr(blk, "feed") and comp in blk.feed:
                    blk.feed[comp].fix(mol_val)

    def _wire_units(self, m, flowsheet, comp_list):
        """Create inter-unit linking constraints."""
        # Build output → unit map
        output_map = {}  # stream_name → (unit_id, port_type)
        for u in flowsheet.units:
            for i, out_name in enumerate(u.outputs):
                port = "organic" if i == 0 else "raffinate"
                # For precipitator: first=solid, second=barren
                if u.type in PRECIPITATOR_TYPES:
                    port = "solid" if i == 0 else "barren"
                elif u.type in IX_TYPES:
                    port = "loaded" if i == 0 else "eluate"
                elif u.type in REACTOR_TYPES:
                    port = "product"
                output_map[out_name] = (u.id, port)

        # Build input → unit map
        input_map = {}
        for u in flowsheet.units:
            for inp_name in u.inputs:
                input_map[inp_name] = u.id

        # For each stream that connects units, create linking constraints
        link_idx = 0
        for stream in flowsheet.streams:
            sname = stream.name
            if sname in output_map and sname in input_map:
                src_uid, src_port = output_map[sname]
                dst_uid = input_map[sname]

                if src_uid not in m.unit_blocks or dst_uid not in m.unit_blocks:
                    continue

                src_blk = m.unit_blocks[src_uid]
                dst_blk = m.unit_blocks[dst_uid]

                src_var = getattr(src_blk, src_port, None)
                dst_var = getattr(dst_blk, "feed", None)

                if src_var is None or dst_var is None:
                    continue

                # Create linking constraint
                def _link_rule(b, j, sv=src_var, dv=dst_var):
                    return sv[j] == dv[j]

                cname = f"link_{link_idx}"
                setattr(m, cname, Constraint(
                    comp_list, rule=_link_rule,
                    doc=f"Stream link: {sname} ({src_uid}.{src_port} → {dst_uid}.feed)"
                ))
                link_idx += 1

    # ─── Internal: objective ────────────────────────────────────

    def _add_objective(self, m, flowsheet, objective):
        """Add an objective function to the model."""
        if objective == "none":
            return

        # Collect REE recovery across all units
        ree_comps = [c for c in m.props.component_list if c in REE_SYMBOLS]
        if not ree_comps:
            return

        if objective == "maximize_recovery":
            # Maximize total REE in organic/solid/loaded outputs
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

        elif objective == "minimize_opex":
            # Simple OPEX proxy: sum of all unit flows × cost factor
            opex_terms = []
            for uid, blk in m.unit_blocks.items():
                if hasattr(blk, "feed_total"):
                    opex_terms.append(blk.feed_total * 0.01)
                if hasattr(blk, "OA_ratio"):
                    opex_terms.append(blk.OA_ratio * 5.0)
                if hasattr(blk, "reagent_dosage"):
                    opex_terms.append(blk.reagent_dosage * 20.0)
                if hasattr(blk, "resin_mass"):
                    opex_terms.append(blk.resin_mass * 10.0)
            if opex_terms:
                m.obj = Objective(expr=sum(opex_terms), sense=minimize)

    # ─── Internal: solve ────────────────────────────────────────

    def _solve(self, m, solver_name, solver_options):
        """Solve the model with the specified solver."""
        solver = SolverFactory(solver_name)
        if solver_options:
            for k, v in solver_options.items():
                solver.options[k] = v

        # Default IPOPT options for robustness
        if solver_name == "ipopt":
            solver.options.setdefault("max_iter", 3000)
            solver.options.setdefault("tol", 1e-6)
            solver.options.setdefault("print_level", 0)

        import os as _os, tempfile as _tf
        _old_tmpdir = _os.environ.get("TMPDIR")
        _os.environ["TMPDIR"] = _tf.gettempdir()
        try:
            # keepfiles=True prevents IPOPT ASL SIGSEGV (return code -11)
            # when Python ExternalFunction callbacks are used across
            # multiple solves in the same process.
            result = solver.solve(m, tee=False, keepfiles=True,
                                  load_solutions=True)
        finally:
            if _old_tmpdir is None:
                _os.environ.pop("TMPDIR", None)
            else:
                _os.environ["TMPDIR"] = _old_tmpdir
            # Clean up keepfiles artefacts
            import glob as _glob
            for f in _glob.glob(_os.path.join(_tf.gettempdir(), "tmp*.*")):
                try:
                    if f.endswith((".nl", ".sol", ".log")):
                        _os.remove(f)
                except OSError:
                    pass

        if result.solver.termination_condition not in (
            TerminationCondition.optimal,
            TerminationCondition.locallyOptimal,
            TerminationCondition.feasible,
        ):
            raise RuntimeError(
                f"Solver terminated with: {result.solver.termination_condition}. "
                f"Message: {result.solver.message}"
            )
        return result

    # ─── Internal: extract results ──────────────────────────────

    def _extract_states(self, m, flowsheet) -> Dict[str, Dict]:
        """Extract stream states from the solved model."""
        from ..sim.idaes_adapter import StreamState
        states = {}

        for unit in flowsheet.units:
            uid = unit.id
            blk = m.unit_blocks.get(uid)
            if blk is None:
                continue

            # Map output names to port attributes
            port_map = {}
            if unit.type in SX_TYPES:
                if len(unit.outputs) >= 1:
                    port_map[unit.outputs[0]] = "organic"
                if len(unit.outputs) >= 2:
                    port_map[unit.outputs[1]] = "raffinate"
            elif unit.type in PRECIPITATOR_TYPES:
                if len(unit.outputs) >= 1:
                    port_map[unit.outputs[0]] = "solid"
                if len(unit.outputs) >= 2:
                    port_map[unit.outputs[1]] = "barren"
            elif unit.type in IX_TYPES:
                if len(unit.outputs) >= 1:
                    port_map[unit.outputs[0]] = "loaded"
                if len(unit.outputs) >= 2:
                    port_map[unit.outputs[1]] = "eluate"
            elif unit.type in REACTOR_TYPES:
                if len(unit.outputs) >= 1:
                    port_map[unit.outputs[0]] = "product"
            else:
                if len(unit.outputs) >= 1:
                    port_map[unit.outputs[0]] = "organic"

            for stream_name, port_name in port_map.items():
                pvar = getattr(blk, port_name, None)
                if pvar is None:
                    continue

                species_amounts = {}
                for j in m.props.component_list:
                    if j in pvar:
                        v = value(pvar[j])
                        if v > 1e-15:
                            # Convert back to DSL naming (add +3 for REE ions)
                            dsl_name = f"{j}+3" if j in REE_SYMBOLS else j
                            if j == "H2O":
                                dsl_name = "H2O(aq)"
                            elif j == "HCl":
                                dsl_name = "HCl(aq)"
                            species_amounts[dsl_name] = v

                states[stream_name] = StreamState(
                    species_amounts=species_amounts,
                    flow_mol=sum(species_amounts.values()),
                )

            # Also extract feed states
            feed_var = getattr(blk, "feed", None)
            if feed_var is not None:
                for inp_name in unit.inputs:
                    if inp_name not in states:
                        sp = {}
                        for j in m.props.component_list:
                            if j in feed_var:
                                v = value(feed_var[j])
                                if v > 1e-15:
                                    dsl_name = f"{j}+3" if j in REE_SYMBOLS else j
                                    if j == "H2O":
                                        dsl_name = "H2O(aq)"
                                    elif j == "HCl":
                                        dsl_name = "HCl(aq)"
                                    sp[dsl_name] = v
                        states[inp_name] = StreamState(
                            species_amounts=sp,
                            flow_mol=sum(sp.values()),
                        )

        return states

    def _compute_kpis(self, m, flowsheet, states) -> Dict[str, float]:
        """Compute KPIs from the EO solution."""
        kpis = {}

        # Per-unit recovery
        for unit in flowsheet.units:
            blk = m.unit_blocks.get(unit.id)
            if blk is None:
                continue

            if hasattr(blk, "ree_recovery"):
                kpis[f"{unit.id}.ree_recovery"] = round(value(blk.ree_recovery), 4)

            # Extraction per component for SX
            if hasattr(blk, "extraction"):
                for j in m.props.component_list:
                    if j in REE_SYMBOLS and j in blk.extraction:
                        kpis[f"{unit.id}.E_{j}"] = round(value(blk.extraction[j]), 4)

        # Overall OPEX (from objective if available)
        if hasattr(m, "obj"):
            kpis["overall.opex_EO"] = round(value(m.obj), 4)

        # Compute via SM-compatible KPI path too
        try:
            from ..cost.tea import estimate_opex_usd
            from ..cost.lca import estimate_co2e
            kpis["overall.opex_USD"] = estimate_opex_usd(flowsheet, states)
            kpis["overall.lca_kg_CO2e"] = estimate_co2e(flowsheet, states)
        except Exception:
            pass

        return kpis


# ═══════════════════════════════════════════════════════════════════════
# Convenience function
# ═══════════════════════════════════════════════════════════════════════

def run_eo(
    flowsheet: Flowsheet,
    preset: str = "lree",
    objective: str = "minimize_opex",
) -> Dict[str, Any]:
    """One-line entry point: build + solve EO model + return results dict."""
    builder = EOFlowsheetBuilder(preset=preset)
    return builder.build_and_solve(flowsheet, objective=objective)
