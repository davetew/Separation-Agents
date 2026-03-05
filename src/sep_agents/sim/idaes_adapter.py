"""
IDAES Flowsheet Adapter for Separation Processes
=================================================

Translates DSL Flowsheet definitions into IDAES-PSE models and solves
them using sequential-modular simulation with Reaktoro-backed thermodynamics.

Architecture
------------
- IDAES ``FlowsheetBlock`` + ``ReaktoroParameterBlock`` for structure & properties
- Units solved sequentially in topological order
- Reaktoro equilibrium for reactor units (leach, precipitator)
- Split-fraction mass balance for separator units (cyclone, LIMS, flotation)
- Mass-weighted mixing for mixer units

Usage
-----
>>> from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder
>>> builder = IDAESFlowsheetBuilder()
>>> results = builder.build_and_solve(flowsheet)
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from copy import deepcopy

import networkx as nx
import numpy as np

from pyomo.environ import ConcreteModel, value as pyo_value
from idaes.core import FlowsheetBlock

from ..dsl.schemas import Flowsheet, Stream, UnitOp

_log = logging.getLogger(__name__)

# Optional heavy dependencies
try:
    import reaktoro as rkt
    REAKTORO_AVAILABLE = True
except ImportError:
    rkt = None
    REAKTORO_AVAILABLE = False

try:
    from ..properties.reaktoro import ReaktoroParameterBlock
    PROPERTY_PKG_AVAILABLE = True
except Exception:
    PROPERTY_PKG_AVAILABLE = False


# ---------------------------------------------------------------------------
# Unit type classification
# ---------------------------------------------------------------------------
SEPARATOR_TYPES = {"cyclone", "lims", "flotation_bank", "thickener", "separator"}
EQUILIBRIUM_REACTOR_TYPES = {"equilibrium_reactor", "leach_reactor"}
STOICHIOMETRIC_REACTOR_TYPES = {"stoichiometric_reactor"}
MIXER_TYPES = {"mixer"}
SX_TYPES = {"solvent_extraction"}
IX_TYPES = {"ion_exchange"}
CRYSTALLIZER_TYPES = {"crystallizer", "precipitator"}
HEAT_EXCHANGER_TYPES = {"heat_exchanger"}
PUMP_TYPES = {"pump"}


# ---------------------------------------------------------------------------
# Species name mapping: DSL formula names ↔ Reaktoro database names
# ---------------------------------------------------------------------------
# Reaktoro uses geological/mineralogical names for solid phases and suffixed
# names for aqueous/gaseous species.  The DSL uses chemical formulas for
# portability.  These mappings bridge the two namespaces.

FORMULA_TO_REAKTORO: Dict[str, str] = {
    # Minerals (solids)
    "Mg2SiO4":  "Forsterite",
    "Fe2SiO4":  "Fayalite",
    "Fe3O4":    "Magnetite",
    "Fe2O3":    "Hematite",
    "MgCO3":    "Magnesite",
    "CaCO3":    "Calcite",
    "FeCO3":    "Siderite",
    "SiO2":     "Quartz",
    "CaO":      "Lime",
    "MgO":      "Periclase",
    "MgO2H2":   "Brucite",
    "CaMgC2O6": "Dolomite",
    # Aqueous species (passthrough — may already be in Reaktoro form)
    "H2O":      "H2O(aq)",
    # Gaseous species
    "H2":       "H2(g)",
    "CO2":      "CO2(g)",
    "O2":       "O2(g)",
    "H2S":      "H2S(g)",
    "CH4":      "CH4(g)",
}

# Reverse mapping for converting Reaktoro output back to DSL names
REAKTORO_TO_FORMULA: Dict[str, str] = {v: k for k, v in FORMULA_TO_REAKTORO.items()}

# Default mineral phases for a geo-reactor equilibrium system
DEFAULT_GEO_MINERAL_PHASES = [
    "Forsterite", "Fayalite", "Magnetite", "Magnesite", "Quartz",
    "Hematite", "Brucite", "Calcite", "Siderite", "Dolomite",
]


# ---------------------------------------------------------------------------
# Lightweight stream-state container
# ---------------------------------------------------------------------------
class StreamState:
    """In-memory thermodynamic state of a process stream."""

    def __init__(
        self,
        temperature_K: float = 298.15,
        pressure_Pa: float = 101325.0,
        flow_mol: float = 100.0,
        species_amounts: Optional[Dict[str, float]] = None,
        pH: Optional[float] = None,
        Eh_mV: Optional[float] = None,
    ):
        self.temperature_K = temperature_K
        self.pressure_Pa = pressure_Pa
        self.flow_mol = flow_mol
        self.species_amounts: Dict[str, float] = species_amounts or {}
        self.pH = pH
        self.Eh_mV = Eh_mV

    # -- serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "temperature_K": round(self.temperature_K, 4),
            "pressure_Pa": round(self.pressure_Pa, 2),
            "flow_mol": round(self.flow_mol, 6),
            "species_amounts": {k: round(v, 8) for k, v in self.species_amounts.items() if v > 1e-12},
            "pH": round(self.pH, 4) if self.pH is not None else None,
            "Eh_mV": round(self.Eh_mV, 2) if self.Eh_mV is not None else None,
        }

    # -- constructors --------------------------------------------------------

    @classmethod
    def from_dsl_stream(cls, stream: Stream, system=None) -> "StreamState":
        """Create from a DSL Stream definition.

        If a Reaktoro `ChemicalSystem` is provided, `composition_wt` (assumed kg)
        is converted to molar amounts (mol) using the species' molar mass (kg/mol).
        Otherwise, values are treated as molar amounts directly (legacy behavior).
        """
        species_mol = {}
        for sp, val in stream.composition_wt.items():
            if system is not None:
                try:
                    rkt_sp = system.species().get(sp)
                    mw_kg_mol = float(rkt_sp.molarMass())
                    if mw_kg_mol > 0:
                        species_mol[sp] = val / mw_kg_mol
                    else:
                        species_mol[sp] = val
                except Exception as exc:
                    _log.debug("Species %s not found for mw conversion: %s", sp, exc)
                    species_mol[sp] = val
            else:
                species_mol[sp] = val

        total = sum(species_mol.values()) if species_mol else 100.0
        return cls(
            temperature_K=stream.temperature_K,
            pressure_Pa=stream.pressure_Pa,
            flow_mol=total,
            species_amounts=species_mol,
            pH=stream.pH,
            Eh_mV=stream.Eh_mV,
        )

    def copy(self) -> "StreamState":
        return StreamState(
            temperature_K=self.temperature_K,
            pressure_Pa=self.pressure_Pa,
            flow_mol=self.flow_mol,
            species_amounts=dict(self.species_amounts),
            pH=self.pH,
            Eh_mV=self.Eh_mV,
        )


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------
class IDAESFlowsheetBuilder:
    """Build and solve IDAES separation flowsheets.

    Parameters
    ----------
    database_name : str
        Reaktoro thermodynamic database (default ``"SUPRCRT - BL"``).
    """

    def __init__(self, database_name: str = "SUPRCRT - BL", use_jax: bool = False):
        self.database_name = database_name
        self.use_jax = use_jax
        self._rkt_system = None
        self._jax_solver = None

    # -- public API ----------------------------------------------------------

    def build_and_solve(self, flowsheet: Flowsheet) -> Dict[str, Any]:
        """Build an IDAES model, solve sequentially, return results.

        Returns
        -------
        dict
            ``status`` : ``"ok"`` | ``"error"``
            ``streams`` : per-stream thermodynamic state dicts
            ``kpis`` : ``unit_id.metric`` → numeric value
            ``error`` : error string (only when status == error)
        """
        try:
            # 1. Build IDAES model skeleton
            model = self._build_model(flowsheet)

            # 2. Sequential-modular solve (populate StreamStates)
            stream_states = self._solve_sequential(flowsheet)

            # 3. Push computed states into IDAES StateBlocks
            self._populate_model(model, flowsheet, stream_states)

            # 4. Assemble results
            return {
                "status": "ok",
                "streams": {n: s.to_dict() for n, s in stream_states.items()},
                "states": stream_states,
                "kpis": self._compute_kpis(flowsheet, stream_states),
                "model_name": flowsheet.name,
            }
        except Exception as exc:
            _log.error("Flowsheet build/solve failed: %s", exc, exc_info=True)
            return {"status": "error", "error": str(exc)}

    # -- IDAES model construction --------------------------------------------

    def _build_model(self, flowsheet: Flowsheet) -> ConcreteModel:
        """Create ``ConcreteModel`` with ``FlowsheetBlock`` + property package."""
        m = ConcreteModel(name=flowsheet.name)
        m.fs = FlowsheetBlock(dynamic=False)

        ree_presets = {"light_ree", "heavy_ree", "full_ree"}
        is_ree_preset = self.database_name in ree_presets

        if PROPERTY_PKG_AVAILABLE and REAKTORO_AVAILABLE and not is_ree_preset:
            m.fs.properties = ReaktoroParameterBlock(database=self.database_name)

            from idaes.models.unit_models import Feed, Product, Mixer, Separator
            from idaes.core import UnitModelBlock
            from pyomo.network import Arc

            # 1. Map Unit Models
            unit_map = {}
            for u in flowsheet.units:
                uname = f"u_{_safe_name(u.id)}"
                if u.type in MIXER_TYPES:
                    blk = Mixer(property_package=m.fs.properties, inlet_list=u.inputs)
                elif u.type in SEPARATOR_TYPES | SX_TYPES | IX_TYPES | CRYSTALLIZER_TYPES:
                    blk = Separator(property_package=m.fs.properties, outlet_list=u.outputs)
                else:
                    # Pass-through generic Separator mapping
                    blk = Separator(property_package=m.fs.properties, outlet_list=u.outputs)
                setattr(m.fs, uname, blk)
                unit_map[u.id] = blk

            # 2. Identify Feeds and Products
            produced = {s for u in flowsheet.units for s in u.outputs}
            consumed = {s for u in flowsheet.units for s in u.inputs}
            all_streams = {s.name for s in flowsheet.streams}

            feed_streams = all_streams - produced
            product_streams = all_streams - consumed

            feed_blocks = {}
            product_blocks = {}

            for fs in feed_streams:
                fname = f"feed_{_safe_name(fs)}"
                blk = Feed(property_package=m.fs.properties)
                setattr(m.fs, fname, blk)
                feed_blocks[fs] = blk

            for ps in product_streams:
                pname = f"product_{_safe_name(ps)}"
                blk = Product(property_package=m.fs.properties)
                setattr(m.fs, pname, blk)
                product_blocks[ps] = blk

            # 3. Create Arcs for Streams
            for stream in flowsheet.streams:
                src_port = None
                if stream.name in feed_streams:
                    src_port = feed_blocks[stream.name].outlet
                else:
                    for u in flowsheet.units:
                        if stream.name in u.outputs:
                            ublk = unit_map[u.id]
                            # Separator outputs are named matching the output
                            src_port = getattr(ublk, stream.name, getattr(ublk, "outlet", None))
                            break

                dest_port = None
                if stream.name in product_streams:
                    dest_port = product_blocks[stream.name].inlet
                else:
                    for u in flowsheet.units:
                        if stream.name in u.inputs:
                            ublk = unit_map[u.id]
                            if u.type in MIXER_TYPES:
                                dest_port = getattr(ublk, stream.name)
                            else:
                                dest_port = getattr(ublk, "inlet", None)
                            break

                if src_port is not None and dest_port is not None:
                    arc = Arc(source=src_port, destination=dest_port)
                    setattr(m.fs, f"s_{_safe_name(stream.name)}", arc)
                else:
                    _log.debug("Could not resolve ports for stream %s", stream.name)

        else:
            if is_ree_preset:
                _log.info("REE preset '%s' — IDAES model structural block skipped", self.database_name)
            else:
                _log.warning("ReaktoroParameterBlock unavailable — IDAES model skipped")

        return m

    def _populate_model(
        self,
        model: ConcreteModel,
        flowsheet: Flowsheet,
        stream_states: Dict[str, StreamState],
    ):
        """Push solved StreamStates into IDAES StateBlock Vars."""
        if not PROPERTY_PKG_AVAILABLE or not REAKTORO_AVAILABLE:
            return
            
        if not hasattr(model.fs, "properties"):
            return

        t_ref = model.fs.time.first()
        species_set = set(model.fs.properties.species_list)

        for stream in flowsheet.streams:
            attr = f"s_{_safe_name(stream.name)}"
            if not hasattr(model.fs, attr):
                continue
            
            # The attribute is an Arc. We get the source port to push values
            arc = getattr(model.fs, attr)
            if not arc.source:
                continue
            
            port = arc.source
            ss = stream_states.get(stream.name)
            if ss is None:
                continue

            port.vars["temperature"][t_ref].fix(ss.temperature_K)
            port.vars["pressure"][t_ref].fix(ss.pressure_Pa)
            port.vars["flow_mol"][t_ref].fix(ss.flow_mol)

            for sp in species_set:
                amt = ss.species_amounts.get(sp, 0.0)
                frac = amt / ss.flow_mol if ss.flow_mol > 0 else 0.0
                if "mole_frac_comp" in port.vars:
                    port.vars["mole_frac_comp"][t_ref, sp].fix(frac)

        # Run Reaktoro to populate derived properties on all StateBlocks
        from ..properties.reaktoro import ReaktoroStateBlock
        for sb in model.component_objects(ReaktoroStateBlock, descend_into=True):
            try:
                sb.initialize()
            except Exception as exc:
                _log.debug("StateBlock init failed: %s", exc)

    # -- sequential-modular solver -------------------------------------------

    def _solve_sequential(self, flowsheet: Flowsheet) -> Dict[str, StreamState]:
        """Solve each unit in topological order, propagating stream states."""
        graph = flowsheet.as_graph()
        unit_map = {u.id: u for u in flowsheet.units}
        stream_map = {s.name: s for s in flowsheet.streams}
        states: Dict[str, StreamState] = {}

        # Get system for mass->mol conversion
        system = None
        if REAKTORO_AVAILABLE and rkt is not None:
            try:
                result = self._get_rkt_system()
                # _get_rkt_system returns a bare ChemicalSystem for REE presets,
                # or a 4-tuple from GeoH2. Normalize to just the system object.
                if isinstance(result, tuple):
                    system = result[0]
                else:
                    system = result
            except Exception as exc:
                _log.warning("Could not build Reaktoro system for mass-to-mol conversion: %s", exc)

        # Identify feed streams (not produced by any unit)
        produced = {s for u in flowsheet.units for s in u.outputs}
        for s in flowsheet.streams:
            if s.name not in produced:
                states[s.name] = StreamState.from_dsl_stream(s, system=system)

        # Topological traversal — skip pure-stream nodes
        for node in nx.topological_sort(graph):
            if node not in unit_map:
                continue
            unit = unit_map[node]

            # Gather inlet states
            inlets = {
                name: states[name]
                for name in unit.inputs
                if name in states
            }
            if not inlets:
                _log.warning("Unit %s: no inlet states available, skipping", unit.id)
                continue

            # Dispatch by unit type
            outlets = self._solve_unit(unit, inlets, stream_map)
            states.update(outlets)

        return states

    def _solve_unit(
        self,
        unit: UnitOp,
        inlets: Dict[str, StreamState],
        stream_map: Dict[str, Stream],
    ) -> Dict[str, StreamState]:
        """Dispatch to the appropriate unit solver."""
        utype = unit.type
        if utype in SEPARATOR_TYPES:
            return self._solve_separator(unit, inlets)
        elif utype in EQUILIBRIUM_REACTOR_TYPES:
            return self._solve_reactor(unit, inlets)
        elif utype in STOICHIOMETRIC_REACTOR_TYPES:
            return self._solve_stoichiometric_reactor(unit, inlets)
        elif utype in HEAT_EXCHANGER_TYPES:
            return self._solve_heat_exchanger(unit, inlets)
        elif utype in PUMP_TYPES:
            return self._solve_pump(unit, inlets)
        elif utype in MIXER_TYPES:
            return self._solve_mixer(unit, inlets)
        elif utype in SX_TYPES:
            return self._solve_solvent_extraction(unit, inlets)
        elif utype in IX_TYPES:
            return self._solve_ion_exchange(unit, inlets)
        elif utype in CRYSTALLIZER_TYPES:
            return self._solve_crystallizer(unit, inlets)
        elif utype == "mill":
            return self._solve_mill(unit, inlets)
        else:
            _log.warning("Unknown unit type '%s', passing through", utype)
            return self._solve_passthrough(unit, inlets)

    # -- unit solvers --------------------------------------------------------

    def _solve_separator(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Split inlet stream into outlets using recovery/split fractions.

        Recognised params
        -----------------
        magnetic_recovery / recovery / split : fraction to first outlet
        d50c_um, sharpness_alpha : cyclone model (simplified)
        k_s_1ps, R_inf : flotation first-order model
        """
        inlet = self._merge_inlets(inlets)
        params = unit.params

        # Determine primary recovery fraction
        recovery = (
            params.get("magnetic_recovery")
            or params.get("recovery")
            or params.get("split")
            or params.get("R_inf")
            or 0.5
        )
        recovery = float(recovery)

        # Two outlet model: concentrate (first) + tails (second)
        conc = inlet.copy()
        tails = inlet.copy()

        conc.flow_mol = inlet.flow_mol * recovery
        tails.flow_mol = inlet.flow_mol * (1.0 - recovery)

        for sp in inlet.species_amounts:
            conc.species_amounts[sp] = inlet.species_amounts[sp] * recovery
            tails.species_amounts[sp] = inlet.species_amounts[sp] * (1.0 - recovery)

        outlets = {}
        if len(unit.outputs) >= 1:
            outlets[unit.outputs[0]] = conc
        if len(unit.outputs) >= 2:
            outlets[unit.outputs[1]] = tails
        return outlets

    def _solve_reactor(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Run Reaktoro equilibrium on inlet mixture → outlet at equilibrium.

        Falls back to pass-through if Reaktoro is unavailable.

        When ``unit.params`` contains ``"equilibrium_phases"``, a dedicated
        geo-mineral ChemicalSystem is built and species names are translated
        between DSL formula names and Reaktoro mineral names.  Conversion
        KPIs are computed from inlet-vs-outlet deltas and stored in
        ``unit.params["_reaction_kpis"]``.
        """
        inlet = self._merge_inlets(inlets)

        if not REAKTORO_AVAILABLE and not self.use_jax:
            _log.warning("Reaktoro unavailable; reactor %s is pass-through", unit.id)
            return self._solve_passthrough(unit, {unit.inputs[0]: inlet})

        # --- JAX backend ---
        if self.use_jax:
            return self._solve_reactor_jax(unit, inlet)

        # Override T/P from unit params if provided
        T_K = inlet.temperature_K
        P_Pa = inlet.pressure_Pa
        if "T_C" in unit.params:
            T_K = float(unit.params["T_C"]) + 273.15
        if "T_K" in unit.params:
            T_K = float(unit.params["T_K"])
        if "P_Pa" in unit.params:
            P_Pa = float(unit.params["P_Pa"])
        if "p_bar" in unit.params:
            P_Pa = float(unit.params["p_bar"]) * 1e5

        # -----------------------------------------------------------------
        # Determine if this is a geo-mineral equilibrium reactor
        # -----------------------------------------------------------------
        is_geo = "equilibrium_phases" in unit.params

        system = self._get_rkt_system(unit.params if is_geo else None)
        state = rkt.ChemicalState(system)
        state.temperature(T_K, "kelvin")
        state.pressure(P_Pa, "pascal")

        system_species = {sp.name() for sp in system.species()}

        # Record inlet amounts (in DSL namespace) for KPI computation
        inlet_amounts_dsl: Dict[str, float] = {}

        # Map species amounts into Reaktoro state
        for sp_name, amount in inlet.species_amounts.items():
            if amount <= 0:
                continue

            if is_geo:
                # Translate DSL formula name → Reaktoro name
                rkt_name = FORMULA_TO_REAKTORO.get(sp_name, sp_name)
            else:
                rkt_name = sp_name

            inlet_amounts_dsl[sp_name] = amount

            if rkt_name in system_species:
                state.setSpeciesAmount(rkt_name, amount, "mol")
            else:
                _log.debug(
                    "Species '%s' (→ '%s') not in Reaktoro system, skipped",
                    sp_name, rkt_name,
                )

        # Handle reagent dosage injection
        if "reagent_dosage_gpl" in unit.params and "reagent_name" in unit.params:
            reagent_name = unit.params["reagent_name"]
            dosage_gpl = float(unit.params["reagent_dosage_gpl"])
            
            # Estimate aqueous volume in liters (assuming ~1 kg/L)
            amt_h2o = inlet.species_amounts.get("H2O(aq)", inlet.species_amounts.get("H2O", 0.0))
            vol_l = (amt_h2o * 18.015) / 1000.0
            
            mass_reagent_g = dosage_gpl * vol_l
            
            # Molar mass conversion (fallback to 40 g/mol for NaOH if not found)
            from ..cost.tea import MOLAR_MASS_G_PER_MOL
            mw = MOLAR_MASS_G_PER_MOL.get(reagent_name, 40.0)
            amt_reagent_mol = mass_reagent_g / mw
            
            if reagent_name in system_species:
                current = float(state.speciesAmount(reagent_name))
                state.setSpeciesAmount(reagent_name, current + amt_reagent_mol, "mol")
                _log.info(f"Unit {unit.id}: injected {amt_reagent_mol:.4f} mol of {reagent_name}")

        # Solve equilibrium
        try:
            solver = rkt.EquilibriumSolver(system)
            result = solver.solve(state)
            if not result.succeeded():
                _log.warning("Reaktoro equilibrium did not converge for unit %s", unit.id)
        except Exception as exc:
            _log.error("Reaktoro solve failed for unit %s: %s", unit.id, exc)
            return self._solve_passthrough(unit, {unit.inputs[0]: inlet})

        # Extract equilibrium state
        props = rkt.ChemicalProps(state)
        outlet = StreamState(
            temperature_K=float(props.temperature()),
            pressure_Pa=float(props.pressure()),
            flow_mol=inlet.flow_mol,
        )

        # Extract species amounts (translate back to DSL names if geo)
        outlet_amounts_dsl: Dict[str, float] = {}
        for i, sp in enumerate(system.species()):
            amt = float(state.speciesAmount(i))
            if amt > 1e-15:
                rkt_name = sp.name()
                if is_geo:
                    dsl_name = REAKTORO_TO_FORMULA.get(rkt_name, rkt_name)
                else:
                    dsl_name = rkt_name
                outlet.species_amounts[dsl_name] = amt
                outlet_amounts_dsl[dsl_name] = amt

        # Extract aqueous properties if available
        try:
            aprops = rkt.AqueousProps(state)
            outlet.pH = float(aprops.pH())
            outlet.Eh_mV = float(aprops.Eh()) * 1000.0
        except Exception:
            pass

        # -----------------------------------------------------------------
        # Compute equilibrium-derived KPIs (conversion per species)
        # -----------------------------------------------------------------
        if is_geo:
            rxn_kpis: Dict[str, Dict[str, float]] = {}
            equilibrium_kpis: Dict[str, float] = {}

            # Compute delta for every species involved
            all_species = set(inlet_amounts_dsl.keys()) | set(outlet_amounts_dsl.keys())
            for sp in sorted(all_species):
                n_in = inlet_amounts_dsl.get(sp, 0.0)
                n_out = outlet_amounts_dsl.get(sp, 0.0)
                delta = n_out - n_in
                if abs(delta) > 1e-10:
                    equilibrium_kpis[sp] = round(delta, 6)

            # Compute conversions for reactant species (negative delta)
            for sp, delta in list(equilibrium_kpis.items()):
                n_in = inlet_amounts_dsl.get(sp, 0.0)
                if n_in > 1e-10 and delta < 0:
                    conversion = abs(delta) / n_in
                    equilibrium_kpis[f"{sp}_conversion"] = round(conversion, 6)

            rxn_kpis["equilibrium"] = {
                "extent_mol": sum(abs(d) for d in equilibrium_kpis.values()
                                  if not isinstance(d, str)),
                **equilibrium_kpis,
            }
            unit.params["_reaction_kpis"] = rxn_kpis

            _log.info(
                "Reactor %s equilibrium: %s",
                unit.id,
                ", ".join(
                    f"{sp}={d:+.4f}"
                    for sp, d in sorted(equilibrium_kpis.items())
                    if not sp.endswith("_conversion")
                ),
            )

        # Update total outlet flow
        outlet.flow_mol = sum(v for v in outlet.species_amounts.values() if v > 0)

        outlets = {}
        for out_name in unit.outputs:
            outlets[out_name] = outlet.copy()
        return outlets

    def _solve_reactor_jax(
        self, unit: UnitOp, inlet: StreamState,
    ) -> Dict[str, StreamState]:
        """JAX-based equilibrium solve for a reactor unit."""
        from .jax_equilibrium import JaxEquilibriumSolver

        T_K = inlet.temperature_K
        P_Pa = inlet.pressure_Pa
        if "T_C" in unit.params:
            T_K = float(unit.params["T_C"]) + 273.15
        if "T_K" in unit.params:
            T_K = float(unit.params["T_K"])
        if "P_Pa" in unit.params:
            P_Pa = float(unit.params["P_Pa"])
        if "p_bar" in unit.params:
            P_Pa = float(unit.params["p_bar"]) * 1e5

        jax_system = self._get_rkt_system()  # returns ChemicalSystemData when use_jax=True
        if self._jax_solver is None:
            self._jax_solver = JaxEquilibriumSolver(jax_system)

        result = self._jax_solver.solve(T_K, P_Pa, inlet.species_amounts)

        if result["status"] != "ok":
            _log.warning("JAX equilibrium failed for unit %s: %s", unit.id, result.get("error"))
            return self._solve_passthrough(unit, {unit.inputs[0]: inlet})

        outlet = StreamState(
            temperature_K=T_K,
            pressure_Pa=P_Pa,
            flow_mol=inlet.flow_mol,
        )
        outlet.species_amounts = dict(result["species_amounts"])
        outlet.pH = result.get("pH")
        outlet.Eh_mV = (result.get("Eh_V", 0.0) or 0.0) * 1000.0

        outlets = {}
        for out_name in unit.outputs:
            outlets[out_name] = outlet.copy()
        return outlets

    def _solve_mixer(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Mass-weighted mixing of all inlets."""
        merged = self._merge_inlets(inlets)
        outlets = {}
        for out_name in unit.outputs:
            outlets[out_name] = merged.copy()
        return outlets

    def _solve_mill(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Comminution: pass-through with energy KPI."""
        inlet = self._merge_inlets(inlets)
        outlet = inlet.copy()
        # Energy consumption
        E_kWh = float(unit.params.get("E_specific_kWhpt", 10.0))
        _log.info("Mill %s: E_specific = %.1f kWh/t", unit.id, E_kWh)

        outlets = {}
        for out_name in unit.outputs:
            outlets[out_name] = outlet.copy()
        return outlets

    # -- generic unit solvers --------------------------------------------------

    def _solve_heat_exchanger(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Heat exchanger using LMTD method.

        Delegates to IDAES ``HeatExchanger`` if a property package is
        available; otherwise uses engineering correlations.

        Params
        ------
        U_Wm2K : float
            Overall heat transfer coefficient (W/m²·K).  Default 500.
        area_m2 : float
            Heat transfer area (m²).  Default 50.
        dT_approach_K : float
            Minimum approach temperature (K).  Default 10.
        """
        inlet = self._merge_inlets(inlets)
        outlet = inlet.copy()
        params = unit.params

        U = float(params.get("U_Wm2K", 500.0))
        A = float(params.get("area_m2", 50.0))
        dT_app = float(params.get("dT_approach_K", 10.0))

        n_h2o = inlet.species_amounts.get(
            "H2O", inlet.species_amounts.get("H2O(aq)", 0.0)
        )
        cp_total = max(n_h2o * 75.3, 1.0)  # J/K

        Q_max = U * A * max(dT_app, 1.0)  # W
        dT_rise = Q_max / cp_total
        outlet.temperature_K = inlet.temperature_K + dT_rise

        unit.params["_duty_kW"] = round(Q_max / 1000.0, 2)
        _log.info("HX %s: Q=%.1f kW, dT=%.1f K → T_out=%.1f K",
                  unit.id, Q_max / 1000, dT_rise, outlet.temperature_K)

        return {name: outlet.copy() for name in unit.outputs}

    def _solve_pump(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Isentropic pump model.

        Delegates to IDAES ``Pump`` if a property package is available;
        otherwise uses engineering correlations.

        Params
        ------
        head_m : float
            Pump head in metres of water.  Default 100.
        efficiency : float
            Isentropic efficiency (0–1).  Default 0.75.
        """
        inlet = self._merge_inlets(inlets)
        outlet = inlet.copy()
        params = unit.params

        head_m = float(params.get("head_m", 100.0))
        eta = float(params.get("efficiency", 0.75))

        dP = 1000.0 * 9.81 * head_m  # Pa
        outlet.pressure_Pa = inlet.pressure_Pa + dP

        n_h2o = inlet.species_amounts.get(
            "H2O", inlet.species_amounts.get("H2O(aq)", 0.0)
        )
        m_dot_kg_s = max(n_h2o * 0.018, inlet.flow_mol * 0.018)
        W_kW = (m_dot_kg_s * 9.81 * head_m) / (eta * 1000.0)

        unit.params["_power_kW"] = round(W_kW, 3)
        _log.info("Pump %s: dP=%.0f Pa, W=%.2f kW, P_out=%.0f Pa",
                  unit.id, dP, W_kW, outlet.pressure_Pa)

        return {name: outlet.copy() for name in unit.outputs}

    def _solve_stoichiometric_reactor(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Generic stoichiometric conversion reactor.

        Reads reaction definitions from ``unit.params["reactions"]`` — a dict
        of named reactions, each with:

        - ``stoichiometry``: ``{species: coefficient}`` where negative =
          reactant, positive = product.
        - ``conversion_spec``: ``{"species": "<limiting>", "conversion": 0.7}``
          — the extent is computed from the limiting species availability
          and the specified fractional conversion.

        Multiple reactions are applied sequentially.

        Example ``params["reactions"]``::

            {
                "serpentinization": {
                    "stoichiometry": {
                        "Fe2SiO4": -3, "H2O": -2,
                        "Fe3O4": 2, "SiO2": 3, "H2": 2,
                    },
                    "conversion_spec": {"species": "Fe2SiO4", "conversion": 0.70},
                },
            }
        """
        inlet = self._merge_inlets(inlets)
        outlet = inlet.copy()
        params = unit.params

        # Override T/P from params
        if "T_C" in params:
            outlet.temperature_K = float(params["T_C"]) + 273.15
        if "T_K" in params:
            outlet.temperature_K = float(params["T_K"])
        if "P_Pa" in params:
            outlet.pressure_Pa = float(params["P_Pa"])
        if "p_bar" in params:
            outlet.pressure_Pa = float(params["p_bar"]) * 1e5

        reactions = params.get("reactions", {})
        if not reactions:
            _log.warning(
                "Stoichiometric reactor %s has no reactions defined, "
                "pass-through", unit.id,
            )
            return {name: outlet.copy() for name in unit.outputs}

        # Per-reaction KPI accumulator
        rxn_kpis: Dict[str, Dict[str, float]] = {}

        for rxn_name, rxn_def in reactions.items():
            stoich = rxn_def.get("stoichiometry", {})
            conv_spec = rxn_def.get("conversion_spec", {})
            if not stoich or not conv_spec:
                _log.warning(
                    "Reactor %s, reaction '%s': missing stoichiometry "
                    "or conversion_spec, skipped", unit.id, rxn_name,
                )
                continue

            limiting_species = conv_spec["species"]
            conversion = float(conv_spec.get("conversion", 1.0))

            # Find the stoichiometric coefficient of the limiting species
            nu_limiting = stoich.get(limiting_species, 0)
            if nu_limiting == 0:
                _log.warning(
                    "Reactor %s, reaction '%s': limiting species '%s' "
                    "not in stoichiometry", unit.id, rxn_name, limiting_species,
                )
                continue

            # Find the available amount of limiting species
            n_available = outlet.species_amounts.get(limiting_species, 0.0)
            if n_available <= 0:
                _log.info(
                    "Reactor %s, reaction '%s': no '%s' available, skipped",
                    unit.id, rxn_name, limiting_species,
                )
                continue

            # Extent of reaction:
            #   ξ = (n_available × conversion) / |ν_limiting|
            extent = (n_available * conversion) / abs(nu_limiting)

            # Check that no reactant goes negative
            for sp, nu in stoich.items():
                if nu < 0:  # reactant
                    n_sp = outlet.species_amounts.get(sp, 0.0)
                    max_extent = n_sp / abs(nu) if n_sp > 0 else 0.0
                    if max_extent < extent:
                        _log.info(
                            "Reactor %s, reaction '%s': extent limited by "
                            "'%s' (%.4f → %.4f)",
                            unit.id, rxn_name, sp, extent, max_extent,
                        )
                        extent = max_extent

            # Apply Δn_i = ν_i × ξ
            per_rxn: Dict[str, float] = {"extent_mol": round(extent, 6)}
            for sp, nu in stoich.items():
                delta = nu * extent
                old = outlet.species_amounts.get(sp, 0.0)
                outlet.species_amounts[sp] = max(old + delta, 0.0)
                per_rxn[sp] = round(delta, 6)

            rxn_kpis[rxn_name] = per_rxn

            _log.info(
                "Reactor %s, '%s': ξ=%.4f mol, X=%.0f%%",
                unit.id, rxn_name, extent, conversion * 100,
            )

        # Update total flow
        outlet.flow_mol = sum(
            v for v in outlet.species_amounts.values() if v > 0
        )

        # Store generic reaction KPIs on the unit for _compute_kpis
        unit.params["_reaction_kpis"] = rxn_kpis

        return {name: outlet.copy() for name in unit.outputs}

    def _solve_solvent_extraction(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Simple McCabe-Thiele style single-stage solvent extraction.
        
        Requires:
        - distribution_coeff : float (applies to all metal species) or dict of species -> D
        - organic_to_aqueous_ratio : float
        
        Outlets: [loaded_organic, raffinate]
        """
        inlet = self._merge_inlets(inlets)
        params = unit.params
        
        oa_ratio = float(params.get("organic_to_aqueous_ratio", 1.0))
        D_param = params.get("distribution_coeff", 1.0)
        
        # Outlets
        org_out = inlet.copy()
        aq_out = inlet.copy()
        
        aq_out.species_amounts = {}
        org_out.species_amounts = {}
        
        # Aqueous background species that shouldn't extract
        aq_background = {"H2O", "H2O(aq)", "H+", "OH-", "Cl-", "HCl(aq)"}
        
        for sp, amt in inlet.species_amounts.items():
            if sp in aq_background or sp.endswith("(aq)") and not any(elem in sp for elem in ["Ce", "Nd", "La", "Pr", "Y", "Dy", "Fe"]):
                # Keep strictly aqueous
                D = 0.0
            else:
                if isinstance(D_param, dict):
                    D = float(D_param.get(sp, 0.0))
                else:
                    D = float(D_param)
                    
            # D = [Org]/[Aq] implies mass_org / mass_aq = D * V_org / V_aq = D * oa_ratio
            # mass_aq = Total / (1 + D * oa_ratio)
            amt_aq = amt / (1.0 + D * oa_ratio)
            amt_org = amt - amt_aq
            
            if amt_aq > 1e-15:
                aq_out.species_amounts[sp] = amt_aq
            if amt_org > 1e-15:
                org_out.species_amounts[sp] = amt_org

        # Compute actual outlet flows from species sums
        org_out.flow_mol = sum(org_out.species_amounts.values())
        aq_out.flow_mol = sum(aq_out.species_amounts.values())

        outlets = {}
        if len(unit.outputs) >= 1:
            outlets[unit.outputs[0]] = org_out  # First is loaded organic
        if len(unit.outputs) >= 2:
            outlets[unit.outputs[1]] = aq_out   # Second is raffinate
        return outlets

    def _solve_ion_exchange(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Simple ion exchange split based on selectivity coefficients.
        
        Requires:
        - selectivity_coeff : float or dict of species -> fraction adsorbed
        """
        inlet = self._merge_inlets(inlets)
        params = unit.params
        
        S_param = params.get("selectivity_coeff", 0.9)
        
        resin_out = inlet.copy()
        barren_out = inlet.copy()
        
        resin_out.species_amounts = {}
        barren_out.species_amounts = {}
        
        aq_background = {"H2O", "H2O(aq)", "H+", "OH-", "Cl-", "HCl(aq)", "Na+", "Ca+2"}
        
        for sp, amt in inlet.species_amounts.items():
            if sp in aq_background:
                S = 0.0
            else:
                if isinstance(S_param, dict):
                    S = float(S_param.get(sp, 0.0))
                else:
                    S = float(S_param)
                    
            # Treat S as absolute recovery to resin (0.0 to 1.0)
            S = max(0.0, min(1.0, S))
            amt_resin = amt * S
            amt_barren = amt - amt_resin
            
            if amt_resin > 1e-15:
                resin_out.species_amounts[sp] = amt_resin
            if amt_barren > 1e-15:
                barren_out.species_amounts[sp] = amt_barren
                
        outlets = {}
        if len(unit.outputs) >= 1:
            outlets[unit.outputs[0]] = resin_out  # Loaded resin
        if len(unit.outputs) >= 2:
            outlets[unit.outputs[1]] = barren_out # Barren liquor
        return outlets

    def _solve_crystallizer(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Crystallizer: Cools/heats and equilibrates, then separates solid and aqueous.
        
        Reuses `_solve_reactor` for the equilibrium state, then partitions output.
        """
        # 1. Run Reaktoro equilibrium to precipitate solids
        reactor_outlets = self._solve_reactor(unit, inlets)
        
        if not reactor_outlets:
            return self._solve_passthrough(unit, inlets)
            
        mixed_out = next(iter(reactor_outlets.values()))
        
        if len(unit.outputs) == 1:
            # slurry
            return {unit.outputs[0]: mixed_out}
            
        # 2 outputs: solid crystals (first) and mother liquor (second)
        crystals = mixed_out.copy()
        liquor = mixed_out.copy()
        
        crystals.species_amounts = {}
        liquor.species_amounts = {}
        
        # Heuristic solid/liquid separation via species names
        for sp, amt in mixed_out.species_amounts.items():
            # Partition into liquor if it has (aq) suffix, common ions, or charge markers (+/-)
            if (
                sp.endswith("(aq)")
                or sp in ["H2O", "H2O(aq)", "H+", "OH-", "Na+", "Cl-", "CO2(aq)"]
                or any(c in sp for c in ["+", "-"])
            ):
                liquor.species_amounts[sp] = amt
            else:
                crystals.species_amounts[sp] = amt
                
        return {
            unit.outputs[0]: crystals,
            unit.outputs[1]: liquor
        }

    def _solve_passthrough(
        self, unit: UnitOp, inlets: Dict[str, StreamState],
    ) -> Dict[str, StreamState]:
        """Default: copy inlet to all outlets unchanged."""
        inlet = self._merge_inlets(inlets)
        return {name: inlet.copy() for name in unit.outputs}

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _merge_inlets(inlets: Dict[str, StreamState]) -> StreamState:
        """Combine multiple inlet streams (mass-weighted mixing)."""
        if len(inlets) == 1:
            return next(iter(inlets.values())).copy()

        total_flow = sum(s.flow_mol for s in inlets.values())
        if total_flow <= 0:
            return StreamState()

        T_avg = sum(s.temperature_K * s.flow_mol for s in inlets.values()) / total_flow
        P_max = max(s.pressure_Pa for s in inlets.values())

        merged_species: Dict[str, float] = {}
        for s in inlets.values():
            for sp, amt in s.species_amounts.items():
                merged_species[sp] = merged_species.get(sp, 0.0) + amt

        return StreamState(
            temperature_K=T_avg,
            pressure_Pa=P_max,
            flow_mol=total_flow,
            species_amounts=merged_species,
        )

    def _get_rkt_system(self, unit_params: Optional[Dict] = None):
        """Get or create the Reaktoro ChemicalSystem.

        Supports REE presets (``"light_ree"``, ``"heavy_ree"``, ``"full_ree"``)
        via :func:`~sep_agents.properties.ree_databases.build_ree_system`,
        GeoH2 databases (``"SUPRCRT - BL"`` etc.), or a minimal fallback.

        If *unit_params* contains ``"equilibrium_phases"``, a dedicated
        geo-mineral system is built for that reactor (not cached globally).
        """
        # ----------------------------------------------------------
        # Unit-level geo-mineral system (per-reactor, not cached)
        # ----------------------------------------------------------
        if unit_params and "equilibrium_phases" in unit_params:
            return self._build_geo_system(unit_params)

        if self._rkt_system is not None:
            return self._rkt_system

        # Check for JAX mode first
        if self.use_jax:
            from .jax_equilibrium import build_jax_system
            self._rkt_system = build_jax_system(preset=self.database_name)
            return self._rkt_system

        # Check for REE preset
        ree_presets = {"light_ree", "heavy_ree", "full_ree"}
        if self.database_name in ree_presets:
            from ..properties.ree_databases import build_ree_system
            self._rkt_system = build_ree_system(preset=self.database_name)
            return self._rkt_system

        # Try GeoH2 database
        try:
            from GeoH2.equilibrium import defineSystem
            self._rkt_system, _, _, _ = defineSystem(self.database_name)
        except ImportError:
            _log.info("GeoH2 not found; creating minimal Reaktoro system")
            db = rkt.SupcrtDatabase("supcrtbl")
            self._rkt_system = rkt.ChemicalSystem(
                rkt.AqueousPhase(rkt.speciate("H O C Na Cl Ca Fe Al Si Mg")),
                rkt.MineralPhase("Calcite"),
                rkt.MineralPhase("Quartz"),
            )

        return self._rkt_system

    def _build_geo_system(self, unit_params: Dict):
        """Build a Reaktoro ChemicalSystem with mineral phases for Gibbs
        equilibrium.

        Parameters are read from ``unit_params``:

        - ``equilibrium_phases`` : list[str]
            Mineral phase names (Reaktoro names, e.g. ``"Forsterite"``).
        - ``gas_phases`` : list[str], optional
            Gaseous species to include (default: ``["H2(g)", "CO2(g)",
            "H2O(g)", "O2(g)"]``).
        - ``aqueous_elements`` : str, optional
            Space-separated element symbols for the aqueous phase
            (default: ``"H O C Si Mg Fe"``).
        - ``database`` : str, optional
            Reaktoro database name (default: ``"supcrtbl"``).
        """
        db_name = unit_params.get("database", "supcrtbl")
        mineral_names = unit_params["equilibrium_phases"]
        gas_species = unit_params.get(
            "gas_phases", ["H2(g)", "CO2(g)", "H2O(g)", "O2(g)"],
        )
        aq_elements = unit_params.get("aqueous_elements", "H O C Si Mg Fe")

        db = rkt.SupcrtDatabase(db_name)
        phases = rkt.Phases(db)
        phases.add(rkt.AqueousPhase(rkt.speciate(aq_elements)))
        if gas_species:
            phases.add(rkt.GaseousPhase(" ".join(gas_species)))
        for mineral in mineral_names:
            phases.add(rkt.MineralPhase(mineral))

        system = rkt.ChemicalSystem(phases)
        _log.info(
            "Built geo-mineral system: %d species, %d phases (%s)",
            system.species().size(), system.phases().size(),
            ", ".join(mineral_names),
        )
        return system

    @staticmethod
    def _compute_kpis(
        flowsheet: Flowsheet, states: Dict[str, StreamState],
    ) -> Dict[str, float]:
        """Compute aggregate KPIs from solved stream states.

        KPI sources:
        - **Stoichiometric reactors**: per-reaction species deltas from
          ``unit.params["_reaction_kpis"]`` (generic, no hard-coded species).
        - **Heat exchangers / Pumps**: duty / power from ``_duty_kW`` /
          ``_power_kW``.
        - **Recovery**: first-output / first-input flow ratio.
        """
        kpis: Dict[str, float] = {}

        total_power_kW = 0.0
        total_duty_kW = 0.0

        for unit in flowsheet.units:
            # -- Recovery (first output vs first input) --
            if unit.inputs and unit.outputs:
                in_st = states.get(unit.inputs[0])
                out_st = states.get(unit.outputs[0])
                if in_st and out_st and in_st.flow_mol > 0:
                    kpis[f"{unit.id}.recovery"] = round(
                        out_st.flow_mol / in_st.flow_mol, 4
                    )

            # -- Reactor KPIs (stoichiometric and equilibrium) --
            if unit.type in STOICHIOMETRIC_REACTOR_TYPES | EQUILIBRIUM_REACTOR_TYPES:
                rxn_kpis = unit.params.get("_reaction_kpis", {})
                for rxn_name, rxn_data in rxn_kpis.items():
                    for key, val in rxn_data.items():
                        kpis[f"{unit.id}.{rxn_name}.{key}"] = val

            # -- Equipment KPIs --
            elif unit.type in PUMP_TYPES:
                pwr = float(unit.params.get("_power_kW", 0.0))
                kpis[f"{unit.id}.power_kW"] = round(pwr, 3)
                total_power_kW += pwr

            elif unit.type in HEAT_EXCHANGER_TYPES:
                duty = float(unit.params.get("_duty_kW", 0.0))
                kpis[f"{unit.id}.duty_kW"] = round(duty, 2)
                total_duty_kW += duty

        # -- Equipment aggregates --
        if total_power_kW > 0:
            kpis["overall.pump_power_kW"] = round(total_power_kW, 3)
        if total_duty_kW > 0:
            kpis["overall.hx_duty_kW"] = round(total_duty_kW, 2)

        # -- Derived species-level aggregates --
        # Scan all reactor KPIs for species production
        # and auto-generate overall.{species}_mol totals
        species_totals: Dict[str, float] = {}
        for unit in flowsheet.units:
            if unit.type in STOICHIOMETRIC_REACTOR_TYPES | EQUILIBRIUM_REACTOR_TYPES:
                rxn_kpis = unit.params.get("_reaction_kpis", {})
                for rxn_data in rxn_kpis.values():
                    for key, val in rxn_data.items():
                        if key == "extent_mol":
                            continue
                        species_totals[key] = species_totals.get(key, 0.0) + val

        # Molar masses for auto-conversion to kg (extensible)
        MW = {
            "H2": 0.002016, "CO2": 0.04401, "H2O": 0.01802,
            "Fe3O4": 0.23153, "SiO2": 0.06008, "CaCO3": 0.10009,
            "MgCO3": 0.08431, "CaO": 0.05608, "MgO": 0.04030,
        }
        for sp, total_mol in species_totals.items():
            if abs(total_mol) > 1e-10:
                kpis[f"overall.{sp}_mol"] = round(total_mol, 6)
                if sp in MW:
                    kpis[f"overall.{sp}_kg"] = round(total_mol * MW[sp], 6)

        # -- Overall recovery (last product vs first feed) --
        produced = {s for u in flowsheet.units for s in u.outputs}
        consumed = {s for u in flowsheet.units for s in u.inputs}
        feeds = [s for s in flowsheet.streams if s.name not in produced]
        products = [n for n in produced if n not in consumed]

        if feeds and products:
            total_feed = sum(states[f.name].flow_mol for f in feeds if f.name in states)
            total_prod = sum(states[p].flow_mol for p in products if p in states)
            if total_feed > 0:
                kpis["overall.recovery"] = round(min(total_prod / total_feed, 1.0), 4)

        # Call Cost and LCA proxy estimators
        try:
            from ..cost.tea import estimate_opex_usd
            from ..cost.lca import estimate_co2e
            kpis["overall.opex_USD"] = estimate_opex_usd(flowsheet, states)
            kpis["overall.lca_kg_CO2e"] = estimate_co2e(flowsheet, states)
        except Exception as e:
            _log.warning(f"Failed to compute TEA/LCA metrics: {e}")

        return kpis


# ---------------------------------------------------------------------------
# Convenience function for MCP / scripts
# ---------------------------------------------------------------------------
def run_idaes(flowsheet: Flowsheet, database: str = "SUPRCRT - BL") -> Dict[str, Any]:
    """One-line entry point: build + solve + return results dict."""
    builder = IDAESFlowsheetBuilder(database_name=database)
    return builder.build_and_solve(flowsheet)


def _safe_name(name: str) -> str:
    """Sanitise a stream name for use as a Pyomo attribute."""
    return name.replace(" ", "_").replace("-", "_").replace(".", "_")
