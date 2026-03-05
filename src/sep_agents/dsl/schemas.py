
from __future__ import annotations
from typing import Any, List, Dict, Optional, Literal, Tuple
from pydantic import BaseModel, Field, field_validator
import networkx as nx

Phase = Literal["solid", "liquid", "gas"]

UNIT_PARAM_SPEC = {
    "mill": {"req": ["fineness_factor"], "opt": ["E_specific_kWhpt", "media_type"]},
    "cyclone": {"req": ["d50c_um"], "opt": ["sharpness_alpha", "pressure_kPa"]},
    "lims": {"req": ["magnetic_recovery"], "opt": ["field_T"]},
    "flotation_bank": {"req": ["k_s_1ps", "R_inf"], "opt": ["air_rate_m3m2s", "froth_recovery", "stages"]},
    "leach_reactor": {"req": ["residence_time_s", "T_C"], "opt": ["tank_volume_m3", "agitation_power_kW", "p_bar"]},
    "precipitator": {"req": ["residence_time_s", "reagent_dosage_gpl"], "opt": ["T_C", "target_pH", "p_bar", "reagent_name"]},
    "mixer": {"req": [], "opt": []},
    "solvent_extraction": {
        "req": ["distribution_coeff", "organic_to_aqueous_ratio"],
        "opt": ["stages", "T_C", "extractant", "diluent"],
    },
    "ion_exchange": {
        "req": ["selectivity_coeff", "bed_volume_m3"],
        "opt": ["resin_type", "flow_rate_BV_per_hr", "T_C"],
    },
    "crystallizer": {
        "req": ["T_C", "residence_time_s"],
        "opt": ["cooling_rate_K_per_s", "seed_loading_gpl", "reagent_dosage_gpl", "reagent_name"],
    },
    "thickener": {
        "req": [],
        "opt": ["recovery", "underflow_solids_frac", "overflow_clarity"],
    },
    # ── Geological process units ──────────────────────────────────────
    "heat_exchanger": {
        "req": ["U_Wm2K", "area_m2"],
        "opt": ["dT_approach_K", "type"],  # type: "counter" or "parallel"
    },
    "pump": {
        "req": ["head_m", "efficiency"],
        "opt": ["power_kW", "type"],  # type: "centrifugal" or "positive_displacement"
    },
    "stoichiometric_reactor": {
        "req": ["reactions"],
        "opt": ["residence_time_s", "T_C", "T_K", "p_bar", "P_Pa",
                "tank_volume_m3", "agitation_power_kW"],
    },
    "equilibrium_reactor": {
        "req": ["residence_time_s", "T_C"],
        "opt": ["tank_volume_m3", "agitation_power_kW", "p_bar",
                "reagent_dosage_gpl", "reagent_name"],
    },
    "separator": {
        "req": [],
        "opt": ["recovery", "split_fraction"],
    },
}
class PSD(BaseModel):
    bins_um: List[float] = Field(..., description="Upper size in microns for each bin")
    mass_frac: List[float] = Field(..., description="Mass fraction per bin (sum ~ 1.0)")

    @field_validator("mass_frac") # Updated to field_validator from validator by DET due to Pydantic v2 changes
    def _sum_to_one(cls, v):
        s = sum(v)
        if not (0.99 <= s <= 1.01):
            raise ValueError(f"PSD mass fractions sum to {s:.3f}, expected ~1.0")
        return v

class LiberationMatrix(BaseModel):
    # rows = PSD bins, cols = minerals; values are liberated mass fraction per mineral per bin
    minerals: List[str]
    matrix: List[List[float]]

class Stream(BaseModel):
    name: str
    phase: Phase
    temperature_K: float = 298.15
    pressure_Pa: float = 101325.0
    composition_wt: Dict[str, float] = Field(default_factory=dict)  # element or mineral basis
    psd: Optional[PSD] = None
    liberation: Optional[LiberationMatrix] = None
    pH: Optional[float] = None
    Eh_mV: Optional[float] = None
    solids_wtfrac: Optional[float] = None

class UnitOp(BaseModel):
    id: str
    type: str  # e.g., 'mill', 'cyclone', 'lims', 'flotation', 'leach', 'thickener'
    params: Dict[str, Any] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)

    # -- GDP superstructure annotations --
    optional: bool = False
    """If True, this unit may be bypassed (inlet passes straight to outlet)."""
    alternatives: List[str] = Field(default_factory=list)
    """Mutually exclusive group name(s).  Units sharing the same alternative
    group are placed in an XOR disjunction — exactly one is active."""
    stage_range: Optional[Tuple[int, int]] = None
    """For multi-stage units (e.g. SX cascades): (min_stages, max_stages).
    Each candidate stage count becomes a separate disjunct."""

    from pydantic import model_validator

    @model_validator(mode="after")
    def check_params(self):
        unit_type = self.type
        if unit_type not in UNIT_PARAM_SPEC:
            import warnings
            warnings.warn(f"Unknown unit type '{unit_type}'; skipping param validation")
            return self
        spec = UNIT_PARAM_SPEC[unit_type]
        req_params = spec["req"]
        opt_params = spec["opt"]
        for p in req_params:
            if p not in self.params:
                raise ValueError(f"Missing required parameter '{p}' for unit type '{unit_type}'")
        for p in self.params:
            if p not in req_params and p not in opt_params:
                raise ValueError(f"Unknown parameter '{p}' for unit type '{unit_type}'")
        return self


class Flowsheet(BaseModel):
    name: str
    units: List[UnitOp]
    streams: List[Stream]

    def as_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        for u in self.units:
            g.add_node(u.id, type=u.type, params=u.params)
            for i in u.inputs:
                g.add_edge(i, u.id)
            for o in u.outputs:
                g.add_edge(u.id, o)
        return g
    
    def validate_graph(self):
        stream_names = {s.name for s in self.streams}
        for u in self.units:
            for s in u.inputs + u.outputs:
                if s not in stream_names:
                    raise ValueError(f"Unit {u.id} references unknown stream '{s}'")
        # basic source/sink presence
        used_as_input = {s for u in self.units for s in u.inputs}
        used_as_output = {s for u in self.units for s in u.outputs}
        feeds = [s for s in stream_names if s not in used_as_output]
        prods = [s for s in stream_names if s not in used_as_input]
        if not feeds:  raise ValueError("No feed streams (not produced by any unit).")
        if not prods:  raise ValueError("No product/sink streams (not consumed by any unit).")
        return True


# ---------------------------------------------------------------------------
# GDP Superstructure models
# ---------------------------------------------------------------------------

Objective = Literal[
    "minimize_opex",
    "maximize_recovery",
    "minimize_lca",
    "maximize_value_per_kg_ore",
]


class DisjunctionDef(BaseModel):
    """A named group of mutually exclusive unit choices.

    Exactly one of the listed ``unit_ids`` will be active in the
    optimized flowsheet.  This maps to a Pyomo ``Disjunction``.
    """
    name: str
    unit_ids: List[str] = Field(..., min_length=2)
    description: str = ""


class Superstructure(BaseModel):
    """A process superstructure for GDP topology optimization.

    The ``base_flowsheet`` contains the **superset** of all possible units
    (both required and optional).  The GDP solver determines which subset
    to activate.
    """
    name: str
    base_flowsheet: Flowsheet
    disjunctions: List[DisjunctionDef] = Field(default_factory=list)
    fixed_units: List[str] = Field(default_factory=list)
    """Unit IDs that must always be active (never bypassed)."""
    objective: Objective = "minimize_opex"
    continuous_bounds: Dict[str, Tuple[float, float]] = Field(default_factory=dict)
    """Optional bounds on continuous params for BoTorch, e.g.
    ``{"sx_1.organic_to_aqueous_ratio": (0.5, 3.0)}``."""
    description: str = ""
