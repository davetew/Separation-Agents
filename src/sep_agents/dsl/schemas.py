
from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import networkx as nx

Phase = Literal["solid", "liquid", "gas"]

UNIT_PARAM_SPEC = {
    "mill": {"req": ["fineness_factor"], "opt": ["E_specific_kWhpt", "media_type"]},
    "cyclone": {"req": ["d50c_um"], "opt": ["sharpness_alpha","pressure_kPa"]},
    "lims": {"req": ["magnetic_recovery"], "opt": ["field_T"]},
    "flotation_bank": {"req": ["k_s_1ps","R_inf"], "opt": ["air_rate_m3m2s","froth_recovery","stages"]},
    # ... add others as you use them
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
    params: Dict[str, float] = Field(default_factory=dict)
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)

    @field_validator("params")
    def check_params(cls, v, values):
        unit_type = values.get("type")
        if unit_type not in UNIT_PARAM_SPEC:
            raise ValueError(f"Unknown unit type: {unit_type}")
        req_params = UNIT_PARAM_SPEC[unit_type]["req"]
        opt_params = UNIT_PARAM_SPEC[unit_type]["opt"]
        for p in req_params:
            if p not in v:
                raise ValueError(f"Missing required parameter '{p}' for unit type '{unit_type}'")
        for p in v:
            if p not in req_params and p not in opt_params:
                raise ValueError(f"Unknown parameter '{p}' for unit type '{unit_type}'")
        return v

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
