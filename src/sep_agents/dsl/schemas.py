
from __future__ import annotations
from typing import List, Dict, Optional, Literal
from pydantic import BaseModel, Field, field_validator
import networkx as nx

Phase = Literal["solid", "liquid", "gas"]

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
