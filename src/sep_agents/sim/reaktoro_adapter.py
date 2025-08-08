
"""Reaktoro adapter stub.

Implement leach/speciation steps by calling Reaktoro to compute equilibria.
"""
from typing import Dict
from ..dsl.schemas import Stream, Flowsheet

def run_reaktoro(stream: Stream) -> Stream:
    # TODO: call Reaktoro equilibrium/speciation and update stream
    return stream
