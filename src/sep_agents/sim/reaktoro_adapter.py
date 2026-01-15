
"""Reaktoro adapter stub.

Implement leach/speciation steps by calling Reaktoro to compute equilibria.
"""
from typing import Dict
from ..dsl.schemas import Stream, Flowsheet
from .equilibrium_agent import EquilibriumAgent
import logging

_log = logging.getLogger(__name__)

def run_reaktoro(stream: Stream) -> Stream:
    """
    Run Reaktoro equilibrium calculation on the stream and update its properties.
    """
    agent = EquilibriumAgent() # Defaults to SUPRCRT - BL
    
    # Map Stream to Reaktoro state
    # We need to map composition_wt (which might be element or mineral basis) to something Reaktoro understands
    # For now, let's assume composition_wt keys map to species/minerals in the database
    
    # Simplified mapping for initial testing:
    # We'll just take T and P. Composition mapping is complex without a robust parser.
    # But we can try to pass what we have.
    
    try:
        # TODO: Better handling of composition units. 
        # Stream.composition_wt is a dict of name -> value.
        # define_state expects 'mineral_spec'.
        
        state = agent.define_state(
            T_C=stream.temperature_K - 273.15,
            p_bar=stream.pressure_Pa / 1e5,
            mineral_spec=stream.composition_wt, # Assumes keys are valid syntax for Reaktoro
            # Defaulting other params for now
            w_r=1.0, 
            c_r=0.0
        )
        
        eq_state = agent.solve(state)
        
        # Extract results
        import reaktoro as rkt
        aprops = rkt.AqueousProps(eq_state)
        
        stream.pH = float(aprops.pH())
        stream.Eh_mV = float(aprops.Eh()) * 1000.0
        
        _log.info(f"Reaktoro solve successful for {stream.name}: pH={stream.pH:.2f}")

    except Exception as e:
        _log.error(f"Reaktoro solve failed for stream {stream.name}: {e}")
        # We return the stream unmodified in case of failure, maybe adding a flag?
        
    return stream
