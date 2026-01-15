"""
Equilibrium Agent
=================

This agent is responsible for performing equilibrium calculations and parameter sweeps.
It wraps the functionality of `GeoH2.equilibrium`.
"""

import reaktoro as rkt
import pandas as pd
import numpy as np
from GeoH2.equilibrium import defineInitialState, equilibrium, defineSystem
from GeoH2 import Q_
from typing import Union, Dict, Any, List

# Database-specific species synonyms
# Key: Substring to match in database name (lowercase)
# Value: Dict of {GenericName: DatabaseSpecificName}
SPECIES_MAPPING = {
    "supcrt": {"H2O": "H2O(aq)", "CO2": "CO2(aq)"},
    "suprcrt": {"H2O": "H2O(aq)", "CO2": "CO2(aq)", "CaCO3": "Calcite"}, # Handle possible typo in stored database names
    "phreeqc": {"H2O": "H2O"}, # Phreeqc uses H2O
    "pitzer": {"H2O": "H2O"},
}

class EquilibriumAgent:
    """
    Agent for performing equilibrium assessments of chemical systems.
    """

    def __init__(self, database_name: str = "SUPRCRT - BL"):
        """
        Initialize the EquilibriumAgent.

        Args:
            database_name (str): The name of the thermodynamic database to use.
                                 Options: 'PHREEQC - LLNL', 'SUPRCRT - 16', 'SUPRCRT - BL'
        """
        self.database_name = database_name
        self.system, self.minerals, self.solution, self.gases = defineSystem(database_name)

    def define_state(self, 
                     T_C: float, 
                     p_bar: float, 
                     mineral_spec: Union[float, Dict[str, float]], 
                     w_r: float = 1.0, 
                     c_r: float = 0.0,
                     salinity_g_kg: float = 0.0,
                     pH: float = None,
                     charge: float = None,
                     mineral_spec_type: str = "mol") -> rkt.ChemicalState:
        """
        Define the initial chemical state of the system.

        Args:
            T_C (float): Temperature in Celsius.
            p_bar (float): Pressure in bar.
            mineral_spec (Union[float, Dict[str, float]]): Mineral composition. 
            w_r (float): Water to rock mass ratio.
            c_r (float): CO2 to rock mass ratio.
            salinity_g_kg (float): Salinity in g/kg.
            pH (float): Initial pH (optional).
            charge (float): Initial charge imbalance (optional).
            mineral_spec_type (str): "mol" or "mass".

        Returns:
            rkt.ChemicalState: The defined chemical state.
        """
        # Determine mapping based on database name
        db_key = next((k for k in SPECIES_MAPPING if k in self.database_name.lower()), None)
        mapping = SPECIES_MAPPING.get(db_key, {})

        # Remap composition keys
        spec = mineral_spec
        if isinstance(mineral_spec, dict):
             spec = {mapping.get(k, k): v for k, v in mineral_spec.items()}

        return defineInitialState(
            T_C=T_C,
            p_bar=p_bar,
            mineralSpec=spec,
            w_r=w_r,
            c_r=c_r,
            salinity_g_kg=salinity_g_kg,
            pH=pH,
            charge=charge,
            system=self.system,
            minerals=self.minerals,
            mineralSpecType=mineral_spec_type
        )

    def solve(self, state: rkt.ChemicalState, constraint: str = "TP") -> rkt.ChemicalState:
        """
        Calculate the equilibrium state for a given initial state.

        Args:
            state (rkt.ChemicalState): The initial chemical state.
            constraint (str): The constraint for the equilibrium calculation (e.g., 'TP', 'PH', 'TV').

        Returns:
            rkt.ChemicalState: The equilibrium state.
        """
        return equilibrium(system=self.system, state=state, constraint=constraint)

    def sweep(self, 
              initial_conditions: Dict[str, Any], 
              param_name: str, 
              param_values: List[float], 
              constraint: str = "TP") -> pd.DataFrame:
        """
        Perform an equilibrium parameter sweep.

        Args:
            initial_conditions (Dict[str, Any]): Base conditions for `define_state`.
            param_name (str): The parameter to vary (e.g., 'T_C', 'p_bar', 'w_r').
            param_values (List[float]): List of values for the parameter.
            constraint (str): Equilibrium constraint.

        Returns:
            pd.DataFrame: Results of the sweep.
        """
        results = []
        
        print(f"Running sweep for {param_name} over {len(param_values)} points...")
        
        prev_state = None
        
        for val in param_values:
            # Update conditions with the current sweep value
            current_conditions = initial_conditions.copy()
            current_conditions[param_name] = val
            
            res = {param_name: val}
            
            try:
                # Define initial state
                # Strategy: If we have a previous successful state and we are just changing T or P, 
                # we can try to use the previous state as a guess (Warm Start).
                # However, ensuring mass balance consistency is tricky if we don't know exactly what changed.
                # For safety, we always define a fresh state first (guarantees correct definition)
                # But to enable warm restart, we might need to rely on Reaktoro's solver capabilities 
                # or manually set properties of the fresh state to match previous?
                
                # Simple approach: Always fresh definition first
                state = self.define_state(**current_conditions)
                
                # If we have a previous state and are sweeping T/P, we *could* try to use it,
                # but 'equilibrium()' takes 'state' as the definition of the problem.
                # If we pass 'prev_state', we must ensure it has the correct T/P/Composition for THIS step.
                
                # Let's stick to fresh definition for correctness, but catch failures.
                
                # Solve equilibrium
                eq_state = self.solve(state, constraint)
                
                if eq_state is None:
                    raise RuntimeError("Equilibrium solver returned None (convergence failure)")

                # Extract results
                props = rkt.ChemicalProps(eq_state)
                aprops = rkt.AqueousProps(eq_state)
                
                res.update({
                    "Temperature (C)": float(props.temperature() - 273.15),
                    "Pressure (bar)": float(props.pressure() / 1e5),
                    "pH": float(aprops.pH()),
                    "Eh (V)": float(aprops.Eh()),
                    "Volume (m3)": float(props.volume()),
                    "Enthalpy (J)": float(props.enthalpy()),
                    "status": "converged"
                })
                
                # Store successful state for potential future warm start usage
                prev_state = eq_state
                
            except Exception as e:
                # Log error but continue
                print(f"Warning: Equilibrium calculation failed at {param_name}={val}: {e}")
                res.update({
                    "Temperature (C)": None,
                    "Pressure (bar)": None,
                    "pH": None,
                    "Eh (V)": None,
                    "Volume (m3)": None,
                    "Enthalpy (J)": None,
                    "status": "failed",
                    "error": str(e)
                })
                
            results.append(res)
                
        return pd.DataFrame(results)

    def analyze_results(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the results of a sweep.

        Args:
            df (pd.DataFrame): Sweep results.

        Returns:
            Dict[str, Any]: Summary statistics.
        """
        return {
            "min_pH": df["pH"].min(),
            "max_pH": df["pH"].max(),
            "mean_pH": df["pH"].mean()
        }
