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
        return defineInitialState(
            T_C=T_C,
            p_bar=p_bar,
            mineralSpec=mineral_spec,
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
        
        for val in param_values:
            # Update conditions with the current sweep value
            current_conditions = initial_conditions.copy()
            current_conditions[param_name] = val
            
            # Define state
            try:
                state = self.define_state(**current_conditions)
                
                # Solve equilibrium
                eq_state = self.solve(state, constraint)
                
                # Extract results
                props = rkt.ChemicalProps(eq_state)
                aprops = rkt.AqueousProps(eq_state)
                
                res = {
                    param_name: val,
                    "Temperature (C)": props.temperature() - 273.15,
                    "Pressure (bar)": props.pressure() / 1e5,
                    "pH": aprops.pH(),
                    "Eh (V)": aprops.Eh(),
                    "Volume (m3)": props.volume(),
                    "Enthalpy (J)": props.enthalpy()
                }
                
                # Add species amounts (optional, can be verbose)
                # For now, maybe just add specific important ones if needed
                # or we can add all species > threshold
                
                results.append(res)
                
            except Exception as e:
                print(f"Error at {param_name}={val}: {e}")
                
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
