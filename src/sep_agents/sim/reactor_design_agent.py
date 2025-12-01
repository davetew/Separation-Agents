"""
Reactor Design Agent
====================

This agent is responsible for defining the chemical system, initial state, and reactor parameters.
It wraps the functionality of `GeoH2.equilibrium`.
"""

import reaktoro as rkt
from GeoH2.equilibrium import defineInitialState, equilibrium, defineSystem
from typing import Union, Dict, Any

class ReactorDesignAgent:
    """
    Agent for designing chemical reactors and defining thermodynamic states.
    """

    def __init__(self, database_name: str = "SUPRCRT - BL"):
        """
        Initialize the ReactorDesignAgent.

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
                If float, assumes Fayalite fraction in Olivine.
                If dict, specifies moles/mass of each mineral.
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

    def calculate_equilibrium(self, state: rkt.ChemicalState, constraint: str = "TP") -> rkt.ChemicalState:
        """
        Calculate the equilibrium state for a given initial state.

        Args:
            state (rkt.ChemicalState): The initial chemical state.
            constraint (str): The constraint for the equilibrium calculation (e.g., 'TP', 'PH', 'TV').

        Returns:
            rkt.ChemicalState: The equilibrium state.
        """
        return equilibrium(system=self.system, state=state, constraint=constraint)

    def get_system(self):
        """Return the underlying Reaktoro ChemicalSystem."""
        return self.system
