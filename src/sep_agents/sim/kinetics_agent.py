"""
Kinetics Agent
==============

This agent is responsible for running kinetic simulations to determine reaction extents and times.
It wraps the functionality of `GeoH2.kinetics`.
"""

import pandas as pd
from GeoH2.kinetics import Kinetics
from GeoH2 import Q_
from typing import Union, Dict, Any

class KineticsAgent:
    """
    Agent for performing kinetic simulations of chemical processes.
    """

    def __init__(self, 
                 state: Any, 
                 constraint: str = "TV",
                 particle_radii: Union[Dict[str, Any], Any] = {'min': Q_(50, "um"), "max": Q_(100, "um")},
                 alpha_BET: float = 5,
                 constant_volume_specs: Dict[str, Any] = None,
                 kinetic_data_file: str = None):
        """
        Initialize the KineticsAgent.

        Args:
            state (Any): Initial chemical state (dict or rkt.ChemicalState).
            constraint (str): Simulation constraint ('TP', 'TV', etc.).
            particle_radii (Union[Dict, Q_]): Particle radii specification.
            alpha_BET (float): BET surface area factor.
            constant_volume_specs (Dict): Specifications for constant volume simulations.
            kinetic_data_file (str): Path to a custom kinetic data YAML file.
        """
        # Handle parameter mapping if state is a dict (initial_conditions)
        if isinstance(state, dict):
            if "mineral_spec" in state:
                state["mineralSpec"] = state.pop("mineral_spec")
            # Add other mappings if needed (e.g., T_C, p_bar are already matching GeoH2/ReactorDesignAgent?)
            # ReactorDesignAgent uses T_C, p_bar. GeoH2 uses T_C, p_bar. So those are fine.

        kwargs = {
            "state": state,
            "constraint": constraint,
            "particle_radii": particle_radii,
            "alpha_BET": alpha_BET
        }
        if constant_volume_specs is not None:
            kwargs["constant_volume_specs"] = constant_volume_specs
        if kinetic_data_file is not None:
            kwargs["kinetic_data_file"] = kinetic_data_file
            
        self.kinetics_model = Kinetics(**kwargs)

    def run_simulation(self, 
                       duration: str = "1 day", 
                       n_points: int = 100, 
                       stopping_criteria: float = 0.99) -> pd.DataFrame:
        """
        Run a kinetic simulation.

        Args:
            duration (str): Duration of the simulation (e.g., "1 day", "10 hours").
            n_points (int): Number of time points.
            stopping_criteria (float): Fraction of equilibrium to stop at.

        Returns:
            pd.DataFrame: Simulation results.
        """
        return self.kinetics_model.simulation(
            duration=Q_(duration),
            n_points=n_points,
            stopping_criteria=stopping_criteria,
            save_sim_data_file=False # Don't save to file by default for agent operations
        )

    def get_performance_metrics(self, state: Any) -> Dict[str, float]:
        """
        Get performance metrics for a given state.
        
        Args:
            state: The chemical state to evaluate.
            
        Returns:
            Dict[str, float]: Performance metrics (e.g., conversion, yield).
        """
        # This wraps the performance method from the underlying Kinetics class
        # Note: The GeoH2 Kinetics class has a performance method, but it might need the state to be updated first
        # For now, we'll assume the user extracts data from the simulation dataframe
        pass
