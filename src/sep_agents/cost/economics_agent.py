"""
Economics Agent
===============

This agent is responsible for evaluating CAPEX, OPEX, and Levelized Cost of Product (LCOP).
It wraps the functionality of `GeoH2.economics`.
"""

import pandas as pd
import numpy as np
from GeoH2.economics import Levelized_Cost, default_cost_params
from GeoH2 import Q_
from typing import Dict, Any

class GeneralizedLevelizedCost(Levelized_Cost):
    """
    Generalized version of Levelized_Cost that supports any primary product.
    """
    def __init__(self, cost_params: dict, primary_product: str = "H2"):
        super().__init__(cost_params)
        self.primary_product = primary_product

    @property
    def M_Product(self):
        # Calculate and return the target mass of Product to be produced annually in tonnes
        # Assuming P_H2 in params is actually P_Product (Power/Energy equivalent) or we need a mass target.
        # GeoH2 uses P_H2 (Power) and LHV to get Mass. 
        # For metals, we might just want a Mass target directly.
        # If 'M_target_tonnes' is in params, use it. Else derive from P_H2 if product is H2.
        
        if 'M_target_tonnes' in self.cost_params:
            return Q_(self.cost_params['M_target_tonnes'], 'tonnes')
        
        if self.primary_product == "H2":
            return super().M_H2
            
        # Fallback: Assume P_H2 param represents a generic production scale proxy if no mass target
        # This is a bit hacky, but preserves structure. 
        # Better: User should supply 'M_target_tonnes' for non-H2.
        return Q_(1000, 'tonnes') # Default placeholder

    @property
    def Y_Product_actual(self):
        # Actual yield of the primary product
        return self.cost_params['mat_yield'].get(self.primary_product, Q_(0, 'kg/tonne')) * self.reaction_extent

    @property
    def C_capital_MProduct(self):
        # Generalized CAPEX calculation
        # Uses Y_Product_actual instead of Y_H2
        α = self.cost_params['α']
        M_rock_batch_ref = self.cost_params['M_rock_batch_ref']
        
        # Avoid division by zero if yield is 0
        if self.Y_Product_actual.magnitude == 0:
            return Q_(float('inf'), 'USD/kg')

        return (self.cost_params["CAPEX_ref"]/self.M_Product**(1-α)*(1/M_rock_batch_ref/self.Y_Product_actual/self.N_b/self.reaction_extent)**α).to('USD/kg')

    @property
    def C_labor_MProduct(self):
        if self.Y_Product_actual.magnitude == 0: return Q_(float('inf'), 'USD/kg')
        return (self.cost_params['c_wage'] / self.cost_params['π_labor'] / self.Y_Product_actual).to('USD/kg')

    @property
    def C_energy_MProduct(self):
        if self.Y_Product_actual.magnitude == 0: return Q_(float('inf'), 'USD/kg')
        return (self.cost_params['e_rock']*self.cost_params['c_energy'] / self.Y_Product_actual).to('USD/kg')

    @property
    def C_maint_MProduct(self):
        if self.Y_Product_actual.magnitude == 0: return Q_(float('inf'), 'USD/kg')
        return self.cost_params['c_maint'] / self.Y_Product_actual

    @property
    def C_raw_material_MProduct(self):
        C_raw_material = 0
        for mat, cost in self.cost_params['c_material'].items():
            C_raw_material += cost * self.cost_params['mat_usage'].get(mat, 0) # Use get to be safe
        
        if self.Y_Product_actual.magnitude == 0: return Q_(float('inf'), 'USD/kg')
        return (C_raw_material / self.Y_Product_actual).to('USD/kg')

    @property
    def C_operating_MProduct(self):
        return self.C_labor_MProduct + self.C_energy_MProduct + self.C_maint_MProduct + self.C_raw_material_MProduct

    @property
    def Revenue_MProduct(self):
        # Calculate and return the gross revenue per kg Product produced
        gross_revenue = 0
        for mat, mat_yield in self.cost_params['mat_yield'].items():
            gross_revenue += self.cost_params['mat_value'].get(mat, Q_(0, 'USD/kg')) * mat_yield
            
        if self.Y_Product_actual.magnitude == 0: return Q_(float('inf'), 'USD/kg')
        return (gross_revenue / self.Y_Product_actual).to('USD/kg')

    @property
    def LCOP(self):
        # Levelized Cost of Product
        return self.C_capital_MProduct/self.discount_factor + self.C_operating_MProduct
        
    @property
    def cost_breakdown(self):
        # Override to return generalized breakdown
        return pd.DataFrame({
            'Cost Elements': {
                'capital': self.C_capital_MProduct.magnitude,
                'labor': self.C_labor_MProduct.magnitude,
                'maint': self.C_maint_MProduct.magnitude,
                'energy': self.C_energy_MProduct.magnitude,
                'raw_material': self.C_raw_material_MProduct.magnitude
            }
        })

class EconomicsAgent:
    """
    Agent for performing economic assessments of chemical processes.
    """

    def __init__(self, cost_params: Dict[str, Any] = None, primary_product: str = "H2"):
        """
        Initialize the EconomicsAgent.

        Args:
            cost_params (Dict[str, Any]): Dictionary of cost parameters. 
                                          Defaults to `GeoH2.economics.default_cost_params`.
            primary_product (str): The primary product to calculate LCOP for (e.g., "H2", "Mg", "Fe").
        """
        self.cost_params = cost_params if cost_params else default_cost_params
        self.primary_product = primary_product
        self.model = GeneralizedLevelizedCost(self.cost_params, self.primary_product)

    def update_params(self, new_params: Dict[str, Any]):
        """
        Update the cost parameters.

        Args:
            new_params (Dict[str, Any]): New parameters to update.
        """
        self.cost_params.update(new_params)
        # Re-initialize model to ensure updates propagate
        self.model = GeneralizedLevelizedCost(self.cost_params, self.primary_product)

    def calculate_lcop(self) -> float:
        """
        Calculate the Levelized Cost of Product.

        Returns:
            float: The levelized cost in USD/kg.
        """
        return self.model.LCOP.to('USD/kg').magnitude

    def get_cost_breakdown(self) -> pd.DataFrame:
        """
        Get the breakdown of costs (CAPEX, OPEX, etc.).

        Returns:
            pd.DataFrame: Cost breakdown.
        """
        return self.model.cost_breakdown

    def get_design_summary(self) -> pd.DataFrame:
        """
        Get the design summary (reactor size, batches, etc.).

        Returns:
            pd.DataFrame: Design summary.
        """
        return self.model.design_summary
