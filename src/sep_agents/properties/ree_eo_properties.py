"""
REE Equation-Oriented Property Package for IDAES
=================================================

A lightweight, fully-differentiable IDAES property package for REE
hydrometallurgy modelling.  All thermodynamic properties are expressed as
native Pyomo expressions so that IPOPT can compute analytic (AD) gradients
through the entire flowsheet.

Design choices
--------------
* **State variables**: ``temperature``, ``pressure``, ``flow_mol_comp[j]``
  (component molar flows) — the "FcTP" formulation natural for mass-balance
  equations.
* **Phases**: ``Liq`` (aqueous), ``Org`` (organic extract), ``Sol`` (solid
  precipitate).  A given stream will typically have non-zero flow in just one
  phase; multi-phase splits are handled by unit model constraints, not by the
  property block itself.
* **Properties**: molecular weight, molar density (polynomial in T),
  molar enthalpy (Cp · T), and activity-coefficient placeholder.

This module intentionally avoids calling Reaktoro so that the property block
remains a pure algebraic system compatible with any Pyomo solver (IPOPT, CBC,
GLPK, etc.).
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

from pyomo.environ import (
    ConcreteModel,
    Constraint,
    Expression,
    Param,
    Set,
    Suffix,
    Var,
    NonNegativeReals,
    PositiveReals,
    log,
    units as pyunits,
    value,
)
from pyomo.common.config import ConfigValue

from idaes.core import (
    Component as IdaesComponent,
    Phase,
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    MaterialFlowBasis,
)
from idaes.core.util.initialization import fix_state_vars

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Constants & correlations
# ═══════════════════════════════════════════════════════════════════════

# Molar masses [kg/mol]
_MOLAR_MASS = {
    "H2O":  0.01802,
    "HCl":  0.03646,
    "NaOH": 0.04000,
    "H2C2O4": 0.09003,   # oxalic acid
    "La":   0.13891,
    "Ce":   0.14012,
    "Pr":   0.14091,
    "Nd":   0.14424,
    "Sm":   0.15036,
    "Y":    0.08891,
    "Dy":   0.16250,
    "Fe":   0.05585,
    "Al":   0.02698,
}

# Heat capacity [J/mol/K] — simplified single-value Cp
_CP_MOL = {
    "H2O":  75.3,
    "HCl":  29.1,
    "NaOH": 59.5,
    "H2C2O4": 100.0,
    "La":   27.1,
    "Ce":   26.9,
    "Pr":   27.2,
    "Nd":   27.4,
    "Sm":   29.5,
    "Y":    26.5,
    "Dy":   27.7,
    "Fe":   25.1,
    "Al":   24.2,
}

# Default components for each preset
_PRESETS = {
    "lree": ["H2O", "HCl", "La", "Ce", "Nd", "Pr"],
    "hree": ["H2O", "HCl", "Y", "Dy", "Sm"],
    "full": ["H2O", "HCl", "La", "Ce", "Pr", "Nd", "Sm", "Y", "Dy"],
    "lree_process": ["H2O", "HCl", "NaOH", "H2C2O4", "La", "Ce", "Nd", "Pr"],
}


# ═══════════════════════════════════════════════════════════════════════
# Parameter Block
# ═══════════════════════════════════════════════════════════════════════

@declare_process_block_class("REEEOParameterBlock")
class REEEOParameterBlockData(PhysicalParameterBlock):
    """Physical parameter block for REE hydrometallurgy (equation-oriented).

    Configuration
    -------------
    preset : str
        One of ``"lree"``, ``"hree"``, ``"full"``, ``"lree_process"``.
        Determines the default component list.
    components : list[str], optional
        Explicit component list (overrides ``preset``).
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    CONFIG.declare("preset", ConfigValue(
        default="lree",
        domain=str,
        description="Component preset",
    ))
    CONFIG.declare("components", ConfigValue(
        default=None,
        description="Explicit component list (overrides preset)",
    ))

    def build(self):
        super().build()

        self._state_block_class = REEEOStateBlock

        # ── Resolve component list ──────────────────────────────────
        comp_names = self.config.components
        if comp_names is None:
            comp_names = _PRESETS.get(self.config.preset, _PRESETS["lree"])
        comp_names = list(comp_names)

        # ── Phase definitions ───────────────────────────────────────
        self.Liq = Phase()
        self.Org = Phase()
        self.Sol = Phase()

        # ── Component objects ───────────────────────────────────────
        for c in comp_names:
            self.add_component(c, IdaesComponent())

        # ── Look-up parameters (indexed by component) ──────────────
        self.mw = Param(
            self.component_list,
            initialize={c: _MOLAR_MASS.get(c, 0.100) for c in comp_names},
            units=pyunits.kg / pyunits.mol,
            doc="Molar mass",
        )
        self.cp_mol = Param(
            self.component_list,
            initialize={c: _CP_MOL.get(c, 30.0) for c in comp_names},
            units=pyunits.J / pyunits.mol / pyunits.K,
            doc="Molar heat capacity (isobaric)",
        )

        # Reference temperature for enthalpy datum
        self.temperature_ref = Param(
            initialize=298.15,
            units=pyunits.K,
            doc="Reference temperature for h = Cp*(T - T_ref)",
        )

        # Water density polynomial: ρ [mol/m³] ≈ a₀ + a₁*(T-273) + a₂*(T-273)²
        # Fit to liquid water 0-100°C range
        self.rho_water_a0 = Param(initialize=55400.0, units=pyunits.mol / pyunits.m**3)
        self.rho_water_a1 = Param(initialize=-7.6, units=pyunits.mol / pyunits.m**3 / pyunits.K)
        self.rho_water_a2 = Param(initialize=-0.0036, units=pyunits.mol / pyunits.m**3 / pyunits.K**2)

    @classmethod
    def define_metadata(cls, obj):
        obj.add_properties({
            "flow_mol": {"method": None},
            "flow_mol_comp": {"method": None},
            "flow_mol_phase_comp": {"method": None},
            "temperature": {"method": None},
            "pressure": {"method": None},
            "mw_comp": {"method": None},
            "enth_mol": {"method": None},
            "dens_mol_phase": {"method": None},
        })
        obj.add_default_units({
            "time": pyunits.s,
            "length": pyunits.m,
            "mass": pyunits.kg,
            "amount": pyunits.mol,
            "temperature": pyunits.K,
        })


# ═══════════════════════════════════════════════════════════════════════
# State Block
# ═══════════════════════════════════════════════════════════════════════

@declare_process_block_class("REEEOStateBlock", block_class=StateBlock)
class REEEOStateBlockData(StateBlockData):
    """State block for REE hydrometallurgy — equation-oriented.

    State variables
    ---------------
    temperature : Var [K]
    pressure : Var [Pa]
    flow_mol_comp : Var [mol/s], indexed by component

    Derived properties (Pyomo Expressions)
    ----------------------------------------
    flow_mol : total molar flow
    mole_frac_comp : mole fractions
    enth_mol : molar enthalpy [J/mol]
    flow_mass_comp : mass flow per component [kg/s]
    flow_mass : total mass flow [kg/s]
    dens_mol_phase : molar density (liquid) [mol/m³]
    """

    def build(self):
        super().build()

        p = self.params  # shorthand

        # ── State variables ─────────────────────────────────────────
        self.temperature = Var(
            initialize=298.15,
            bounds=(273.15, 573.15),
            domain=PositiveReals,
            doc="Temperature [K]",
            units=pyunits.K,
        )
        self.pressure = Var(
            initialize=101325.0,
            bounds=(1e4, 1e7),
            domain=PositiveReals,
            doc="Pressure [Pa]",
            units=pyunits.Pa,
        )
        self.flow_mol_comp = Var(
            p.component_list,
            initialize=1.0,
            bounds=(0, None),
            domain=NonNegativeReals,
            doc="Component molar flow [mol/s]",
            units=pyunits.mol / pyunits.s,
        )

        # ── Derived: total flow ─────────────────────────────────────
        self.flow_mol = Expression(
            expr=sum(self.flow_mol_comp[j] for j in p.component_list),
            doc="Total molar flow [mol/s]",
        )

        # ── Derived: mole fractions ─────────────────────────────────
        def _mole_frac_rule(blk, j):
            return blk.flow_mol_comp[j] / (blk.flow_mol + 1e-12)
        self.mole_frac_comp = Expression(
            p.component_list,
            rule=_mole_frac_rule,
            doc="Mole fraction",
        )

        # ── Derived: mass flows ─────────────────────────────────────
        def _flow_mass_comp_rule(blk, j):
            return blk.flow_mol_comp[j] * p.mw[j]
        self.flow_mass_comp = Expression(
            p.component_list,
            rule=_flow_mass_comp_rule,
            doc="Component mass flow [kg/s]",
        )

        self.flow_mass = Expression(
            expr=sum(self.flow_mass_comp[j] for j in p.component_list),
            doc="Total mass flow [kg/s]",
        )

        # ── Derived: molar enthalpy (simple Cp model) ──────────────
        # h_mix = sum_j x_j * Cp_j * (T - T_ref)
        self.enth_mol = Expression(
            expr=sum(
                self.mole_frac_comp[j] * p.cp_mol[j]
                * (self.temperature - p.temperature_ref)
                for j in p.component_list
            ),
            doc="Molar enthalpy [J/mol]",
        )

        # ── Derived: total enthalpy flow ────────────────────────────
        self.enth_flow = Expression(
            expr=self.flow_mol * self.enth_mol,
            doc="Enthalpy flow [J/s]",
        )

        # ── Derived: liquid molar density (T-dependent) ────────────
        def _dens_mol_phase_rule(blk, ph):
            dT = blk.temperature - 273.15 * pyunits.K
            return (p.rho_water_a0
                    + p.rho_water_a1 * dT
                    + p.rho_water_a2 * dT**2)
        self.dens_mol_phase = Expression(
            p.phase_list,
            rule=_dens_mol_phase_rule,
            doc="Molar density [mol/m³]",
        )

        # ── Phase-component flow (trivial: all in Liq by default) ──
        # Unit models will manipulate phase assignment via constraints
        def _flow_mol_phase_comp_rule(blk, ph, j):
            # By default everything goes to Liq
            if ph == "Liq":
                return blk.flow_mol_comp[j]
            else:
                return 0.0
        self.flow_mol_phase_comp = Expression(
            p.phase_list, p.component_list,
            rule=_flow_mol_phase_comp_rule,
            doc="Phase-component molar flow [mol/s]",
        )

    # ── IDAES interface methods ─────────────────────────────────────

    def default_material_balance_type(self):
        return MaterialBalanceType.componentTotal

    def default_energy_balance_type(self):
        return EnergyBalanceType.enthalpyTotal

    def get_material_flow_basis(self):
        return MaterialFlowBasis.molar

    def get_material_flow_terms(self, p, j):
        """Material flow for phase p, component j."""
        if p == "Liq":
            return self.flow_mol_comp[j]
        return 0.0

    def get_enthalpy_flow_terms(self, p):
        """Enthalpy flow for phase p."""
        if p == "Liq":
            return self.enth_flow
        return 0.0

    def get_material_density_terms(self, p, j):
        """Density term for material balance."""
        if p == "Liq":
            return self.dens_mol_phase[p] * self.mole_frac_comp[j]
        return 0.0

    def get_energy_density_terms(self, p):
        """Energy density for energy balance."""
        if p == "Liq":
            return self.dens_mol_phase[p] * self.enth_mol
        return 0.0

    def define_state_vars(self):
        return {
            "flow_mol_comp": self.flow_mol_comp,
            "temperature": self.temperature,
            "pressure": self.pressure,
        }
