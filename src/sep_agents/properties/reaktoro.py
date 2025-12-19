
"""
Reaktoro Property Package for IDAES.

This package allows IDAES flowsheets to forward property calculations to Reaktoro.
It relies on the 'GeoH2' or 'reaktoro' libraries being installed.

Note: This implementation currently supports "call-back" style property evaluation
suitable for simulation logic or initialization, rather than full equation-oriented
optimization (which would require external functions with derivatives).
"""

from typing import Dict, List, Optional, Union, Any

# Pyomo & IDAES
from pyomo.environ import (
    Constraint,
    Expression,
    Param,
    Set,
    units as pyunits,
    Var,
    SolverFactory,
    value,
    NonNegativeReals,
)
from idaes.core import (
    PhysicalParameterBlock,
    StateBlock,
    StateBlockData,
    declare_process_block_class,
    MaterialBalanceType,
    EnergyBalanceType,
    Component,
    Phase,
)
from idaes.core.util.initialization import fix_state_vars
from pyomo.common.config import ConfigValue

# Reaktoro
try:
    import reaktoro as rkt
    from GeoH2.equilibrium import defineSystem, equilibrium, defineInitialState
except ImportError:
    rkt = None
    print("Warning: Reaktoro or GeoH2 not found. ReaktoroPropertyPackage will not function.")

import logging

_log = logging.getLogger(__name__)


@declare_process_block_class("ReaktoroParameterBlock")
class ReaktoroParameterBlockData(PhysicalParameterBlock):
    """
    Property Parameter Block Data for Reaktoro-based properties.
    """

    CONFIG = PhysicalParameterBlock.CONFIG()
    
    CONFIG.declare("database", ConfigValue(
        default="SUPRCRT - BL",
        domain=str,
        description="Reaktoro database to use",
        doc="Name of the Reaktoro database (e.g. 'SUPRCRT - BL', 'mines16', etc.)"
    ))

    def build(self):
        """
        Callable method for Block construction.
        """
        super().build()

        self._state_block_class = ReaktoroStateBlock

        # Configuration - We expect the user to pass a 'database' name
        # defaulting to 'SUPRCRT - BL' as used in GeoH2 examples.
        self.database_name = self.config.database
        
        self._rkt_phase_map = {} # Maps IDAES Phase Name -> Reaktoro Phase Name

        # Initialize Reaktoro System
        if rkt:
            # Using GeoH2 helper to get the system
            # system, minerals, solution, gases
            self.rkt_system, self.rkt_minerals, self.rkt_solution, self.rkt_gases = defineSystem(
                self.database_name
            )
            
            # Extract Component List from the System
            # We will use ALL species in the system as components for the IDAES property package
            self.species_list = []
            species_set = set()
            for phase in self.rkt_system.phases():
                for species in phase.species():
                    self.species_list.append(species.name())
                    species_set.add(species.name())
            
            # Extract Phase List & Handle Collisions
            self.phase_list = []
            for p in self.rkt_system.phases():
                p_name = p.name()
                idaes_p_name = p_name
                # Collision check
                if p_name in species_set:
                    idaes_p_name = p_name + "_phase"
                
                self.phase_list.append(idaes_p_name)
                self._rkt_phase_map[idaes_p_name] = p_name

        else:
            # Fallback for when Rektoro is not installed (e.g. testing import)
            self.species_list = ["H2O"]
            self.phase_list = ["AqueousPhase"]


        # Define Components and Phases in IDAES
        # IDAES requires these Sets to exist
        self.component_list = Set(initialize=self.species_list)
        self.phase_list_set = Set(initialize=self.phase_list)

        # Create Component Objects
        # We treat everything as a "Component" for now. 
        # In future we might distinguish Solute/Solvent.
        for c_name in self.species_list:
            self.add_component(c_name, Component())

        # Create Phase Objects
        for p_name in self.phase_list:
            self.add_component(p_name, Phase())
            
    @classmethod
    def define_metadata(cls, obj):
        """Define properties supported and units."""
        obj.add_properties(
            {
                "flow_mol": {"method": None},
                "pressure": {"method": None},
                "temperature": {"method": None},
                "mole_frac_comp": {"method": None},
                "enth_mol_phase": {"method": "_enth_mol_phase"},
                "dens_mol_phase": {"method": "_dens_mol_phase"},
                "entr_mol_phase": {"method": "_entr_mol_phase"},
                # Add other properties as needed
            }
        )
        obj.add_default_units(
            {
                "time": pyunits.s,
                "length": pyunits.m,
                "mass": pyunits.kg,
                "amount": pyunits.mol,
                "temperature": pyunits.K,
            }
        )


@declare_process_block_class("ReaktoroStateBlock")
class ReaktoroStateBlockData(StateBlockData):
    """
    Data container for Reaktoro State Block.
    """

    def build(self):
        """Build the StateBlock objects."""
        super().build()
        
        # State Variables
        self.flow_mol = Var(
            initialize=100.0,
            domain=NonNegativeReals,
            doc="Total molar flowrate [mol/s]",
            units=pyunits.mol / pyunits.s,
        )
        self.pressure = Var(
            initialize=101325.0,
            domain=NonNegativeReals,
            doc="Pressure [Pa]",
            units=pyunits.Pa,
        )
        self.temperature = Var(
            initialize=298.15,
            domain=NonNegativeReals,
            doc="Temperature [K]",
            units=pyunits.K,
        )
        
        self.mole_frac_comp = Var(
            self.params.component_list,
            initialize=1.0 / len(self.params.component_list) if len(self.params.component_list) > 0 else 1.0,
            domain=NonNegativeReals,
            doc="Mole fraction of each component",
            units=pyunits.dimensionless,
        )

        # Property Variables (Variables effectively acting as results)
        # We define them as Variables so IDAES units can link to them.
        # We will NOT add constraints by default, but populate them via methods.
        
        phases = self.params.phase_list_set
        
        self.enth_mol_phase = Var(
            phases,
            initialize=0.0,
            doc="Molar enthalpy of phase [J/mol]",
            units=pyunits.J / pyunits.mol,
        )
        self.dens_mol_phase = Var(
            phases,
            initialize=1000.0, # Dummy init
            doc="Molar density of phase [mol/m^3]",
            units=pyunits.mol / pyunits.m**3
        )
        self.entr_mol_phase = Var(
            phases,
            initialize=0.0,
            doc="Molar entropy of phase [J/mol.K]",
            units=pyunits.J / pyunits.mol / pyunits.K
        )
        self.molecular_weight = Var(
            phases,
            initialize=0.02, # Dummy
            doc="Molecular weight of phase [kg/mol]",
            units=pyunits.kg / pyunits.mol
        )

        # Place to store the underlying Reaktoro state object for persistence/access
        self._rkt_state = None


    def _update_reaktoro_state(self):
        """
        Sync function: Pulls values from Pyomo Vars -> Call Reaktoro -> Push to Pyomo Vars.
        """
        if not rkt:
            return

        # 1. Gather inputs
        T_K = value(self.temperature)
        P_Pa = value(self.pressure)
        
        system = self.params.rkt_system
        state = rkt.ChemicalState(system)
        
        # Set T and P
        state.temperature(T_K, "kelvin")
        state.pressure(P_Pa, "pascal")
        
        # Set Composition (Amounts in Mol)
        for comp_name, frac in self.mole_frac_comp.items():
            val = value(frac)
            if val > 1e-20:
                try:
                    state.setSpeciesAmount(comp_name, val, "mol")
                except:
                    pass
        
        try:
             # Flash calculation to find phase distribution
             res = rkt.EquilibriumSpecs(system)
             res.temperature()
             res.pressure()
             
             # DEBUG: Inspect input state
             # print(f"DEBUG: Solving Equilibrium at T={T_K} K, P={P_Pa} Pa")
             # print(f"DEBUG: Element Amounts: {state.elementAmounts()}")
             
             solver = rkt.SmartEquilibriumSolver(res)
             conditions = rkt.EquilibriumConditions(res)
             conditions.temperature(T_K, "kelvin")
             conditions.pressure(P_Pa, "pascal")
             
             # Solve
             result = solver.solve(state, conditions)
             
             if not result.succeeded():
                 _log.warning(f"Reaktoro equilibrium failed at T={T_K}, P={P_Pa}")
             
             # If successful, printing might be spammy, but useful for now if we only run one test.
                 
        except Exception as e:
            _log.error(f"Error in Reaktoro call: {e}")
            return

        self._rkt_state = state
        props = rkt.ChemicalProps(state)
        
        # 4. Extract Properties and Push to Variables
        
        for phase_idaes in self.params.phase_list:
            
            # Map back to Reaktoro phase name
            # If mapping exists use it, otherwise assume 1:1
            phase_rkt = phase_idaes
            if hasattr(self.params, "_rkt_phase_map"):
                phase_rkt = self.params._rkt_phase_map.get(phase_idaes, phase_idaes)
                
            try:
                # Check if phase exists in current state or system
                # Using props.phaseProps(name)
                
                # Careful: calling phaseProps on a phase that has 0 amount might result in properties?
                p_props = props.phaseProps(phase_rkt)
                
                # Enthalpy (J/kg -> J/mol)
                mw = p_props.molarMass() # kg/mol
                
                # specificEnthalpy() is J/kg.
                h_mass = float(p_props.specificEnthalpy())
                self.enth_mol_phase[phase_idaes].set_value(h_mass * float(mw))
                
                # Density (kg/m3 -> mol/m3)
                rho_mass = float(p_props.density())
                self.dens_mol_phase[phase_idaes].set_value(rho_mass / float(mw)) 
                
                # Entropy (J/kg/K -> J/mol/K)
                s_mass = float(p_props.specificEntropy())
                self.entr_mol_phase[phase_idaes].set_value(s_mass * float(mw))

                self.molecular_weight[phase_idaes].set_value(float(mw))
            except Exception as e:
                pass


    # Defining Standard Accessor Methods for IDAES
    def _enth_mol_phase(self):
        return self.enth_mol_phase

    def _dens_mol_phase(self):
        return self.dens_mol_phase

    def _entr_mol_phase(self):
        return self.entr_mol_phase
        
    def get_material_flow_terms(self, p, j):
        return self.flow_mol * self.mole_frac_comp[j]

    def get_enthalpy_flow_terms(self, p):
        return self.flow_mol * self.enth_mol_phase[p] 

    def get_material_density_terms(self, p, j):
        return self.dens_mol_phase[p] * self.mole_frac_comp[j] 

    def get_energy_density_terms(self, p):
        return self.dens_mol_phase[p] * self.enth_mol_phase[p]

    def define_state_vars(self):
        return {
            "flow_mol": self.flow_mol,
            "pressure": self.pressure,
            "temperature": self.temperature,
            "mole_frac_comp": self.mole_frac_comp,
        }
        
    def fix_initialization_states(self):
        fix_state_vars(self)

    def initialize(
        self,
        state_args=None,
        hold_state=False,
        state_vars_fixed=False,
        outlvl=0,
        solver=None,
        optarg=None,
    ):
        if state_args is None:
            state_args = {}
        self.calculate_properties()

    def calculate_properties(self):
        self._update_reaktoro_state()
