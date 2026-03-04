"""
REE Thermodynamic Database Configurations for Reaktoro
======================================================

Provides pre-configured Reaktoro ``ChemicalSystem`` objects optimised for
rare earth element (REE) separation processes.  All configurations use
the ``supcrtbl`` database, which contains 300+ REE aqueous species
covering all 14 lanthanides plus Y and Sc.

Supported REE elements per database
------------------------------------
- La, Ce, Pr, Nd, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, Y, Sc

Available system presets
------------------------
- ``light_ree``  : La, Ce, Pr, Nd  (dominant in monazite/bastnäsite)
- ``heavy_ree``  : Sm–Lu, Y, Sc    (ion-adsorption clays, xenotime)
- ``full_ree``   : all 16 REE elements
- ``custom``     : user-selected element set

Each preset includes common gangue elements (Fe, Al, Ca, Si, Mg, Na)
and acid/base species (HCl, HNO₃, NaOH, H₂SO₄).

Usage
-----
>>> from sep_agents.properties.ree_databases import build_ree_system
>>> system = build_ree_system("light_ree")           # preset
>>> system = build_ree_system(elements=["Ce", "Nd"])  # custom
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

_log = logging.getLogger(__name__)

try:
    import reaktoro as rkt
    REAKTORO_AVAILABLE = True
except ImportError:
    rkt = None
    REAKTORO_AVAILABLE = False


# ---------------------------------------------------------------------------
# Element groups
# ---------------------------------------------------------------------------
LIGHT_REE = ["La", "Ce", "Pr", "Nd"]
HEAVY_REE = ["Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
ALL_REE = LIGHT_REE + HEAVY_REE + ["Y", "Sc"]

# Common gangue / leaching elements
GANGUE_ELEMENTS = ["Fe", "Al", "Ca", "Si", "Mg", "Mn"]
BASE_ELEMENTS = ["H", "O", "Na", "K", "Cl", "C", "S", "N", "P", "F"]

# Pre-defined element sets
PRESET_ELEMENTS: Dict[str, List[str]] = {
    "light_ree": BASE_ELEMENTS + LIGHT_REE + GANGUE_ELEMENTS,
    "heavy_ree": BASE_ELEMENTS + HEAVY_REE + ["Y", "Sc"] + GANGUE_ELEMENTS,
    "full_ree":  BASE_ELEMENTS + ALL_REE + GANGUE_ELEMENTS,
}

# Common REE minerals (available in supcrtbl as mineral phases)
# These are minerals that appear in the database — verified via species scan.
REE_MINERALS_COMMON = [
    # Will be populated dynamically when building the system,
    # since mineral availability depends on the element set.
]

# ---------------------------------------------------------------------------
# System builder
# ---------------------------------------------------------------------------
def build_ree_system(
    preset: Optional[str] = None,
    elements: Optional[Sequence[str]] = None,
    database: str = "supcrtbl",
    include_minerals: bool = True,
    include_gases: bool = True,
    extra_elements: Optional[Sequence[str]] = None,
) -> "rkt.ChemicalSystem":
    """Build a Reaktoro ChemicalSystem for REE separation.

    Parameters
    ----------
    preset : str, optional
        One of ``"light_ree"``, ``"heavy_ree"``, ``"full_ree"``.
        Ignored if *elements* is provided.
    elements : sequence of str, optional
        Explicit list of elements to include (e.g. ``["Ce", "Nd", "Fe"]``).
        Base elements (H, O, Na, Cl, …) are always added automatically.
    database : str
        Reaktoro database name (default ``"supcrtbl"``).
    include_minerals : bool
        If True, add mineral phases found in the database for the element set.
    include_gases : bool
        If True, add a gas phase (CO₂, H₂O, O₂, …).
    extra_elements : sequence of str, optional
        Additional elements to add on top of preset/specified elements.

    Returns
    -------
    reaktoro.ChemicalSystem
    """
    if not REAKTORO_AVAILABLE:
        raise ImportError("Reaktoro is required. Install via: conda install -c conda-forge reaktoro")

    # Resolve element set
    if elements is not None:
        elem_set = list(set(BASE_ELEMENTS + list(elements)))
    elif preset is not None:
        if preset not in PRESET_ELEMENTS:
            raise ValueError(f"Unknown preset '{preset}'. Choose from: {list(PRESET_ELEMENTS)}")
        elem_set = list(PRESET_ELEMENTS[preset])
    else:
        elem_set = list(PRESET_ELEMENTS["light_ree"])

    if extra_elements:
        elem_set = list(set(elem_set + list(extra_elements)))

    _log.info("Building REE system with elements: %s", sorted(elem_set))

    # Load database and add custom REE precipitates (e.g. Hydroxides)
    db = rkt.SupcrtDatabase(database)
    
    # Ksp -> G0 conversion logic for Light REE hydroxides
    # Ksp values (as pKsp) typical for REE(OH)3
    hydroxides = [
        ("La(OH)3(s)", "La+3", 20.7),
        ("Ce(OH)3(s)", "Ce+3", 19.7),
        ("Pr(OH)3(s)", "Pr+3", 23.47),
        ("Nd(OH)3(s)", "Nd+3", 21.49),
        ("Sm(OH)3(s)", "Sm+3", 22.08)
    ]
    
    import math
    R = 8.31446261815324
    T = 298.15
    P = 100000.0
    vol_params = rkt.StandardVolumeModelParamsConstant()
    vol_params.V0 = 5.0e-5  # ~50 cm3/mol
    
    try:
        OH = db.species().get("OH-")
        G0_OH = OH.standardThermoProps(T, P).G0
        
        for sp_name, cation_name, pKsp in hydroxides:
            # Only add if the cation exists in the current element set
            if not any(cation_name.startswith(el) for el in elem_set):
                continue
                
            cation = db.species().get(cation_name)
            G0_cation = cation.standardThermoProps(T, P).G0
            
            dG_rxn = -R * T * pKsp * math.log(10)
            G0_product = dG_rxn + G0_cation + 3.0 * G0_OH
            
            # Element composition
            elem_comp = [(cation.elements().symbols()[0], 1.0), ("O", 3.0), ("H", 3.0)]
            
            # Standard thermo model
            model = rkt.StandardThermoModelParamsConstant()
            model.G0 = float(G0_product)
            model.V0 = vol_params.V0
            thermo_model = rkt.StandardThermoModelConstant(model)
            
            # Define and add species
            sp = rkt.Species().withName(sp_name) \
                .withElements(rkt.ElementalComposition(elem_comp)) \
                .withAggregateState(rkt.AggregateState.CrystallineSolid) \
                .withStandardThermoModel(thermo_model)
                
            db.addSpecies(sp)
            _log.debug("Injected custom species %s (pKsp=%.2f)", sp_name, pKsp)
    except Exception as e:
        _log.warning("Failed to inject custom REE species: %s", e)

    # ---------------------------------------------------------
    # Custom Oxalate Injection (Metastable organic pseudo-element)
    # ---------------------------------------------------------
    if "Ox" in elem_set:
        try:
            # 1. Define 'Ox' pseudo-element to isolate oxalate from inorganic Carbon (preventing CO2/CH4 conversion)
            try:
                ox = rkt.Element().withName("Oxalate").withSymbol("Ox").withMolarMass(88.019)
                # Register globally so ElementalComposition parser can find it
                rkt.Elements.append(ox)
                db.addElement(ox)
            except Exception:
                pass # Might already exist if db object is reused
                
            # Base thermodynamic values for Oxalate
            G0_C2O4 = -674000.0  # J/mol for C2O4-2 (aq) NBS tables
            
            # Form thermodynamic models for aqueous oxalate species based on pKa
            # pKa2 = 4.14 for HC2O4- -> H+ + C2O4-2
            dG_a2 = R * T * 4.14 * math.log(10)
            G0_HC2O4 = 0.0 + G0_C2O4 - dG_a2
            
            # pKa1 = 1.25 for H2C2O4 -> H+ + HC2O4-
            dG_a1 = R * T * 1.25 * math.log(10)
            G0_H2C2O4 = 0.0 + G0_HC2O4 - dG_a1
            
            aq_ox_data = [
                ("C2O4-2", [("Ox", 1.0)], G0_C2O4, -2.0),
                ("HC2O4-", [("H", 1.0), ("Ox", 1.0)], G0_HC2O4, -1.0),
                ("H2C2O4(aq)", [("H", 2.0), ("Ox", 1.0)], G0_H2C2O4, 0.0)
            ]
            
            for name, comp, g0, charge in aq_ox_data:
                model = rkt.StandardThermoModelParamsConstant()
                model.G0 = g0
                model.V0 = 5.0e-5
                sp = rkt.Species().withName(name) \
                    .withElements(rkt.ElementalComposition(comp)) \
                    .withCharge(charge) \
                    .withAggregateState(rkt.AggregateState.Aqueous) \
                    .withStandardThermoModel(rkt.StandardThermoModelConstant(model))
                db.addSpecies(sp)
                _log.debug("Injected aqueous oxalate %s (G0=%.2f)", name, g0)
                
            # Now add REE oxalates: 2 REE+3 + 3 C2O4-2 -> REE2(C2O4)3(s)
            # Literature pKsp ranges: La = ~25, Ce = ~28, Pr = 30.82, Nd = 31.14, Sm = ~32
            oxalates = [
                ("La2(C2O4)3(s)", "La+3", 25.0),
                ("Ce2(C2O4)3(s)", "Ce+3", 28.0),
                ("Pr2(C2O4)3(s)", "Pr+3", 30.82),
                ("Nd2(C2O4)3(s)", "Nd+3", 31.14),
                ("Sm2(C2O4)3(s)", "Sm+3", 32.0)
            ]
            
            for sp_name, cation_name, pKsp in oxalates:
                if not any(cation_name.startswith(el) for el in elem_set):
                    continue
                    
                cation = db.species().get(cation_name)
                G0_cation = cation.standardThermoProps(T, P).G0
                
                # dG_rxn = Product - Reactants = G0_Product - (2 * G0_Cation + 3 * G0_C2O4)
                # dG_rxn = -RT ln Kf = -RT * pKsp * ln 10
                dG_rxn = -R * T * pKsp * math.log(10)
                G0_product = dG_rxn + 2.0 * G0_cation + 3.0 * G0_C2O4
                
                elem_comp = [(cation.elements().symbols()[0], 2.0), ("Ox", 3.0)]
                model = rkt.StandardThermoModelParamsConstant()
                model.G0 = float(G0_product)
                model.V0 = 1.5e-4 # ~150 cm3/mol for solid
                
                sp = rkt.Species().withName(sp_name) \
                    .withElements(rkt.ElementalComposition(elem_comp)) \
                    .withAggregateState(rkt.AggregateState.CrystallineSolid) \
                    .withStandardThermoModel(rkt.StandardThermoModelConstant(model))
                db.addSpecies(sp)
                _log.debug("Injected solid oxalate %s (pKsp=%.2f)", sp_name, pKsp)
                
        except Exception as e:
            _log.warning("Failed to inject custom Oxalate species: %s", e)

    # Build aqueous phase from element speciation
    elem_string = " ".join(sorted(set(elem_set)))
    aq_phase = rkt.AqueousPhase(rkt.speciate(elem_string))

    phases = [aq_phase]

    # Find and add mineral phases
    if include_minerals:
        mineral_phases = _find_mineral_phases(db, elem_set)
        for mineral_name in mineral_phases:
            try:
                phases.append(rkt.MineralPhase(mineral_name))
            except Exception as exc:
                _log.debug("Could not add mineral phase '%s': %s", mineral_name, exc)

    # Add gas phase
    if include_gases:
        gas_species = _find_gas_species(db, elem_set)
        if gas_species:
            try:
                phases.append(rkt.GaseousPhase(gas_species))
            except Exception as exc:
                _log.debug("Could not add gas phase: %s", exc)

    system = rkt.ChemicalSystem(db, *phases)
    _log.info(
        "REE system: %d species, %d phases",
        system.species().size(),
        system.phases().size(),
    )
    return system


def _find_mineral_phases(db: "rkt.SupcrtDatabase", elements: List[str], targets: Optional[List[str]] = None) -> List[str]:
    """Find mineral species in the database whose elements are a subset of *elements*.
    Restricts search to `targets` to prevent solver Jacobian singularities from 0-mass trace solids.
    """
    elem_set = set(elements)
    minerals = []
    
    if targets is None:
        targets = [
            "La(OH)3(s)", "Ce(OH)3(s)", "Pr(OH)3(s)", "Nd(OH)3(s)", "Sm(OH)3(s)",
            "La2(C2O4)3(s)", "Ce2(C2O4)3(s)", "Pr2(C2O4)3(s)", "Nd2(C2O4)3(s)", "Sm2(C2O4)3(s)"
        ]

    for sp in db.species():
        if sp.name() in targets:
            sp_elements = set(sp.elements().symbols())
            if sp_elements and sp_elements.issubset(elem_set):
                minerals.append(sp.name())

    _log.info("Found %d targeted mineral phases compatible with element set", len(minerals))
    return minerals


def _find_gas_species(db: "rkt.SupcrtDatabase", elements: List[str]) -> List[str]:
    """Find gaseous species compatible with the element set."""
    elem_set = set(elements)
    gases = []

    for sp in db.species():
        agg_state = str(sp.aggregateState())
        if "Gas" not in agg_state:
            continue

        sp_elements = set(sp.elements().symbols())

        if sp_elements and sp_elements.issubset(elem_set):
            gases.append(sp.name())

    return gases


# ---------------------------------------------------------------------------
# REE-specific equilibrium helper
# ---------------------------------------------------------------------------
class REEEquilibriumSolver:
    """Convenience wrapper for REE speciation calculations.

    Parameters
    ----------
    preset : str
        System preset (``"light_ree"``, ``"heavy_ree"``, ``"full_ree"``).
    database : str
        Reaktoro database name.
    use_jax : bool
        If True, use the JAX-based GEM solver instead of Reaktoro.
        The JAX solver is fully differentiable via ``jax.grad`` and
        does not require Reaktoro to be installed.
    """

    def __init__(self, preset: str = "light_ree", database: str = "supcrtbl", extra_elements: Optional[Sequence[str]] = None, use_jax: bool = False):
        self.use_jax = use_jax
        self.preset = preset

        if use_jax:
            from ..sim.jax_equilibrium import build_jax_system, JaxEquilibriumSolver
            self._jax_system = build_jax_system(preset=preset)
            self._jax_solver = JaxEquilibriumSolver(self._jax_system)
            self.system = None
            self.solver = None
        else:
            if not REAKTORO_AVAILABLE:
                raise ImportError(
                    "Reaktoro is required when use_jax=False. "
                    "Install via: conda install -c conda-forge reaktoro, "
                    "or set use_jax=True to use the JAX backend."
                )
            self.system = build_ree_system(preset=preset, database=database, extra_elements=extra_elements)
            self.solver = rkt.EquilibriumSolver(self.system)
            self._jax_system = None
            self._jax_solver = None

    def speciate(
        self,
        temperature_C: float = 25.0,
        pressure_atm: float = 1.0,
        water_kg: float = 1.0,
        acid_mol: Optional[Dict[str, float]] = None,
        ree_mol: Optional[Dict[str, float]] = None,
        other_mol: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Run equilibrium speciation and return results.

        Parameters
        ----------
        temperature_C : float
            Temperature in °C.
        pressure_atm : float
            Pressure in atm.
        water_kg : float
            Mass of water in kg.
        acid_mol : dict
            Acid species and amounts, e.g. ``{"HCl(aq)": 0.5}``.
        ree_mol : dict
            REE species and amounts, e.g. ``{"Ce+3": 0.01, "Nd+3": 0.005}``.
        other_mol : dict
            Other species and amounts, e.g. ``{"NaCl(aq)": 0.1}``.

        Returns
        -------
        dict
            Keys: ``pH``, ``Eh_V``, ``species`` (sorted by amount),
            ``ree_distribution`` (REE species only).
        """
        # --- JAX backend ---
        if self.use_jax:
            return self._jax_solver.solve_speciation(
                temperature_C=temperature_C,
                pressure_atm=pressure_atm,
                water_kg=water_kg,
                acid_mol=acid_mol,
                ree_mol=ree_mol,
                other_mol=other_mol,
            )

        # --- Reaktoro backend (original) ---
        state = rkt.ChemicalState(self.system)
        state.temperature(temperature_C, "celsius")
        state.pressure(pressure_atm, "atm")
        state.set("H2O(aq)", water_kg, "kg")

        for species_dict in [acid_mol, ree_mol, other_mol]:
            if species_dict:
                for sp_name, amount in species_dict.items():
                    state.set(sp_name, amount, "mol")

    # Seed only our custom REE hydroxides to help solver convergence if not already mapped
        # Seeding universally across 200+ mineral phases causes Jacobian singularity.
        user_species = set()
        for d in [acid_mol, ree_mol, other_mol]:
            if d: user_species.update(d.keys())
            
        custom_targets = [
            "La(OH)3(s)", "Ce(OH)3(s)", "Pr(OH)3(s)", "Nd(OH)3(s)", "Sm(OH)3(s)",
            "La2(C2O4)3(s)", "Ce2(C2O4)3(s)", "Pr2(C2O4)3(s)", "Nd2(C2O4)3(s)", "Sm2(C2O4)3(s)"
        ]
        system_sp_names = [s.name() for s in self.system.species()]
        for sp_name in custom_targets:
            if sp_name in system_sp_names and sp_name not in user_species:
                state.set(sp_name, 1e-8, "mol")

        result = self.solver.solve(state)
        if not result.succeeded():
            return {"status": "error", "error": "Equilibrium solve did not converge"}

        # Extract results
        aprops = rkt.AqueousProps(state)
        props = rkt.ChemicalProps(state)

        # All species with non-negligible amounts
        all_species = {}
        ree_species = {}
        ree_symbols = set(ALL_REE)

        for i, sp in enumerate(self.system.species()):
            amt = float(state.speciesAmount(i))
            if amt > 1e-15:
                all_species[sp.name()] = amt

                # Check if it's an REE species
                sp_elems = set(sp.elements().symbols())
                if sp_elems & ree_symbols:
                    ree_species[sp.name()] = amt

        return {
            "status": "ok",
            "temperature_C": temperature_C,
            "pressure_atm": pressure_atm,
            "pH": round(float(aprops.pH()), 4),
            "Eh_V": round(float(aprops.Eh()), 4),
            "ionic_strength": round(float(aprops.ionicStrength()), 4),
            "species": dict(sorted(all_species.items(), key=lambda kv: -kv[1])),
            "ree_distribution": dict(sorted(ree_species.items(), key=lambda kv: -kv[1])),
        }

    def separation_factors(
        self,
        speciation_result: Dict,
        element_a: str,
        element_b: str,
    ) -> float:
        """Calculate separation factor β(A/B) from speciation result.

        β = (total_A_in_solution / total_A_input) / (total_B_in_solution / total_B_input)

        For an equilibrium calculation, this simplifies to the ratio of total
        dissolved amounts of A vs B since both were fully dissolved.

        Parameters
        ----------
        speciation_result : dict
            Output from :meth:`speciate`.
        element_a, element_b : str
            REE element symbols (e.g. ``"Ce"``, ``"Nd"``).

        Returns
        -------
        float
            Separation factor β(A/B). Values > 1 mean A partitions more
            into solution; values near 1 mean poor separation.
        """
        ree_dist = speciation_result.get("ree_distribution", {})

        total_a = sum(v for k, v in ree_dist.items() if element_a in k)
        total_b = sum(v for k, v in ree_dist.items() if element_b in k)

        if total_b <= 0:
            return float("inf")
        return total_a / total_b
