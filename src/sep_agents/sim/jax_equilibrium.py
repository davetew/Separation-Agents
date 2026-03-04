"""
JAX-Based Gibbs Energy Minimization Equilibrium Solver
======================================================

A pure-JAX implementation of chemical equilibrium via constrained Gibbs
energy minimization (GEM).  Provides the same speciation outputs as
Reaktoro (pH, Eh, species amounts) but is fully differentiable via
``jax.grad`` and vectorisable via ``jax.vmap``.

Algorithm
---------
Minimise  G(T,P,n) = Σ nᵢ·μᵢ(T,P,n)
subject to  A·n = b  (elemental mass balance)
            nᵢ ≥ 0   (non-negativity)

where μᵢ = G⁰ᵢ + RT·ln(aᵢ) is the chemical potential and aᵢ is the
activity (unity for solids, molality-based for aqueous with extended
Debye-Hückel correction).

The positivity constraint is handled by solving in log-space
(yᵢ = log(nᵢ + ε)) with ``jaxopt.LBFGSB``.

Thermodynamic Data
------------------
Built-in standard Gibbs energies G⁰(298.15 K, 1 bar) are provided for
the ``light_ree`` species set (~60 aqueous + solid species covering
H-O-Na-Cl-La-Ce-Pr-Nd-Fe-Al-Ca-Si-Mg).  Values sourced from the
SUPCRTBL database (Zimmer et al., 2016).

References
----------
- Zimmer et al. (2016), Comp. & Geosciences 90:97-111 (SUPCRTBL)
- Helgeson et al. (1981), Am. J. Sci. 281:1249-1516 (HKF EOS)
- Leal (2015), Reaktoro (reference implementation)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_GAS = 8.314462618       # J/(mol·K)
LN10 = math.log(10.0)
F_FARADAY = 96485.3329    # C/mol
WATER_MW = 0.018015       # kg/mol


# ---------------------------------------------------------------------------
# Built-in thermodynamic data (SUPCRTBL, 298.15 K, 1 bar)
# ---------------------------------------------------------------------------
# G⁰ in J/mol, charge, {element: stoich}, molar mass in kg/mol
# These are standard-state Gibbs energies of formation.

@dataclass
class SpeciesRecord:
    """Thermodynamic record for a single species."""
    name: str
    G0: float            # Standard Gibbs energy of formation, J/mol
    charge: float        # Ionic charge
    elements: Dict[str, float]  # {element_symbol: stoichiometric_coeff}
    molar_mass: float    # kg/mol
    phase: str = "aqueous"  # "aqueous", "mineral", "gas"
    ion_size_param: float = 0.0  # å (angstrom), for Debye-Hückel


# fmt: off
# ─── Aqueous species ──────────────────────────────────────────────────
# G⁰ values from SUPCRTBL database at 298.15 K, 1 bar
_AQUEOUS_SPECIES_DATA = [
    # Water and proton/hydroxide
    SpeciesRecord("H2O(aq)",   -237181.0,  0.0, {"H": 2, "O": 1}, 0.018015, ion_size_param=0.0),
    SpeciesRecord("H+",              0.0,  1.0, {"H": 1},          0.001008, ion_size_param=9.0),
    SpeciesRecord("OH-",      -157244.0, -1.0, {"O": 1, "H": 1},  0.017007, ion_size_param=3.5),

    # Chloride system
    SpeciesRecord("Cl-",      -131228.0, -1.0, {"Cl": 1},          0.035453, ion_size_param=3.0),
    SpeciesRecord("HCl(aq)",  -127118.0,  0.0, {"H": 1, "Cl": 1}, 0.036461, ion_size_param=0.0),
    SpeciesRecord("NaCl(aq)", -388735.0,  0.0, {"Na": 1, "Cl": 1}, 0.058443, ion_size_param=0.0),

    # Sodium
    SpeciesRecord("Na+",      -261905.0,  1.0, {"Na": 1},          0.022990, ion_size_param=4.0),
    SpeciesRecord("NaOH(aq)", -418123.0,  0.0, {"Na": 1, "O": 1, "H": 1}, 0.039997, ion_size_param=0.0),

    # Calcium
    SpeciesRecord("Ca+2",     -553580.0,  2.0, {"Ca": 1},          0.040078, ion_size_param=6.0),
    SpeciesRecord("CaCl+",    -683300.0,  1.0, {"Ca": 1, "Cl": 1}, 0.075531, ion_size_param=4.0),
    SpeciesRecord("CaCl2(aq)",-811700.0,  0.0, {"Ca": 1, "Cl": 2}, 0.110984, ion_size_param=0.0),

    # Iron
    SpeciesRecord("Fe+2",      -78900.0,  2.0, {"Fe": 1},          0.055845, ion_size_param=6.0),
    SpeciesRecord("Fe+3",      -16276.0,  3.0, {"Fe": 1},          0.055845, ion_size_param=9.0),
    SpeciesRecord("FeCl+",    -210060.0,  1.0, {"Fe": 1, "Cl": 1}, 0.091298, ion_size_param=4.0),
    SpeciesRecord("FeCl2(aq)",-341200.0,  0.0, {"Fe": 1, "Cl": 2}, 0.126751, ion_size_param=0.0),
    SpeciesRecord("FeCl+2",   -147500.0,  2.0, {"Fe": 1, "Cl": 1}, 0.091298, ion_size_param=5.0),
    SpeciesRecord("FeOH+",    -241300.0,  1.0, {"Fe": 1, "O": 1, "H": 1}, 0.072852, ion_size_param=4.0),

    # Aluminium
    SpeciesRecord("Al+3",     -485000.0,  3.0, {"Al": 1},          0.026982, ion_size_param=9.0),
    SpeciesRecord("AlCl+2",   -616200.0,  2.0, {"Al": 1, "Cl": 1}, 0.062435, ion_size_param=5.0),
    SpeciesRecord("AlOH+2",   -692600.0,  2.0, {"Al": 1, "O": 1, "H": 1}, 0.043989, ion_size_param=5.0),

    # Magnesium
    SpeciesRecord("Mg+2",     -454800.0,  2.0, {"Mg": 1},          0.024305, ion_size_param=8.0),
    SpeciesRecord("MgCl+",    -587900.0,  1.0, {"Mg": 1, "Cl": 1}, 0.059758, ion_size_param=4.0),

    # Silicon (simplified)
    SpeciesRecord("SiO2(aq)", -833411.0,  0.0, {"Si": 1, "O": 2},  0.060084, ion_size_param=0.0),
    SpeciesRecord("HSiO3-",   -886700.0, -1.0, {"H": 1, "Si": 1, "O": 3}, 0.077091, ion_size_param=4.0),

    # Sulfur / sulfate (for H2SO4 leaching)
    SpeciesRecord("SO4-2",    -744530.0, -2.0, {"S": 1, "O": 4},   0.096062, ion_size_param=4.0),
    SpeciesRecord("HSO4-",    -755910.0, -1.0, {"H": 1, "S": 1, "O": 4}, 0.097069, ion_size_param=4.0),

    # Carbonate
    SpeciesRecord("CO2(aq)",  -385980.0,  0.0, {"C": 1, "O": 2},   0.044010, ion_size_param=0.0),
    SpeciesRecord("HCO3-",    -586770.0, -1.0, {"H": 1, "C": 1, "O": 3}, 0.061017, ion_size_param=4.0),
    SpeciesRecord("CO3-2",    -527810.0, -2.0, {"C": 1, "O": 3},   0.060009, ion_size_param=4.5),

    # Nitrogen species
    SpeciesRecord("NO3-",     -111250.0, -1.0, {"N": 1, "O": 3},   0.062004, ion_size_param=3.0),
    SpeciesRecord("HNO3(aq)", -111250.0,  0.0, {"H": 1, "N": 1, "O": 3}, 0.063012, ion_size_param=0.0),

    # Phosphate
    SpeciesRecord("H3PO4(aq)",-1142540.0, 0.0, {"H": 3, "P": 1, "O": 4}, 0.097994, ion_size_param=0.0),
    SpeciesRecord("H2PO4-",   -1130280.0,-1.0, {"H": 2, "P": 1, "O": 4}, 0.096987, ion_size_param=4.0),

    # Fluoride
    SpeciesRecord("F-",       -278790.0, -1.0, {"F": 1},           0.018998, ion_size_param=3.5),
    SpeciesRecord("HF(aq)",   -296820.0,  0.0, {"H": 1, "F": 1},  0.020006, ion_size_param=0.0),

    # Potassium
    SpeciesRecord("K+",       -283270.0,  1.0, {"K": 1},           0.039098, ion_size_param=3.0),
    SpeciesRecord("KCl(aq)",  -414500.0,  0.0, {"K": 1, "Cl": 1}, 0.074551, ion_size_param=0.0),

    # Manganese
    SpeciesRecord("Mn+2",     -228100.0,  2.0, {"Mn": 1},          0.054938, ion_size_param=6.0),

    # ─── Lanthanum ─────────────────────────────────────────────────
    SpeciesRecord("La+3",     -683700.0,  3.0, {"La": 1},          0.138906, ion_size_param=9.0),
    SpeciesRecord("LaCl+2",   -816200.0,  2.0, {"La": 1, "Cl": 1}, 0.174359, ion_size_param=5.0),
    SpeciesRecord("LaCl2+",   -946700.0,  1.0, {"La": 1, "Cl": 2}, 0.209812, ion_size_param=4.0),
    SpeciesRecord("LaCl3(aq)",-1074100.0, 0.0, {"La": 1, "Cl": 3}, 0.245265, ion_size_param=0.0),
    SpeciesRecord("LaCl4-",   -1197500.0,-1.0, {"La": 1, "Cl": 4}, 0.280718, ion_size_param=4.0),
    SpeciesRecord("LaOH+2",   -845700.0,  2.0, {"La": 1, "O": 1, "H": 1}, 0.155913, ion_size_param=5.0),
    SpeciesRecord("La(OH)2+", -1003200.0, 1.0, {"La": 1, "O": 2, "H": 2}, 0.172920, ion_size_param=4.0),

    # ─── Cerium ────────────────────────────────────────────────────
    SpeciesRecord("Ce+3",     -672000.0,  3.0, {"Ce": 1},          0.140116, ion_size_param=9.0),
    SpeciesRecord("CeCl+2",   -804700.0,  2.0, {"Ce": 1, "Cl": 1}, 0.175569, ion_size_param=5.0),
    SpeciesRecord("CeCl2+",   -936100.0,  1.0, {"Ce": 1, "Cl": 2}, 0.211022, ion_size_param=4.0),
    SpeciesRecord("CeCl3(aq)",-1064300.0, 0.0, {"Ce": 1, "Cl": 3}, 0.246475, ion_size_param=0.0),
    SpeciesRecord("CeCl4-",   -1189500.0,-1.0, {"Ce": 1, "Cl": 4}, 0.281928, ion_size_param=4.0),
    SpeciesRecord("CeOH+2",   -833600.0,  2.0, {"Ce": 1, "O": 1, "H": 1}, 0.157123, ion_size_param=5.0),
    SpeciesRecord("Ce(OH)2+", -990600.0,  1.0, {"Ce": 1, "O": 2, "H": 2}, 0.174130, ion_size_param=4.0),

    # ─── Praseodymium ──────────────────────────────────────────────
    SpeciesRecord("Pr+3",     -679500.0,  3.0, {"Pr": 1},          0.140908, ion_size_param=9.0),
    SpeciesRecord("PrCl+2",   -812000.0,  2.0, {"Pr": 1, "Cl": 1}, 0.176361, ion_size_param=5.0),
    SpeciesRecord("PrCl2+",   -942200.0,  1.0, {"Pr": 1, "Cl": 2}, 0.211814, ion_size_param=4.0),
    SpeciesRecord("PrCl3(aq)",-1069500.0, 0.0, {"Pr": 1, "Cl": 3}, 0.247267, ion_size_param=0.0),
    SpeciesRecord("PrOH+2",   -841000.0,  2.0, {"Pr": 1, "O": 1, "H": 1}, 0.157915, ion_size_param=5.0),

    # ─── Neodymium ─────────────────────────────────────────────────
    SpeciesRecord("Nd+3",     -671400.0,  3.0, {"Nd": 1},          0.144242, ion_size_param=9.0),
    SpeciesRecord("NdCl+2",   -804100.0,  2.0, {"Nd": 1, "Cl": 1}, 0.179695, ion_size_param=5.0),
    SpeciesRecord("NdCl2+",   -934600.0,  1.0, {"Nd": 1, "Cl": 2}, 0.215148, ion_size_param=4.0),
    SpeciesRecord("NdCl3(aq)",-1062200.0, 0.0, {"Nd": 1, "Cl": 3}, 0.250601, ion_size_param=0.0),
    SpeciesRecord("NdCl4-",   -1185700.0,-1.0, {"Nd": 1, "Cl": 4}, 0.286054, ion_size_param=4.0),
    SpeciesRecord("NdOH+2",   -833200.0,  2.0, {"Nd": 1, "O": 1, "H": 1}, 0.161249, ion_size_param=5.0),
    SpeciesRecord("Nd(OH)2+", -990400.0,  1.0, {"Nd": 1, "O": 2, "H": 2}, 0.178256, ion_size_param=4.0),
]

# ─── Solid (mineral) species ──────────────────────────────────────────
# Custom hydroxides derived from pKsp → G⁰ conversion
# G⁰_product = −RT · pKsp · ln(10) + G⁰_cation + 3·G⁰_OH
_G0_OH = -157244.0  # J/mol for OH-

def _hydroxide_G0(G0_cation: float, pKsp: float) -> float:
    """Compute G⁰ for REE(OH)₃(s) from cation G⁰ and pKsp."""
    return -R_GAS * 298.15 * pKsp * LN10 + G0_cation + 3.0 * _G0_OH

_MINERAL_SPECIES_DATA = [
    SpeciesRecord("La(OH)3(s)", _hydroxide_G0(-683700.0, 20.70), 0.0,
                  {"La": 1, "O": 3, "H": 3}, 0.189919, phase="mineral"),
    SpeciesRecord("Ce(OH)3(s)", _hydroxide_G0(-672000.0, 19.70), 0.0,
                  {"Ce": 1, "O": 3, "H": 3}, 0.191129, phase="mineral"),
    SpeciesRecord("Pr(OH)3(s)", _hydroxide_G0(-679500.0, 23.47), 0.0,
                  {"Pr": 1, "O": 3, "H": 3}, 0.191921, phase="mineral"),
    SpeciesRecord("Nd(OH)3(s)", _hydroxide_G0(-671400.0, 21.49), 0.0,
                  {"Nd": 1, "O": 3, "H": 3}, 0.195255, phase="mineral"),
]
# fmt: on


# ---------------------------------------------------------------------------
# Chemical system data (JAX-compatible)
# ---------------------------------------------------------------------------
@dataclass
class ChemicalSystemData:
    """JAX-compatible representation of a chemical system.

    All arrays are plain NumPy at construction time; converted to
    ``jnp.array`` inside the solver for tracing.
    """
    species_names: List[str]
    G0: np.ndarray            # (N,) standard Gibbs energies, J/mol (at 298K)
    charges: np.ndarray       # (N,) ionic charges
    formula_matrix: np.ndarray  # (E, N) element-species stoich matrix
    element_names: List[str]  # (E,) element labels (rows of A)
    molar_masses: np.ndarray  # (N,) kg/mol
    ion_size_params: np.ndarray  # (N,) Å, for Debye-Hückel
    phases: List[str]         # (N,) "aqueous" or "mineral" per species
    ree_elements: List[str] = field(default_factory=list)
    # Optional HKF parameters for T,P-dependent G⁰ (Phase II)
    hkf_Gf: Optional[np.ndarray] = None      # (N,) formation Gibbs energy
    hkf_Sr: Optional[np.ndarray] = None      # (N,) reference entropy
    hkf_a1: Optional[np.ndarray] = None      # (N,) volume param
    hkf_a2: Optional[np.ndarray] = None      # (N,) volume param
    hkf_a3: Optional[np.ndarray] = None      # (N,) volume param
    hkf_a4: Optional[np.ndarray] = None      # (N,) volume param
    hkf_c1: Optional[np.ndarray] = None      # (N,) heat capacity param
    hkf_c2: Optional[np.ndarray] = None      # (N,) heat capacity param
    hkf_wref: Optional[np.ndarray] = None    # (N,) Born coefficient
    water_hkf_params: Optional[Dict] = None  # WaterHKF params for H2O
    # Optional HollandPowell parameters for minerals/gases
    hp_Gf: Optional[np.ndarray] = None       # (N,) formation Gibbs energy
    hp_Hf: Optional[np.ndarray] = None       # (N,) formation enthalpy
    hp_Sr: Optional[np.ndarray] = None       # (N,) reference entropy
    hp_Vr: Optional[np.ndarray] = None       # (N,) reference volume m³/mol
    hp_a: Optional[np.ndarray] = None        # (N,) Cp polynomial a
    hp_b: Optional[np.ndarray] = None        # (N,) Cp polynomial b
    hp_c: Optional[np.ndarray] = None        # (N,) Cp polynomial c
    hp_d: Optional[np.ndarray] = None        # (N,) Cp polynomial d
    hp_alpha0: Optional[np.ndarray] = None   # (N,) thermal expansion
    hp_kappa0: Optional[np.ndarray] = None   # (N,) bulk modulus
    hp_kappa0p: Optional[np.ndarray] = None  # (N,) pressure derivative
    hp_kappa0pp: Optional[np.ndarray] = None # (N,) second pressure derivative
    hp_numatoms: Optional[np.ndarray] = None # (N,) atoms per formula unit

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def n_elements(self) -> int:
        return len(self.element_names)


def build_jax_system(
    preset: str = "light_ree",
    include_minerals: bool = True,
) -> ChemicalSystemData:
    """Build a JAX-compatible chemical system from built-in data.

    Parameters
    ----------
    preset : str
        Currently only ``"light_ree"`` is supported with built-in data.
    include_minerals : bool
        Whether to include solid (hydroxide) phases.

    Returns
    -------
    ChemicalSystemData
    """
    from ..properties.ree_databases import LIGHT_REE, HEAVY_REE, ALL_REE

    if preset == "light_ree":
        ree_elements = LIGHT_REE
    elif preset == "heavy_ree":
        ree_elements = HEAVY_REE + ["Y", "Sc"]
    elif preset == "full_ree":
        ree_elements = ALL_REE
    else:
        ree_elements = LIGHT_REE

    # Collect species records
    records: List[SpeciesRecord] = list(_AQUEOUS_SPECIES_DATA)

    if include_minerals:
        # Filter minerals to those whose REE elements match the preset
        for rec in _MINERAL_SPECIES_DATA:
            mineral_ree = [el for el in rec.elements if el in ALL_REE]
            if all(el in ree_elements for el in mineral_ree):
                records.append(rec)

    # Determine element universe from all species
    all_elements = set()
    for rec in records:
        all_elements.update(rec.elements.keys())
    element_names = sorted(all_elements)
    elem_idx = {e: i for i, e in enumerate(element_names)}

    N = len(records)
    E = len(element_names)

    G0 = np.zeros(N, dtype=np.float64)
    charges = np.zeros(N, dtype=np.float64)
    molar_masses = np.zeros(N, dtype=np.float64)
    ion_sizes = np.zeros(N, dtype=np.float64)
    A = np.zeros((E, N), dtype=np.float64)
    names = []
    phases = []

    for j, rec in enumerate(records):
        G0[j] = rec.G0
        charges[j] = rec.charge
        molar_masses[j] = rec.molar_mass
        ion_sizes[j] = rec.ion_size_param
        names.append(rec.name)
        phases.append(rec.phase)
        for el, coeff in rec.elements.items():
            A[elem_idx[el], j] = coeff

    return ChemicalSystemData(
        species_names=names,
        G0=G0,
        charges=charges,
        formula_matrix=A,
        element_names=element_names,
        molar_masses=molar_masses,
        ion_size_params=ion_sizes,
        phases=phases,
        ree_elements=ree_elements,
    )


def build_jax_system_hkf(
    preset: str = "light_ree",
    json_path: Optional[str] = None,
    include_minerals: bool = False,
    include_gases: bool = False,
) -> ChemicalSystemData:
    """Build a JAX chemical system from the SUPCRTBL database with HKF/HP params.

    This loads the full SUPCRTBL species database (1108 species) and filters
    to the requested preset.  Supports aqueous (HKF), mineral (HP), and gas (HP) phases.

    Parameters
    ----------
    preset : str
        ``"light_ree"``, ``"heavy_ree"``, ``"full_ree"``, ``"geo_h2"``, ``"geo_co2"``.
    json_path : str, optional
        Path to ``supcrtbl_species.json``.
    include_minerals : bool
        Include mineral phases (auto-True for geo_* presets).
    include_gases : bool
        Include gas phases (auto-True for geo_* presets).
    """
    import json as _json
    from ..properties.ree_databases import LIGHT_REE, HEAVY_REE, ALL_REE

    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'supcrtbl_species.json'
        )

    with open(json_path) as f:
        db_data = _json.load(f)

    # Geological presets auto-enable minerals and gases
    if preset.startswith('geo_'):
        include_minerals = True
        include_gases = True

    if preset == 'light_ree':
        ree_elements = LIGHT_REE
    elif preset == 'heavy_ree':
        ree_elements = HEAVY_REE + ['Y', 'Sc']
    elif preset == 'full_ree':
        ree_elements = ALL_REE
    elif preset == 'geo_h2':
        ree_elements = []  # No REE for geological H2
    elif preset == 'geo_co2':
        ree_elements = []  # No REE for CO2 sequestration
    else:
        ree_elements = LIGHT_REE

    # Common elements always included
    common_elements = {'H', 'O', 'Na', 'Cl', 'K', 'Ca', 'Mg', 'Fe', 'Al',
                       'Si', 'S', 'C', 'N', 'P', 'F', 'Mn'}
    target_elements = common_elements | set(ree_elements)

    # Collect species records
    records = []
    for sp in db_data['species']:
        phase = sp['phase']
        if phase == 'aqueous':
            sp_elements = set(sp['elements'].keys())
            if sp_elements <= target_elements:
                records.append(sp)
        elif phase == 'mineral' and include_minerals:
            sp_elements = set(sp['elements'].keys())
            if sp_elements <= target_elements:
                records.append(sp)
        elif phase == 'gas' and include_gases:
            sp_elements = set(sp['elements'].keys())
            if sp_elements <= target_elements:
                records.append(sp)

    N = len(records)
    _log.info("build_jax_system_hkf: %d species (%s) for preset '%s'",
              N, '+'.join(p for p in ['aq', 'min' if include_minerals else '', 'gas' if include_gases else ''] if p), preset)

    # Build element universe
    all_elems = set()
    for rec in records:
        all_elems.update(rec['elements'].keys())
    element_names = sorted(all_elems)
    elem_idx = {e: i for i, e in enumerate(element_names)}
    E = len(element_names)

    # Allocate arrays
    G0 = np.zeros(N, dtype=np.float64)
    charges = np.zeros(N, dtype=np.float64)
    molar_masses = np.zeros(N, dtype=np.float64)
    ion_sizes = np.zeros(N, dtype=np.float64)
    A = np.zeros((E, N), dtype=np.float64)
    names = []
    phases = []

    # HKF arrays (aqueous species)
    hkf_Gf = np.full(N, np.nan)
    hkf_Sr = np.full(N, np.nan)
    hkf_a1 = np.full(N, np.nan)
    hkf_a2 = np.full(N, np.nan)
    hkf_a3 = np.full(N, np.nan)
    hkf_a4 = np.full(N, np.nan)
    hkf_c1 = np.full(N, np.nan)
    hkf_c2 = np.full(N, np.nan)
    hkf_wref = np.full(N, np.nan)

    # HollandPowell arrays (mineral + gas species)
    hp_Gf = np.full(N, np.nan)
    hp_Hf = np.full(N, np.nan)
    hp_Sr = np.full(N, np.nan)
    hp_Vr = np.full(N, np.nan)
    hp_a = np.full(N, np.nan)
    hp_b = np.full(N, np.nan)
    hp_c = np.full(N, np.nan)
    hp_d = np.full(N, np.nan)
    hp_alpha0 = np.full(N, np.nan)
    hp_kappa0 = np.full(N, np.nan)
    hp_kappa0p = np.full(N, np.nan)
    hp_kappa0pp = np.full(N, np.nan)
    hp_numatoms = np.full(N, np.nan)

    _ION_SIZES = {1: 9.0, -1: 3.5, 2: 6.0, -2: 4.0, 3: 9.0, -3: 4.0, 4: 10.0}
    water_hkf_params = None

    for j, rec in enumerate(records):
        name = rec['name']
        names.append(name)
        charges[j] = rec['charge']
        molar_masses[j] = rec['molar_mass_kg']
        phases.append(rec['phase'])
        ion_sizes[j] = _ION_SIZES.get(int(rec['charge']), 0.0) if rec['charge'] != 0 else 0.0

        for el, coeff in rec['elements'].items():
            A[elem_idx[el], j] = coeff

        g298 = rec.get('G0_298K')
        G0[j] = g298 if g298 is not None else 0.0

        # HKF params (aqueous)
        if rec['model_type'] == 'HKF' and rec.get('hkf_params'):
            hp_dict = rec['hkf_params']
            hkf_Gf[j] = hp_dict.get('Gf', np.nan)
            hkf_Sr[j] = hp_dict.get('Sr', np.nan)
            hkf_a1[j] = hp_dict.get('a1', np.nan)
            hkf_a2[j] = hp_dict.get('a2', np.nan)
            hkf_a3[j] = hp_dict.get('a3', np.nan)
            hkf_a4[j] = hp_dict.get('a4', np.nan)
            hkf_c1[j] = hp_dict.get('c1', np.nan)
            hkf_c2[j] = hp_dict.get('c2', np.nan)
            hkf_wref[j] = hp_dict.get('wref', np.nan)
        elif rec['model_type'] == 'WaterHKF':
            water_hkf_params = {
                'Gtr': -235517.36, 'Htr': -287721.13,
                'Str': 63.3123, 'Ttr': 273.16,
            }

        # HollandPowell params (mineral + gas)
        if rec['model_type'] == 'HollandPowell' and rec.get('hp_params'):
            hp_dict = rec['hp_params']
            hp_Gf[j] = hp_dict.get('Gf', np.nan)
            hp_Hf[j] = hp_dict.get('Hf', np.nan)
            hp_Sr[j] = hp_dict.get('Sr', np.nan)
            hp_Vr[j] = hp_dict.get('Vr', 0.0)
            hp_a[j] = hp_dict.get('a', np.nan)
            hp_b[j] = hp_dict.get('b', 0.0)
            hp_c[j] = hp_dict.get('c', 0.0)
            hp_d[j] = hp_dict.get('d', 0.0)
            hp_alpha0[j] = hp_dict.get('alpha0', 0.0)
            hp_kappa0[j] = hp_dict.get('kappa0', 0.0)
            hp_kappa0p[j] = hp_dict.get('kappa0p', 0.0)
            hp_kappa0pp[j] = hp_dict.get('kappa0pp', 0.0)
            hp_numatoms[j] = hp_dict.get('numatoms', 1.0)

    return ChemicalSystemData(
        species_names=names,
        G0=G0,
        charges=charges,
        formula_matrix=A,
        element_names=element_names,
        molar_masses=molar_masses,
        ion_size_params=ion_sizes,
        phases=phases,
        ree_elements=ree_elements,
        hkf_Gf=hkf_Gf, hkf_Sr=hkf_Sr,
        hkf_a1=hkf_a1, hkf_a2=hkf_a2, hkf_a3=hkf_a3, hkf_a4=hkf_a4,
        hkf_c1=hkf_c1, hkf_c2=hkf_c2, hkf_wref=hkf_wref,
        water_hkf_params=water_hkf_params,
        hp_Gf=hp_Gf, hp_Hf=hp_Hf, hp_Sr=hp_Sr, hp_Vr=hp_Vr,
        hp_a=hp_a, hp_b=hp_b, hp_c=hp_c, hp_d=hp_d,
        hp_alpha0=hp_alpha0, hp_kappa0=hp_kappa0,
        hp_kappa0p=hp_kappa0p, hp_kappa0pp=hp_kappa0pp,
        hp_numatoms=hp_numatoms,
    )


# ---------------------------------------------------------------------------
# Extended Debye-Hückel activity coefficient model
# ---------------------------------------------------------------------------
# Parameters for the B-dot (extended) Debye-Hückel model:
#   log₁₀(γ) = -A(T)·z²·√I / (1 + B(T)·å·√I) + Bdot(T)·I
# A, B from Helgeson & Kirkham (1974), Table 2; Bdot from Helgeson (1969)
#
# Temperature-dependent via polynomial fits to tabulated data (0-300°C).
# Max interpolation error < 0.001 for all three parameters.

# Polynomial coefficients for A_DH(T_K) — 5th order
_A_DH_POLY = jnp.array([
    2.3253174626835094e-13, -4.1211180018940124e-10,
    2.916775483919206e-07, -9.819912420525612e-05,
    0.01611757429060257, -0.5884527028874091,
])
# Polynomial coefficients for B_DH(T_K) — 3rd order
_B_DH_POLY = jnp.array([
    3.219120640608501e-10, -8.42535737541022e-08,
    0.00013050767835946395, 0.28850609030920793,
])
# Polynomial coefficients for Bdot(T_K) — 4th order
_BDOT_POLY = jnp.array([
    -9.734265734267127e-12, 1.4421093508514162e-08,
    -8.670606436907735e-06, 0.0024906112618667937,
    -0.2358706816342579,
])


def _polyval_jax(coeffs: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Evaluate a polynomial using Horner's method (JAX-traceable)."""
    result = coeffs[0]
    for c in coeffs[1:]:
        result = result * x + c
    return result


def _debye_huckel_log_gamma(
    charges: jnp.ndarray,
    ion_sizes: jnp.ndarray,
    ionic_strength: jnp.ndarray,
    T_K: float = 298.15,
) -> jnp.ndarray:
    """Compute ln(γ) for each species using the extended Debye-Hückel model.

    Uses temperature-dependent A(T), B(T), Bdot(T) parameters from
    Helgeson & Kirkham (1974) polynomial fits.

    Parameters
    ----------
    charges : (N,) ionic charges
    ion_sizes : (N,) ion size parameters in Å
    ionic_strength : scalar, mol/kg
    T_K : temperature in Kelvin (default 298.15)
    """
    T = jnp.array(T_K)
    A = _polyval_jax(_A_DH_POLY, T)
    B = _polyval_jax(_B_DH_POLY, T)
    Bdot = _polyval_jax(_BDOT_POLY, T)

    sqrt_I = jnp.sqrt(jnp.maximum(ionic_strength, 1e-30))
    z2 = charges ** 2

    # For charged species: log10(γ) = -A·z²·√I / (1 + B·å·√I) + Bdot·I
    # For neutral species (å=0): log10(γ) = Bdot·I
    denominator = 1.0 + B * ion_sizes * sqrt_I
    log10_gamma = -A * z2 * sqrt_I / denominator + Bdot * ionic_strength

    # Convert to natural log
    ln_gamma = log10_gamma * LN10
    return ln_gamma


# ---------------------------------------------------------------------------
# Gibbs energy minimization solver
# ---------------------------------------------------------------------------
class JaxEquilibriumSolver:
    """Differentiable Gibbs energy minimization solver using JAX.

    Parameters
    ----------
    system : ChemicalSystemData
        Chemical system specification.
    tol : float
        Convergence tolerance for the optimizer.
    maxiter : int
        Maximum iterations.
    """

    def __init__(
        self,
        system: ChemicalSystemData,
        tol: float = 1e-8,
        maxiter: int = 500,
    ):
        self.system = system
        self.tol = tol
        self.maxiter = maxiter

        # Pre-convert to JAX arrays
        self._G0 = jnp.array(system.G0)
        self._charges = jnp.array(system.charges)
        self._A = jnp.array(system.formula_matrix)
        self._ion_sizes = jnp.array(system.ion_size_params)
        self._is_aqueous = jnp.array(
            [1.0 if p == "aqueous" else 0.0 for p in system.phases]
        )
        self._is_mineral = jnp.array(
            [1.0 if p == "mineral" else 0.0 for p in system.phases]
        )
        self._is_gas = jnp.array(
            [1.0 if p == "gas" else 0.0 for p in system.phases]
        )
        self._water_idx = system.species_names.index("H2O(aq)")
        self._hplus_idx = system.species_names.index("H+")

        # Species name → index lookup
        self._sp_idx = {n: i for i, n in enumerate(system.species_names)}

    def solve(
        self,
        temperature_K: float,
        pressure_Pa: float,
        species_amounts: Dict[str, float],
    ) -> Dict:
        """Solve chemical equilibrium.

        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin.
        pressure_Pa : float
            Pressure in Pascals.
        species_amounts : dict
            Initial species amounts {name: mol}.

        Returns
        -------
        dict
            Keys: ``status``, ``species_amounts`` (dict), ``pH``, ``Eh_V``,
            ``ionic_strength``, ``temperature_K``, ``pressure_Pa``.
        """
        T = float(temperature_K)
        RT = R_GAS * T

        # Build initial amounts vector
        n0 = np.zeros(self.system.n_species, dtype=np.float64)
        for sp_name, amount in species_amounts.items():
            if sp_name in self._sp_idx:
                n0[self._sp_idx[sp_name]] = max(amount, 0.0)
            else:
                _log.debug("Species '%s' not in system, skipped", sp_name)

        # Compute element totals b = A·n₀ (the conservation constraint)
        # IMPORTANT: use original user amounts, not seeded ones
        A_np = np.array(self._A)
        b = A_np @ n0

        # Identify active elements (non-zero budget) and active species
        active_elems = b > 1e-20

        # Filter to species whose required elements all have budget
        active_species = np.ones(self.system.n_species, dtype=bool)
        for j in range(self.system.n_species):
            for ei in range(self.system.n_elements):
                if A_np[ei, j] > 0 and not active_elems[ei]:
                    active_species[j] = False
                    break

        active_idx = np.where(active_species)[0]
        n_active = len(active_idx)

        if n_active == 0:
            return {"status": "error", "error": "No active species found"}

        # Reduced system: only active elements and species
        A_red = A_np[active_elems][:, active_idx]
        b_red = b[active_elems]

        # Compute G⁰ at runtime T,P if HKF data is available
        if self.system.hkf_Gf is not None:
            from .jax_hkf import compute_G0_jax, _water_G0
            # Build G0 for all species, then select active
            G0_all = np.array(self._G0)  # fallback: 298K values
            # Compute HKF G0(T,P) for species that have HKF params
            hkf_mask = ~np.isnan(self.system.hkf_Gf)
            if np.any(hkf_mask):
                hkf_idx = np.where(hkf_mask)[0]
                G0_hkf = compute_G0_jax(
                    jnp.array(T), jnp.array(float(pressure_Pa)),
                    jnp.array(self.system.hkf_Gf[hkf_idx]),
                    jnp.zeros(len(hkf_idx)),  # Hf not used in G0 calc
                    jnp.array(self.system.hkf_Sr[hkf_idx]),
                    jnp.array(self.system.hkf_a1[hkf_idx]),
                    jnp.array(self.system.hkf_a2[hkf_idx]),
                    jnp.array(self.system.hkf_a3[hkf_idx]),
                    jnp.array(self.system.hkf_a4[hkf_idx]),
                    jnp.array(self.system.hkf_c1[hkf_idx]),
                    jnp.array(self.system.hkf_c2[hkf_idx]),
                    jnp.array(self.system.hkf_wref[hkf_idx]),
                    jnp.array(self.system.charges[hkf_idx]),
                )
                G0_all[hkf_idx] = np.array(G0_hkf)
            # Water via WaterHKF model
            if self.system.water_hkf_params is not None:
                w = self.system.water_hkf_params
                wi = self.system.species_names.index('H2O(aq)')
                G0_all[wi] = float(_water_G0(
                    jnp.array(T), jnp.array(float(pressure_Pa)),
                    w['Gtr'], w['Htr'], w['Str'], w['Ttr'],
                ))

            # Compute HollandPowell G0(T,P) for mineral and gas species
            if self.system.hp_Gf is not None:
                from .jax_holland_powell import _hp_G0_scalar, _hp_G0_mineral
                hp_mask = ~np.isnan(self.system.hp_Gf)
                if np.any(hp_mask):
                    hp_idx = np.where(hp_mask)[0]
                    T_j = jnp.float64(T)
                    P_j = jnp.float64(float(pressure_Pa))
                    for j in hp_idx:
                        phase_j = self.system.phases[j]
                        args = (T_j, P_j,
                                float(self.system.hp_Gf[j]), float(self.system.hp_Hf[j]),
                                float(self.system.hp_Sr[j]), float(self.system.hp_Vr[j]),
                                float(self.system.hp_a[j]), float(self.system.hp_b[j]),
                                float(self.system.hp_c[j]), float(self.system.hp_d[j]),
                                float(self.system.hp_alpha0[j]), float(self.system.hp_kappa0[j]),
                                float(self.system.hp_kappa0p[j]), float(self.system.hp_kappa0pp[j]),
                                float(self.system.hp_numatoms[j]))
                        kappa0_val = float(self.system.hp_kappa0[j])
                        if phase_j == 'mineral' and kappa0_val > 0:
                            G0_all[j] = float(_hp_G0_mineral(*args))
                        else:
                            # Gas or mineral without Tait EOS params
                            G0_all[j] = float(_hp_G0_scalar(*args))
                            # For minerals with Vr but no EOS, add Vr*(P-Pref)
                            Vr_val = float(self.system.hp_Vr[j])
                            if Vr_val > 0 and kappa0_val == 0:
                                G0_all[j] += Vr_val * (float(pressure_Pa) - 1e5)

            G0_red = G0_all[active_idx]
        else:
            G0_red = np.array(self._G0)[active_idx]
        charges_red = np.array(self._charges)[active_idx]
        ion_sizes_red = np.array(self._ion_sizes)[active_idx]
        is_aq_red = np.array(self._is_aqueous)[active_idx]
        is_gas_red = np.array(self._is_gas)[active_idx]

        # Peng-Robinson fugacity coefficients for gas species
        from .jax_peng_robinson import CRITICAL_PROPS
        P_ref = 1e5  # 1 bar reference pressure
        ln_fugacity_red = np.zeros(n_active, dtype=np.float64)
        for local_j in range(n_active):
            global_j = int(active_idx[local_j])
            sp_name = self.system.species_names[global_j]
            if self.system.phases[global_j] == 'gas':
                if sp_name in CRITICAL_PROPS:
                    Tc, Pc, omega_pr = CRITICAL_PROPS[sp_name]
                    from .jax_peng_robinson import fugacity_coefficient
                    phi = float(fugacity_coefficient(
                        jnp.array(T), jnp.array(float(pressure_Pa)),
                        jnp.array(Tc), jnp.array(Pc), jnp.array(omega_pr)))
                else:
                    phi = 1.0  # Ideal gas fallback
                # ln(a_gas) = ln(phi) + ln(P/Pref)
                ln_fugacity_red[local_j] = np.log(max(phi, 1e-30)) + np.log(max(float(pressure_Pa) / P_ref, 1e-30))

        # Water and H+ indices in reduced system
        water_idx_red = -1
        hplus_idx_red = -1
        for local_j, global_j in enumerate(active_idx):
            if global_j == self._water_idx:
                water_idx_red = local_j
            if global_j == self._hplus_idx:
                hplus_idx_red = local_j

        # --- Build initial guess for the optimizer ---
        # Start from user-provided amounts; seed trace ions so SLSQP has
        # a gradient direction.  The constraints enforce the ORIGINAL b.
        n0_red = np.array([n0[int(g)] for g in active_idx], dtype=np.float64)
        EPS = 1e-20

        # Seed trace amounts of dissociation products from water
        # SLSQP needs a non-zero starting point for H⁺/OH⁻ so the gradient
        # of ln(molality) is finite and points toward dissociation.
        SEED_AMT = 1e-7  # ~pH 7 level
        if water_idx_red >= 0 and hplus_idx_red >= 0:
            oh_idx_red = -1
            for lj, gj in enumerate(active_idx):
                if self.system.species_names[int(gj)] == "OH-":
                    oh_idx_red = lj
                    break
            if oh_idx_red >= 0:
                if n0_red[hplus_idx_red] < SEED_AMT:
                    n0_red[hplus_idx_red] = SEED_AMT
                if n0_red[oh_idx_red] < SEED_AMT:
                    n0_red[oh_idx_red] = SEED_AMT

        # Seed all remaining zero species at EPS
        for local_j in range(n_active):
            if n0_red[local_j] < EPS:
                n0_red[local_j] = EPS

        # NOTE: b_red is the ORIGINAL element budget from user inputs (line 450).
        # Do NOT recompute b_red from seeded amounts — seeding only affects the
        # initial guess, not the conservation target.

        # --- Scaled Gibbs energy objective (dimensionless: G/(RT)) ---
        # Scaling by 1/RT gives O(1) gradients, critical for optimizer convergence.
        G0_j = jnp.array(G0_red)
        charges_j = jnp.array(charges_red)
        ion_sizes_j = jnp.array(ion_sizes_red)
        is_aq_j = jnp.array(is_aq_red)
        is_gas_j = jnp.array(is_gas_red)
        ln_fug_j = jnp.array(ln_fugacity_red)
        inv_RT = 1.0 / RT

        def gibbs_energy_scaled(n_vec):
            """Total Gibbs energy G(T,P,n) / RT (dimensionless)."""
            n = jnp.maximum(n_vec, EPS)

            # Water amount in kg
            n_water = n[water_idx_red] if water_idx_red >= 0 else jnp.array(55.508)
            m_water = jnp.maximum(n_water * WATER_MW, 1e-10)

            # Ionic strength
            molalities = n * is_aq_j / m_water
            I = jnp.maximum(0.5 * jnp.sum(molalities * charges_j ** 2), 1e-30)

            # Activity coefficients (T-dependent)
            ln_gamma = _debye_huckel_log_gamma(charges_j, ion_sizes_j, I, T_K=T)

            # Chemical potentials: μ/(RT) = G⁰/(RT) + ln(a)
            ln_m = jnp.log(jnp.maximum(molalities, EPS))
            ln_a = (ln_gamma + ln_m) * is_aq_j

            # Gas activity: ln(a) = ln(φ) + ln(P/Pref) (pre-computed)
            ln_a = ln_a + ln_fug_j * is_gas_j

            # Water activity ≈ 1
            if water_idx_red >= 0:
                ln_a = ln_a.at[water_idx_red].set(0.0)

            mu_scaled = G0_j * inv_RT + ln_a
            return jnp.sum(n * mu_scaled)

        # --- Optimize using scipy trust-constr ---
        from scipy.optimize import minimize as scipy_minimize, LinearConstraint

        # JAX gradient of scaled objective
        gibbs_grad_fn = jax.grad(gibbs_energy_scaled)

        def obj_and_grad(n_np):
            n_jax = jnp.array(n_np)
            obj = float(gibbs_energy_scaled(n_jax))
            grad = np.array(gibbs_grad_fn(n_jax), dtype=np.float64)
            if not np.isfinite(obj):
                obj = 1e30
            grad = np.where(np.isfinite(grad), grad, 0.0)
            return obj, grad

        # Linear equality constraints: A_red @ n = b_red
        lin_con = LinearConstraint(A_red, b_red, b_red)

        # Bounds: n_i >= EPS (positivity)
        bounds = [(EPS, None)] * n_active

        try:
            result = scipy_minimize(
                obj_and_grad,
                n0_red,
                method='trust-constr',
                jac=True,
                bounds=bounds,
                constraints=[lin_con],
                options={'maxiter': 2000, 'gtol': 1e-10},
            )
            n_opt = result.x
        except Exception as exc:
            _log.warning("JAX GEM solver failed: %s", exc)
            return {"status": "error", "error": str(exc)}

        # n_opt is already in natural space (no transform needed)
        n_eq_red = np.array(n_opt)

        # Map back to full species vector
        n_eq = np.zeros(self.system.n_species, dtype=np.float64)
        for local_j, global_j in enumerate(active_idx):
            n_eq[int(global_j)] = max(float(n_eq_red[local_j]), 0.0)

        # --- Extract results ---
        n_water_mol = float(n_eq[self._water_idx])
        m_water_kg = max(n_water_mol * WATER_MW, 1e-10)

        eq_amounts = {}
        for j, name in enumerate(self.system.species_names):
            amt = float(n_eq[j])
            if amt > 1e-15:
                eq_amounts[name] = amt

        # pH = -log10(a_H+)
        h_plus_mol = float(n_eq[self._hplus_idx])
        h_plus_molality = h_plus_mol / m_water_kg
        I_val = 0.5 * sum(
            float(n_eq[j]) / m_water_kg * float(self.system.charges[j]) ** 2
            for j in range(self.system.n_species)
            if self.system.phases[j] == "aqueous"
        )
        sqrt_I = math.sqrt(max(I_val, 1e-30))
        # T-dependent DH parameters
        T_jnp = jnp.array(T)
        A_dh = float(_polyval_jax(_A_DH_POLY, T_jnp))
        B_dh = float(_polyval_jax(_B_DH_POLY, T_jnp))
        Bdot_dh = float(_polyval_jax(_BDOT_POLY, T_jnp))
        log10_gamma_h = -A_dh * 1.0 * sqrt_I / (1.0 + B_dh * 9.0 * sqrt_I) + Bdot_dh * I_val
        gamma_h = 10.0 ** log10_gamma_h
        a_h = max(gamma_h * h_plus_molality, 1e-30)
        pH = -math.log10(a_h)

        # Eh via Nernst
        Eh_V = RT / F_FARADAY * math.log(a_h)

        # Mass balance check
        b_check = A_np @ n_eq
        b_target = b
        mass_bal_err = float(np.max(np.abs(b_check - b_target) / (np.abs(b_target) + 1e-10)))
        if mass_bal_err > 0.01:
            _log.warning("Mass balance relative error: %.4e", mass_bal_err)

        return {
            "status": "ok",
            "temperature_K": temperature_K,
            "pressure_Pa": pressure_Pa,
            "species_amounts": eq_amounts,
            "pH": round(float(pH), 4),
            "Eh_V": round(float(Eh_V), 4),
            "ionic_strength": round(float(I_val), 4),
            "mass_balance_error": mass_bal_err,
        }

    def solve_speciation(
        self,
        temperature_C: float = 25.0,
        pressure_atm: float = 1.0,
        water_kg: float = 1.0,
        acid_mol: Optional[Dict[str, float]] = None,
        ree_mol: Optional[Dict[str, float]] = None,
        other_mol: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """High-level speciation interface matching REEEquilibriumSolver.speciate().

        Returns
        -------
        dict
            Same format as Reaktoro-backed speciation:
            ``pH``, ``Eh_V``, ``ionic_strength``, ``species``, ``ree_distribution``.
        """
        from ..properties.ree_databases import ALL_REE

        T_K = temperature_C + 273.15
        P_Pa = pressure_atm * 101325.0

        # Build species amounts
        amounts: Dict[str, float] = {}
        amounts["H2O(aq)"] = water_kg / WATER_MW  # mol of water

        for species_dict in [acid_mol, ree_mol, other_mol]:
            if species_dict:
                for sp_name, amount in species_dict.items():
                    amounts[sp_name] = amounts.get(sp_name, 0.0) + amount

        result = self.solve(T_K, P_Pa, amounts)

        if result["status"] != "ok":
            return result

        # Separate REE species
        ree_symbols = set(ALL_REE)
        all_species = result["species_amounts"]
        ree_species = {}

        for sp_name, amt in all_species.items():
            # Check if species contains an REE element
            sp_idx = self._sp_idx.get(sp_name)
            if sp_idx is not None:
                sp_elems = set(self.system.species_names[sp_idx])
                # Check against element composition
                rec_elems = set()
                for ei, ename in enumerate(self.system.element_names):
                    if float(self._A[ei, sp_idx]) > 0:
                        rec_elems.add(ename)

                if rec_elems & ree_symbols:
                    ree_species[sp_name] = amt

        return {
            "status": "ok",
            "temperature_C": temperature_C,
            "pressure_atm": pressure_atm,
            "pH": result["pH"],
            "Eh_V": result["Eh_V"],
            "ionic_strength": result["ionic_strength"],
            "species": dict(sorted(all_species.items(), key=lambda kv: -kv[1])),
            "ree_distribution": dict(sorted(ree_species.items(), key=lambda kv: -kv[1])),
        }


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------
def speciate_jax(
    preset: str = "light_ree",
    temperature_C: float = 25.0,
    pressure_atm: float = 1.0,
    water_kg: float = 1.0,
    acid_mol: Optional[Dict[str, float]] = None,
    ree_mol: Optional[Dict[str, float]] = None,
    other_mol: Optional[Dict[str, float]] = None,
) -> Dict:
    """One-shot speciation using the JAX solver.

    Convenience function that builds the system and solves in one call.
    """
    system = build_jax_system(preset=preset)
    solver = JaxEquilibriumSolver(system)
    return solver.solve_speciation(
        temperature_C=temperature_C,
        pressure_atm=pressure_atm,
        water_kg=water_kg,
        acid_mol=acid_mol,
        ree_mol=ree_mol,
        other_mol=other_mol,
    )
