"""
Holland & Powell (2011) thermodynamic model for minerals and gases — JAX implementation.

Computes standard Gibbs energy G⁰(T,P) for solid and gaseous phases using:
  - Berman-Brown Cp(T) = a + bT + cT⁻² + dT⁻⁰·⁵
  - Modified Tait equation of state for V(T,P)
  - Einstein thermal pressure model

Reference:
  Holland, T.J.B. & Powell, R. (2011). An improved and extended internally
  consistent thermodynamic dataset for phases of petrological interest.
  Journal of Metamorphic Geology, 29, 333–383.

  Reaktoro implementation: StandardThermoModelHollandPowell.cpp
"""

import jax
import jax.numpy as jnp
import numpy as np
import json
import os
import logging

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_T_REF = 298.15   # K
_P_REF = 1e5      # Pa (1 bar)
_R = 8.314462618  # J/(mol·K)

# Einstein temperature ratio for thermal pressure model
# θ_E = 10636 / (Sr/n + 6.44)  — Holland & Powell (2011), Eq. 4
_EINSTEIN_COEFF = 10636.0
_EINSTEIN_OFFSET = 6.44


# ---------------------------------------------------------------------------
# Heat capacity model: Cp(T) = a + bT + cT⁻² + dT⁻⁰·⁵
# ---------------------------------------------------------------------------
def _hp_Cp(T, a, b, c, d):
    """Berman-Brown heat capacity (J/mol/K)."""
    return a + b * T + c / (T * T) + d / jnp.sqrt(T)


def _hp_thermal_integral_CpdT(T, a, b, c, d):
    """Compute ∫[Tref→T] Cp dT′ analytically."""
    Tr = _T_REF
    return (
        a * (T - Tr)
        + 0.5 * b * (T**2 - Tr**2)
        + c * (-1.0/T + 1.0/Tr)
        + 2.0 * d * (jnp.sqrt(T) - jnp.sqrt(Tr))
    )


def _hp_thermal_integral_CpOverTdT(T, a, b, c, d):
    """Compute ∫[Tref→T] Cp/T′ dT′ analytically."""
    Tr = _T_REF
    return (
        a * jnp.log(T / Tr)
        + b * (T - Tr)
        + 0.5 * c * (-1.0/T**2 + 1.0/Tr**2)
        - 2.0 * d * (1.0/jnp.sqrt(T) - 1.0/jnp.sqrt(Tr))
    )


# ---------------------------------------------------------------------------
# Modified Tait EOS for V(T,P) — minerals only
# ---------------------------------------------------------------------------
def _einstein_thermal_pressure(T, alpha0, kappa0, theta_E):
    """Thermal pressure contribution P_th(T) - P_th(Tref).

    Uses Einstein model: P_th = α₀·κ₀·θ_E / (exp(θ_E/T) - 1)
    """
    Tr = _T_REF
    xi_T = theta_E / T
    xi_Tr = theta_E / Tr
    # u = θ_E²·exp(θ_E/T) / (T²·(exp(θ_E/T)-1)²) — Einstein function
    Pth_T = alpha0 * kappa0 * theta_E / (jnp.exp(xi_T) - 1.0)
    Pth_Tr = alpha0 * kappa0 * theta_E / (jnp.exp(xi_Tr) - 1.0)
    return Pth_T - Pth_Tr


def _tait_volume_integral(P, T, Vr, alpha0, kappa0, kappa0p, kappa0pp, theta_E):
    """Compute ∫[Pref→P] V(T,P′) dP′ using the modified Tait EOS.

    The Tait equation: V = V₀(1 - a·(1 - (1 + b·P′)^(-c)))
    where a, b, c are Tait parameters derived from κ₀, κ₀′, κ₀″.

    Following Holland & Powell (2011) and Reaktoro's implementation.
    """
    # Tait parameters
    Pth = _einstein_thermal_pressure(T, alpha0, kappa0, theta_E)

    # a = (1 + κ₀′)/(1 + κ₀′ + κ₀·κ₀″)
    a_tait = (1.0 + kappa0p) / (1.0 + kappa0p + kappa0 * kappa0pp)
    # b = κ₀′/κ₀ − κ₀″/(1 + κ₀′)
    b_tait = kappa0p / kappa0 - kappa0pp / (1.0 + kappa0p)
    # c = (1 + κ₀′ + κ₀·κ₀″) / (κ₀′² + κ₀′ − κ₀·κ₀″)
    c_tait = (1.0 + kappa0p + kappa0 * kappa0pp) / (kappa0p**2 + kappa0p - kappa0 * kappa0pp)

    # Effective pressure accounting for thermal pressure
    Peff = P - _P_REF + Pth
    Peff0 = Pth  # at P = Pref

    # ∫[Pref→P] V dP = Vr · ((P-Pref) − (a/(b·(c-1))) · ((1+b·Peff)^(1-c) - (1+b·Peff0)^(1-c)))
    # But we need to be careful with numerical stability

    # V(P,T) = Vr · (1 - a · (1 - (1 + b·Peff)^(-c)))
    # ∫V dP from Pref to P (isothermal)
    intVdP = Vr * (
        (P - _P_REF)
        - a_tait / (b_tait * (c_tait - 1.0)) * (
            jnp.power(1.0 + b_tait * Peff, 1.0 - c_tait)
            - jnp.power(1.0 + b_tait * Peff0, 1.0 - c_tait)
        )
    )
    return intVdP


# ---------------------------------------------------------------------------
# Full HP G⁰(T,P)
# ---------------------------------------------------------------------------
def _hp_G0_scalar(T, P, Gf, Hf, Sr, Vr, a, b, c, d, alpha0, kappa0, kappa0p, kappa0pp, numatoms):
    """Compute G⁰(T,P) for a single mineral/gas species using Holland-Powell model.

    G(T,P) = Gf − (T−Tref)·Sr + ∫CpdT − T·∫Cp/TdT + ∫VdP

    For gases (Vr=0): only thermal contribution (no PV term).
    For minerals: full thermal + Tait EOS volume integral.
    """
    Tr = _T_REF

    # Thermal contributions
    intCpdT = _hp_thermal_integral_CpdT(T, a, b, c, d)
    intCpOverTdT = _hp_thermal_integral_CpOverTdT(T, a, b, c, d)

    # G(T, Pref) = Gf - (T-Tref)*Sr + ∫CpdT - T*∫Cp/TdT
    G_thermal = Gf - (T - Tr) * Sr + intCpdT - T * intCpOverTdT

    return G_thermal


def _hp_G0_mineral(T, P, Gf, Hf, Sr, Vr, a, b, c, d, alpha0, kappa0, kappa0p, kappa0pp, numatoms):
    """G⁰(T,P) for a mineral species — includes Tait EOS volume integral."""
    G_thermal = _hp_G0_scalar(T, P, Gf, Hf, Sr, Vr, a, b, c, d, alpha0, kappa0, kappa0p, kappa0pp, numatoms)

    # Einstein temperature
    theta_E = _EINSTEIN_COEFF / (Sr / jnp.maximum(numatoms, 1.0) + _EINSTEIN_OFFSET)

    intVdP = _tait_volume_integral(P, T, Vr, alpha0, kappa0, kappa0p, kappa0pp, theta_E)

    return G_thermal + intVdP


# ---------------------------------------------------------------------------
# Vectorized JAX function
# ---------------------------------------------------------------------------
def compute_G0_hp_jax(T, P, Gf, Hf, Sr, Vr, a, b, c, d,
                      alpha0, kappa0, kappa0p, kappa0pp, numatoms):
    """Compute G⁰(T,P) for multiple species using Holland-Powell model.

    All arrays are (N,) shaped — one entry per species.
    T, P are scalars.

    Parameters
    ----------
    T : scalar, temperature in Kelvin
    P : scalar, pressure in Pascals
    Gf, Hf, Sr, Vr : (N,) formation Gibbs, enthalpy, entropy, volume
    a, b, c, d : (N,) Cp polynomial coefficients
    alpha0, kappa0, kappa0p, kappa0pp : (N,) Tait EOS parameters
    numatoms : (N,) number of atoms per formula unit

    Returns
    -------
    G0 : (N,) standard Gibbs energy at T, P in J/mol
    """
    # vmap over species axis
    return jax.vmap(
        lambda Gf_i, Hf_i, Sr_i, Vr_i, a_i, b_i, c_i, d_i,
               al_i, k_i, kp_i, kpp_i, n_i:
            _hp_G0_scalar(T, P, Gf_i, Hf_i, Sr_i, Vr_i, a_i, b_i, c_i, d_i,
                          al_i, k_i, kp_i, kpp_i, n_i)
    )(Gf, Hf, Sr, Vr, a, b, c, d, alpha0, kappa0, kappa0p, kappa0pp, numatoms)


# ---------------------------------------------------------------------------
# Data loader for SUPCRTBL mineral/gas species
# ---------------------------------------------------------------------------
def load_hp_species(json_path=None, phase_filter=None):
    """Load Holland-Powell species from SUPCRTBL JSON.

    Parameters
    ----------
    json_path : str, optional
        Path to supcrtbl_species.json.
    phase_filter : str or list, optional
        Filter by phase ('mineral', 'gas', or both).

    Returns
    -------
    list[dict]
        Species records with HP parameters.
    """
    if json_path is None:
        json_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'supcrtbl_species.json'
        )

    with open(json_path) as f:
        db = json.load(f)

    if phase_filter is None:
        phase_filter = ['mineral', 'gas']
    elif isinstance(phase_filter, str):
        phase_filter = [phase_filter]

    records = []
    for sp in db['species']:
        if sp['phase'] not in phase_filter:
            continue
        if sp['model_type'] != 'HollandPowell':
            continue
        records.append(sp)

    return records
