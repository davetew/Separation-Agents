"""
Peng-Robinson Equation of State — JAX implementation for gas fugacity.

Computes fugacity coefficients φ(T,P) for geological gas species (CO₂, H₂, 
CH₄, H₂O, etc.) using the Peng-Robinson cubic EOS.  Fugacity replaces the
ideal-gas activity coefficient: a_gas = φ·P/Pref instead of P/Pref.

All functions are JAX-traceable for autodiff.

References
----------
- Peng & Robinson (1976), Ind. Eng. Chem. Fund. 15, 59-64.
- Robinson, Peng & Chung (1985), for acentric factor correlations.
"""
from __future__ import annotations

import jax.numpy as jnp

R_GAS = 8.314462618  # J/(mol·K)

# ── Critical properties of common geological gases ──────────────────────
# Format: (Tc [K], Pc [Pa], acentric factor ω)
# Source: Perry's Chemical Engineers' Handbook & NIST
CRITICAL_PROPS = {
    'CO2(g)':     (304.13, 7.377e6, 0.2236),
    'H2(g)':      (33.15, 1.296e6, -0.219),
    'H2O(g)':     (647.10, 22.064e6, 0.3443),
    'CH4(g)':     (190.56, 4.599e6, 0.0115),
    'O2(g)':      (154.58, 5.043e6, 0.0222),
    'N2(g)':      (126.19, 3.396e6, 0.0372),
    'CO(g)':      (132.86, 3.499e6, 0.0497),
    'H2S(g)':     (373.53, 8.963e6, 0.0942),
    'SO2(g)':     (430.64, 7.884e6, 0.2451),
    'NH3(g)':     (405.40, 11.333e6, 0.2526),
    'Ar(g)':      (150.69, 4.863e6, 0.0),
    'He(g)':      (5.19, 0.227e6, -0.390),
}


def _pr_ab(T_K, Tc, Pc, omega):
    """Peng-Robinson a(T) and b parameters.

    a(T) = 0.45724 · R²·Tc² / Pc · α(T)
    b    = 0.07780 · R·Tc / Pc
    α(T) = [1 + κ·(1 - √(T/Tc))]²
    κ    = 0.37464 + 1.54226·ω - 0.26992·ω²
    """
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    Tr = T_K / Tc
    alpha = (1.0 + kappa * (1.0 - jnp.sqrt(Tr)))**2

    a = 0.45724 * R_GAS**2 * Tc**2 / Pc * alpha
    b = 0.07780 * R_GAS * Tc / Pc

    return a, b


def _pr_Z(A, B):
    """Solve the PR cubic for compressibility factor Z.

    Z³ - (1-B)·Z² + (A-3B²-2B)·Z - (AB-B²-B³) = 0

    Returns the largest real root (vapor phase Z).
    We use the analytical solution for the cubic.
    """
    # Coefficients: Z³ + p·Z² + q·Z + r = 0
    p = -(1.0 - B)
    q = A - 3.0 * B**2 - 2.0 * B
    r = -(A * B - B**2 - B**3)

    # Depressed cubic: t³ + pt² + qt + r = 0
    # Shift: Z = t - p/3
    Q = (3.0 * q - p**2) / 9.0
    R_val = (9.0 * p * q - 27.0 * r - 2.0 * p**3) / 54.0

    D = Q**3 + R_val**2  # Discriminant

    # For gas phase, take the largest real root
    # When D > 0, one real root (use Cardano's formula)
    # When D ≤ 0, three real roots (use trigonometric solution)

    # Cardano's formula for D > 0
    sqrt_D = jnp.sqrt(jnp.maximum(D, 0.0))
    S = jnp.sign(R_val + sqrt_D) * jnp.abs(R_val + sqrt_D)**(1.0/3.0)
    T_val = jnp.sign(R_val - sqrt_D) * jnp.abs(R_val - sqrt_D)**(1.0/3.0)
    Z_cardano = S + T_val - p / 3.0

    # Trigonometric solution for D ≤ 0 (three real roots)
    theta = jnp.arccos(jnp.clip(R_val / jnp.sqrt(jnp.maximum(-Q**3, 1e-30)), -1, 1))
    sqrt_neg_Q = jnp.sqrt(jnp.maximum(-Q, 0.0))
    Z1 = 2.0 * sqrt_neg_Q * jnp.cos(theta / 3.0) - p / 3.0
    Z2 = 2.0 * sqrt_neg_Q * jnp.cos((theta + 2.0 * jnp.pi) / 3.0) - p / 3.0
    Z3 = 2.0 * sqrt_neg_Q * jnp.cos((theta + 4.0 * jnp.pi) / 3.0) - p / 3.0
    Z_trig = jnp.maximum(Z1, jnp.maximum(Z2, Z3))  # Largest root = vapor

    Z = jnp.where(D > 0, Z_cardano, Z_trig)

    # Z must be physical: Z > B
    return jnp.maximum(Z, B + 0.001)


def fugacity_coefficient(T_K, P_Pa, Tc, Pc, omega):
    """Peng-Robinson fugacity coefficient φ for a pure gas.

    ln(φ) = Z - 1 - ln(Z - B) - A/(2√2·B)·ln((Z+(1+√2)B)/(Z+(1-√2)B))

    Parameters
    ----------
    T_K : temperature (K)
    P_Pa : pressure (Pa)
    Tc : critical temperature (K)
    Pc : critical pressure (Pa)
    omega : acentric factor

    Returns
    -------
    phi : fugacity coefficient (dimensionless), φ ∈ (0, ~1)
    """
    a, b = _pr_ab(T_K, Tc, Pc, omega)

    A = a * P_Pa / (R_GAS * T_K)**2
    B = b * P_Pa / (R_GAS * T_K)

    Z = _pr_Z(A, B)

    sqrt2 = jnp.sqrt(2.0)
    ln_phi = (Z - 1.0
              - jnp.log(jnp.maximum(Z - B, 1e-10))
              - A / (2.0 * sqrt2 * B) *
              jnp.log(jnp.maximum((Z + (1.0 + sqrt2) * B) /
                                  (Z + (1.0 - sqrt2) * B), 1e-10)))

    return jnp.exp(ln_phi)


def compute_fugacity_coefficients(T_K, P_Pa, species_names):
    """Compute fugacity coefficients for a list of gas species.

    For species not in the critical property database, returns φ = 1.0
    (ideal gas approximation).

    Parameters
    ----------
    T_K : temperature (K)
    P_Pa : pressure (Pa)
    species_names : list of species names (e.g., ['CO2(g)', 'H2(g)'])

    Returns
    -------
    phi : dict of {species_name: fugacity_coefficient}
    """
    result = {}
    for name in species_names:
        if name in CRITICAL_PROPS:
            Tc, Pc, omega = CRITICAL_PROPS[name]
            phi = float(fugacity_coefficient(
                jnp.array(T_K), jnp.array(P_Pa),
                jnp.array(Tc), jnp.array(Pc), jnp.array(omega)))
            result[name] = phi
        else:
            result[name] = 1.0  # Ideal gas
    return result
