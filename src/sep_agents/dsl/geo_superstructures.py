"""
Geological Process Superstructures
====================================

Superstructures for geological hydrogen production and CO₂ sequestration
using reactive industrial wastes (steel slag, mine tailings).

Each function returns a :class:`Superstructure` ready for
:func:`~sep_agents.opt.gdp_solver.optimize_superstructure`.
"""
from __future__ import annotations

from ..dsl.schemas import (
    DisjunctionDef,
    Flowsheet,
    Stream,
    Superstructure,
    UnitOp,
)


# ═══════════════════════════════════════════════════════════════════════
# Steel Slag → H₂ + CaCO₃ Superstructure
# ═══════════════════════════════════════════════════════════════════════

def steel_slag_h2_co2_superstructure() -> Superstructure:
    """Process superstructure for H₂ production and CO₂ sequestration
    from steel slag.

    Chemistry
    ---------
    **Serpentinization (H₂)**:
        Fe₂SiO₄ (fayalite) + 3H₂O → 2Fe(OH)₂ + SiO₂ + H₂
        Conditions: 200–300°C, 50–300 bar

    **Mineral carbonation (CO₂)**:
        CaO + CO₂ + H₂O → CaCO₃
        MgO + CO₂ + H₂O → MgCO₃
        Conditions: 100–200°C, 10–100 bar

    Superstructure Choices
    ----------------------
    - **Reactor configuration**: Series (serpentinization → carbonation)
      XOR Parallel (independent reactor trains)
    - **Heat recovery**: Full recovery (product heats feed via HX)
      XOR Partial recovery (supplemental external heating)
    - **Intermediate scrubbing**: Optional scrubber between reactors
      (series config only)

    Topology (Series Configuration)
    --------------------------------
    ::

        Slag+H₂O → P-101 → E-101(cold) → R-101(serp,250°C)
                                               ↓
                              E-101(hot) ← product
                                  ↓
                              S-101 → H₂(g) product
                                ↓ (liquid)
                  CO₂ → P-102 → E-102(cold) → R-201(carb,150°C)
                                                    ↓
                                   E-102(hot) ← product
                                       ↓
                                   S-201 → CaCO₃(s) product
                                     ↓ (liquid)
                                    Waste / Recycle

    Topology (Parallel Configuration)
    ----------------------------------
    ::

        Slag+H₂O → P-101 → E-101(cold) → R-101(serp,250°C)
                                               ↓
                              E-101(hot) ← product → S-101 → H₂(g)

        Slag+CaO+CO₂ → P-102 → E-102(cold) → R-201(carb,150°C)
                                                    ↓
                                   E-102(hot) ← product → S-201 → CaCO₃(s)
    """

    units = [
        # ══════════════════════════════════════════════════════════════
        # Feed preparation & pressurization
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="pump_slag",
            type="pump",
            params={
                "head_m": 1000.0,       # ~100 bar for water
                "efficiency": 0.75,
            },
            inputs=["slag_water_feed"],
            outputs=["pressurized_slag"],
        ),
        UnitOp(
            id="pump_co2",
            type="pump",
            params={
                "head_m": 500.0,        # ~50 bar for CO₂
                "efficiency": 0.80,
            },
            inputs=["co2_feed"],
            outputs=["pressurized_co2"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Heat exchangers — feed preheating via hot product streams
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="hx_serp_preheat",
            type="heat_exchanger",
            params={
                "U_Wm2K": 500.0,        # Shell-and-tube, slurry service
                "area_m2": 50.0,
                "dT_approach_K": 10.0,
                "type": "counter",
            },
            inputs=["pressurized_slag"],
            outputs=["heated_slag_feed"],
        ),
        UnitOp(
            id="hx_carb_preheat",
            type="heat_exchanger",
            params={
                "U_Wm2K": 800.0,        # Gas-liquid, cleaner service
                "area_m2": 30.0,
                "dT_approach_K": 10.0,
                "type": "counter",
            },
            inputs=["hot_co2_mix"],
            outputs=["heated_co2_feed"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Reactors — Series configuration (both active)
        # ══════════════════════════════════════════════════════════════

        # Serpentinization reactor: 3 Fe₂SiO₄ + 2 H₂O → 2 Fe₃O₄ + 3 SiO₂ + 2 H₂
        UnitOp(
            id="reactor_serpentinization",
            type="stoichiometric_reactor",
            params={
                "residence_time_s": 14400.0,    # 4 hours
                "T_C": 250.0,                   # 250°C
                "p_bar": 100.0,                 # 100 bar
                "tank_volume_m3": 10.0,
                "reactions": {
                    "serpentinization": {
                        "stoichiometry": {
                            "Fe2SiO4": -3, "H2O": -2,
                            "Fe3O4": 2, "SiO2": 3, "H2": 2,
                        },
                        "conversion_spec": {
                            "species": "Fe2SiO4",
                            "conversion": 0.70,
                        },
                    },
                },
            },
            inputs=["heated_slag_feed"],
            outputs=["serp_product_hot"],
        ),

        # Carbonation reactor: CaO + CO₂ → CaCO₃, MgO + CO₂ → MgCO₃
        UnitOp(
            id="reactor_carbonation",
            type="stoichiometric_reactor",
            params={
                "residence_time_s": 7200.0,     # 2 hours
                "T_C": 150.0,                   # 150°C
                "p_bar": 50.0,                  # 50 bar
                "tank_volume_m3": 8.0,
                "reactions": {
                    "carbonation_CaO": {
                        "stoichiometry": {
                            "CaO": -1, "CO2": -1, "CaCO3": 1,
                        },
                        "conversion_spec": {
                            "species": "CaO",
                            "conversion": 0.85,
                        },
                    },
                    "carbonation_MgO": {
                        "stoichiometry": {
                            "MgO": -1, "CO2": -1, "MgCO3": 1,
                        },
                        "conversion_spec": {
                            "species": "MgO",
                            "conversion": 0.85,
                        },
                    },
                },
            },
            inputs=["heated_co2_feed"],
            outputs=["carb_product_hot"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Heat recovery — hot products preheat cold feeds
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="hx_serp_recovery",
            type="heat_exchanger",
            params={
                "U_Wm2K": 500.0,
                "area_m2": 50.0,
                "dT_approach_K": 15.0,
                "type": "counter",
            },
            inputs=["serp_product_hot"],
            outputs=["serp_product_cooled"],
        ),
        UnitOp(
            id="hx_carb_recovery",
            type="heat_exchanger",
            params={
                "U_Wm2K": 800.0,
                "area_m2": 30.0,
                "dT_approach_K": 15.0,
                "type": "counter",
            },
            inputs=["carb_product_hot"],
            outputs=["carb_product_cooled"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Optional supplemental heater (if heat recovery insufficient)
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="aux_heater_serp",
            type="heat_exchanger",
            params={
                "U_Wm2K": 1000.0,
                "area_m2": 20.0,
                "dT_approach_K": 5.0,
                "type": "counter",
            },
            inputs=["heated_slag_feed"],
            outputs=["boosted_slag_feed"],
            optional=True,
            alternatives=["heat_strategy"],
        ),
        UnitOp(
            id="no_aux_heater",
            type="mixer",
            params={},
            inputs=["heated_slag_feed"],
            outputs=["boosted_slag_feed"],
            optional=True,
            alternatives=["heat_strategy"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Product separation
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="separator_h2",
            type="separator",
            params={
                "recovery": 0.95,
                "split_fraction": 0.99,    # H₂ gas recovery
            },
            inputs=["serp_product_cooled"],
            outputs=["h2_gas_product", "serp_liquid_residue"],
        ),
        UnitOp(
            id="separator_carbonate",
            type="separator",
            params={
                "recovery": 0.90,
                "split_fraction": 0.95,    # CaCO₃ solid recovery
            },
            inputs=["carb_product_cooled"],
            outputs=["carbonate_solid_product", "carb_liquid_residue"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Optional scrubber (series config: between reactor trains)
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="scrubber",
            type="mixer",
            params={},
            inputs=["serp_liquid_residue"],
            outputs=["serp_to_carb_liquid"],
            optional=True,
        ),

        # ══════════════════════════════════════════════════════════════
        # Mixer: combine CO₂ stream with liquid from serpentinization
        # (series config) or with fresh slag (parallel config)
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="mixer_co2",
            type="mixer",
            params={},
            inputs=["pressurized_co2", "serp_to_carb_liquid"],
            outputs=["hot_co2_mix"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Final liquid waste / recycle mixer
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="mixer_waste",
            type="mixer",
            params={},
            inputs=["carb_liquid_residue"],
            outputs=["liquid_waste"],
        ),
    ]

    # ══════════════════════════════════════════════════════════════════
    # Stream definitions
    # ══════════════════════════════════════════════════════════════════
    streams = [
        # ── Feed streams ────────────────────────────────────────────
        Stream(
            name="slag_water_feed",
            phase="liquid",
            temperature_K=298.15,
            pressure_Pa=101325.0,
            composition_wt={
                "H2O": 800.0,           # kg water per batch
                "Fe2SiO4": 100.0,       # Fayalite from steel slag
                "CaO": 50.0,           # Lime from steel slag
                "MgO": 30.0,           # Magnesia from steel slag
                "SiO2": 20.0,          # Silica gangue
            },
        ),
        Stream(
            name="co2_feed",
            phase="gas",
            temperature_K=298.15,
            pressure_Pa=101325.0,
            composition_wt={
                "CO2": 44.0,            # kg CO₂ (1 kmol)
            },
        ),

        # ── Pressurized feeds ───────────────────────────────────────
        Stream(name="pressurized_slag", phase="liquid",
               temperature_K=298.15, pressure_Pa=1e7),
        Stream(name="pressurized_co2", phase="gas",
               temperature_K=298.15, pressure_Pa=5e6),

        # ── Heat exchanger intermediates ────────────────────────────
        Stream(name="heated_slag_feed", phase="liquid",
               temperature_K=473.15, pressure_Pa=1e7),   # ~200°C
        Stream(name="boosted_slag_feed", phase="liquid",
               temperature_K=523.15, pressure_Pa=1e7),   # ~250°C
        Stream(name="hot_co2_mix", phase="liquid",
               temperature_K=298.15, pressure_Pa=5e6),
        Stream(name="heated_co2_feed", phase="liquid",
               temperature_K=423.15, pressure_Pa=5e6),   # ~150°C

        # ── Reactor products ────────────────────────────────────────
        Stream(name="serp_product_hot", phase="liquid",
               temperature_K=523.15, pressure_Pa=1e7),   # 250°C
        Stream(name="carb_product_hot", phase="liquid",
               temperature_K=423.15, pressure_Pa=5e6),   # 150°C

        # ── Cooled products (after heat recovery) ───────────────────
        Stream(name="serp_product_cooled", phase="liquid",
               temperature_K=343.15, pressure_Pa=1e7),   # ~70°C
        Stream(name="carb_product_cooled", phase="liquid",
               temperature_K=323.15, pressure_Pa=5e6),   # ~50°C

        # ── Separated products ──────────────────────────────────────
        Stream(name="h2_gas_product", phase="gas",
               temperature_K=343.15, pressure_Pa=1e7),   # H₂ product
        Stream(name="serp_liquid_residue", phase="liquid",
               temperature_K=343.15, pressure_Pa=1e7),
        Stream(name="carbonate_solid_product", phase="solid",
               temperature_K=323.15, pressure_Pa=101325.0),  # CaCO₃
        Stream(name="carb_liquid_residue", phase="liquid",
               temperature_K=323.15, pressure_Pa=5e6),

        # ── Inter-reactor streams ───────────────────────────────────
        Stream(name="serp_to_carb_liquid", phase="liquid",
               temperature_K=343.15, pressure_Pa=5e6),

        # ── Waste ───────────────────────────────────────────────────
        Stream(name="liquid_waste", phase="liquid",
               temperature_K=323.15, pressure_Pa=101325.0),
    ]

    return Superstructure(
        name="steel_slag_h2_co2",
        base_flowsheet=Flowsheet(
            name="steel_slag_h2_co2_base",
            units=units,
            streams=streams,
        ),
        disjunctions=[
            DisjunctionDef(
                name="heat_strategy",
                unit_ids=["aux_heater_serp", "no_aux_heater"],
                description="Supplemental heating: auxiliary heater vs "
                            "full heat recovery only",
            ),
        ],
        fixed_units=[
            "pump_slag", "pump_co2",
            "hx_serp_preheat", "hx_carb_preheat",
            "reactor_serpentinization", "reactor_carbonation",
            "hx_serp_recovery", "hx_carb_recovery",
            "separator_h2", "separator_carbonate",
            "mixer_co2", "mixer_waste",
        ],
        objective="minimize_opex",
        continuous_bounds={
            # Reactor conditions
            "reactor_serpentinization.T_C": (200.0, 300.0),
            "reactor_serpentinization.p_bar": (50.0, 300.0),
            "reactor_serpentinization.residence_time_s": (3600.0, 86400.0),
            "reactor_carbonation.T_C": (100.0, 200.0),
            "reactor_carbonation.p_bar": (10.0, 100.0),
            "reactor_carbonation.residence_time_s": (1800.0, 28800.0),
            # Heat exchanger sizing
            "hx_serp_preheat.area_m2": (10.0, 100.0),
            "hx_carb_preheat.area_m2": (5.0, 60.0),
            "hx_serp_recovery.area_m2": (10.0, 100.0),
            "hx_carb_recovery.area_m2": (5.0, 60.0),
            # Pump sizing
            "pump_slag.head_m": (500.0, 3000.0),
            "pump_co2.head_m": (100.0, 1000.0),
        },
        description=(
            "Steel slag valorization for H₂ production (serpentinization of "
            "fayalite) and CO₂ sequestration (mineral carbonation of CaO/MgO). "
            "Includes heat recovery network (counter-current HX), pressurization "
            "pumps, and product separators. GDP choices: supplemental heating "
            "vs full heat recovery; optional inter-reactor scrubbing."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

GEO_SUPERSTRUCTURE_REGISTRY = {
    "steel_slag_h2_co2": steel_slag_h2_co2_superstructure,
}


def list_geo_superstructures() -> list[dict]:
    """List available geological process superstructures."""
    result = []
    for name, factory in GEO_SUPERSTRUCTURE_REGISTRY.items():
        ss = factory()
        n_units = len(ss.base_flowsheet.units)
        n_disj = len(ss.disjunctions)
        n_opt = sum(1 for u in ss.base_flowsheet.units if u.optional)
        result.append({
            "name": name,
            "description": ss.description,
            "num_units": n_units,
            "num_disjunctions": n_disj,
            "num_optional_units": n_opt,
            "fixed_units": ss.fixed_units,
            "objective": ss.objective,
        })
    return result
