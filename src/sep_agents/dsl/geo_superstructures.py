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
# Natural Olivine → H₂ + MgCO₃ Superstructure
# ═══════════════════════════════════════════════════════════════════════

def olivine_carbonation_h2_superstructure() -> Superstructure:
    """Process superstructure for H₂ production and CO₂ sequestration
    from natural olivine (forsterite + fayalite).

    Chemistry
    ---------
    **Serpentinization (H₂)** — fayalite fraction:
        3 Fe₂SiO₄ + 2 H₂O → 2 Fe₃O₄ + 3 SiO₂ + 2 H₂
        Conditions: 200–300 °C, 50–300 bar

    **Direct carbonation (CO₂)** — forsterite fraction:
        Mg₂SiO₄ + 2 CO₂ → 2 MgCO₃ + SiO₂
        Conditions: 150–200 °C, 50–200 bar

    Value Streams
    -------------
    - CO₂ credits from MgCO₃ sequestration (primary)
    - Green H₂ from serpentinization
    - Fe₃O₄ magnetite concentrate (optional magnetic separation)
    - SiO₂ aggregate (residual)

    Superstructure Choices
    ----------------------
    - **Heat strategy**: auxiliary heater vs full heat recovery only
    - **Magnetic separation**: optional LIMS for Fe₃O₄ recovery
    """

    units = [
        # ══════════════════════════════════════════════════════════════
        # Feed preparation & pressurization
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="pump_olivine",
            type="pump",
            params={"head_m": 1000.0, "efficiency": 0.75},
            inputs=["olivine_water_feed"],
            outputs=["pressurized_olivine"],
        ),
        UnitOp(
            id="pump_co2",
            type="pump",
            params={"head_m": 1500.0, "efficiency": 0.80},
            inputs=["co2_feed"],
            outputs=["pressurized_co2"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Preheat olivine slurry
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="hx_serp_preheat",
            type="heat_exchanger",
            params={"U_Wm2K": 500.0, "area_m2": 40.0, "dT_approach_K": 15.0},
            inputs=["pressurized_olivine"],
            outputs=["preheated_olivine"],
        ),

        # Auxiliary heater (XOR with no_aux_heater)
        UnitOp(
            id="aux_heater",
            type="heat_exchanger",
            params={"U_Wm2K": 1000.0, "area_m2": 5.0, "dT_approach_K": 5.0},
            inputs=["preheated_olivine"],
            outputs=["heated_olivine"],
            alternatives=["heat_strategy"],
        ),
        UnitOp(
            id="no_aux_heater",
            type="separator",
            params={"recovery": 1.0},
            inputs=["preheated_olivine"],
            outputs=["heated_olivine"],
            alternatives=["heat_strategy"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Reactor 1: Serpentinization (Gibbs equilibrium)
        # Fayalite + H₂O ⇌ Magnetite + SiO₂ + H₂  (+ other equilibria)
        # Conversion computed from thermodynamics at T/P conditions.
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="reactor_serpentinization",
            type="equilibrium_reactor",
            params={
                "T_C": 250.0,
                "p_bar": 100.0,
                "tank_volume_m3": 10.0,
                "equilibrium_phases": [
                    "Forsterite", "Fayalite", "Magnetite",
                    "Quartz", "Hematite", "Brucite",
                ],
                "gas_phases": ["H2(g)", "CO2(g)", "H2O(g)", "O2(g)"],
                "aqueous_elements": "H O C Si Mg Fe",
                "database": "supcrtbl",
            },
            inputs=["heated_olivine"],
            outputs=["serp_product_hot"],
        ),

        # Cool serpentinization product
        UnitOp(
            id="hx_serp_recovery",
            type="heat_exchanger",
            params={"U_Wm2K": 500.0, "area_m2": 50.0, "dT_approach_K": 15.0},
            inputs=["serp_product_hot"],
            outputs=["serp_product_cooled"],
        ),

        # Separate H₂ gas from slurry (Mg₂SiO₄ + Fe₃O₄ + SiO₂)
        UnitOp(
            id="separator_h2",
            type="separator",
            params={"recovery": 0.95},
            inputs=["serp_product_cooled"],
            outputs=["h2_gas_product", "serp_slurry"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Mixer: combine serpentinization residue (Mg₂SiO₄) with CO₂
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="mixer_co2",
            type="mixer",
            params={},
            inputs=["serp_slurry", "pressurized_co2"],
            outputs=["carb_feed_mix"],
        ),

        # Preheat the mixed feed for carbonation
        UnitOp(
            id="hx_carb_preheat",
            type="heat_exchanger",
            params={"U_Wm2K": 800.0, "area_m2": 30.0, "dT_approach_K": 10.0},
            inputs=["carb_feed_mix"],
            outputs=["heated_carb_feed"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Reactor 2: Direct carbonation of forsterite (Gibbs equilibrium)
        # Forsterite + CO₂ ⇌ Magnesite + SiO₂  (+ other equilibria)
        # Conversion computed from thermodynamics at T/P conditions.
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="reactor_carbonation",
            type="equilibrium_reactor",
            params={
                "T_C": 185.0,
                "p_bar": 150.0,
                "tank_volume_m3": 15.0,
                "equilibrium_phases": [
                    "Forsterite", "Fayalite", "Magnetite",
                    "Magnesite", "Quartz", "Hematite", "Brucite",
                ],
                "gas_phases": ["H2(g)", "CO2(g)", "H2O(g)", "O2(g)"],
                "aqueous_elements": "H O C Si Mg Fe",
                "database": "supcrtbl",
            },
            inputs=["heated_carb_feed"],
            outputs=["carb_product_hot"],
        ),

        # Cool carbonation product
        UnitOp(
            id="hx_carb_recovery",
            type="heat_exchanger",
            params={"U_Wm2K": 600.0, "area_m2": 60.0, "dT_approach_K": 10.0},
            inputs=["carb_product_hot"],
            outputs=["carb_product_cooled"],
        ),

        # Separate MgCO₃ solids
        UnitOp(
            id="separator_carbonate",
            type="separator",
            params={"recovery": 0.90},
            inputs=["carb_product_cooled"],
            outputs=["mgco3_solid_product", "carb_liquid_residue"],
        ),

        # ══════════════════════════════════════════════════════════════
        # Optional: Magnetic separation for Fe₃O₄
        # ══════════════════════════════════════════════════════════════
        UnitOp(
            id="lims_fe3o4",
            type="lims",
            params={"magnetic_recovery": 0.85},
            inputs=["carb_liquid_residue"],
            outputs=["fe3o4_concentrate", "silica_residue"],
            alternatives=["mag_separation"],
        ),
        UnitOp(
            id="no_mag_sep",
            type="separator",
            params={"recovery": 1.0},
            inputs=["carb_liquid_residue"],
            outputs=["silica_residue"],
            alternatives=["mag_separation"],
        ),
    ]

    streams = [
        # ── Feeds ───────────────────────────────────────────────────
        Stream(
            name="olivine_water_feed", phase="liquid",
            temperature_K=298.15, pressure_Pa=101325.0,
            composition_wt={
                "H2O": 800.0,
                "Mg2SiO4": 575.0,
                "Fe2SiO4": 94.0,
            },
        ),
        Stream(
            name="co2_feed", phase="gas",
            temperature_K=298.15, pressure_Pa=101325.0,
            composition_wt={"CO2": 1200.0},
        ),

        # ── Pressurized ─────────────────────────────────────────────
        Stream(name="pressurized_olivine", phase="liquid",
               temperature_K=298.15, pressure_Pa=1e7),
        Stream(name="pressurized_co2", phase="gas",
               temperature_K=298.15, pressure_Pa=1.5e7),

        # ── Preheated ───────────────────────────────────────────────
        Stream(name="preheated_olivine", phase="liquid",
               temperature_K=473.15, pressure_Pa=1e7),
        Stream(name="heated_olivine", phase="liquid",
               temperature_K=523.15, pressure_Pa=1e7),

        # ── Reactor 1 (serpentinization) ────────────────────────────
        Stream(name="serp_product_hot", phase="liquid",
               temperature_K=523.15, pressure_Pa=1e7),
        Stream(name="serp_product_cooled", phase="liquid",
               temperature_K=343.15, pressure_Pa=1e7),

        # ── H₂ separation ──────────────────────────────────────────
        Stream(name="h2_gas_product", phase="gas",
               temperature_K=343.15, pressure_Pa=1e7),
        Stream(name="serp_slurry", phase="liquid",
               temperature_K=343.15, pressure_Pa=1e7),

        # ── CO₂ mixing + carbonation preheat ───────────────────────
        Stream(name="carb_feed_mix", phase="liquid",
               temperature_K=335.15, pressure_Pa=1e7),
        Stream(name="heated_carb_feed", phase="liquid",
               temperature_K=458.15, pressure_Pa=1e7),

        # ── Reactor 2 (carbonation) ────────────────────────────────
        Stream(name="carb_product_hot", phase="liquid",
               temperature_K=458.15, pressure_Pa=1e7),
        Stream(name="carb_product_cooled", phase="liquid",
               temperature_K=323.15, pressure_Pa=1e7),

        # ── Product separation ──────────────────────────────────────
        Stream(name="mgco3_solid_product", phase="solid",
               temperature_K=323.15, pressure_Pa=101325.0),
        Stream(name="carb_liquid_residue", phase="liquid",
               temperature_K=323.15, pressure_Pa=1e7),
        Stream(name="fe3o4_concentrate", phase="solid",
               temperature_K=323.15, pressure_Pa=101325.0),
        Stream(name="silica_residue", phase="solid",
               temperature_K=323.15, pressure_Pa=101325.0),
    ]

    return Superstructure(
        name="olivine_carbonation_h2",
        base_flowsheet=Flowsheet(
            name="olivine_carbonation_h2_base",
            units=units,
            streams=streams,
        ),
        disjunctions=[
            DisjunctionDef(
                name="heat_strategy",
                unit_ids=["aux_heater", "no_aux_heater"],
                description="Supplemental heating vs full heat recovery only",
            ),
            DisjunctionDef(
                name="mag_separation",
                unit_ids=["lims_fe3o4", "no_mag_sep"],
                description="Magnetic separation of Fe₃O₄ vs no separation",
            ),
        ],
        fixed_units=[
            "pump_olivine", "pump_co2",
            "hx_serp_preheat", "hx_carb_preheat",
            "reactor_serpentinization", "reactor_carbonation",
            "hx_serp_recovery", "hx_carb_recovery",
            "separator_h2", "separator_carbonate",
            "mixer_co2",
        ],
        objective="minimize_opex",
        continuous_bounds={
            "reactor_serpentinization.T_C": (200.0, 300.0),
            "reactor_serpentinization.p_bar": (50.0, 300.0),
            "reactor_carbonation.T_C": (150.0, 200.0),
            "reactor_carbonation.p_bar": (50.0, 200.0),
            "hx_serp_preheat.area_m2": (10.0, 100.0),
            "hx_carb_preheat.area_m2": (5.0, 60.0),
            "hx_serp_recovery.area_m2": (10.0, 100.0),
            "hx_carb_recovery.area_m2": (10.0, 120.0),
            "pump_olivine.head_m": (500.0, 3000.0),
            "pump_co2.head_m": (500.0, 2000.0),
        },
        description=(
            "Olivine valorization for CO₂ sequestration (direct carbonation "
            "of forsterite → MgCO₃) and H₂ production (serpentinization of "
            "fayalite → Fe₃O₄ + H₂). Includes heat recovery, pressurization, "
            "and product separation. GDP choices: auxiliary heating strategy + "
            "optional magnetic separation for Fe₃O₄ recovery."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# EAF Steel Slag → Fe₃O₄ + H₂ + CaCO₃ Superstructure
# ═══════════════════════════════════════════════════════════════════════

def eaf_steel_slag_superstructure() -> Superstructure:
    """EAF steel slag valorization for iron recovery, H₂ production,
    and CO₂ mineralization.

    Composition Basis (per 1000 kg EAF slag)
    -----------------------------------------
    CaO 30%, FeO 25%, Fe₂O₃ 8%, SiO₂ 17%, MgO 8%,
    Al₂O₃ 6%, MnO 4%, Cr₂O₃ 2%

    Value Streams
    -------------
    1. **Iron concentrate** (Fe₃O₄ + Cr₂O₃ + MnO) via LIMS magnetic sep
    2. **H₂** via serpentinization of fayalite (Fe₂SiO₄)
    3. **CO₂ credits** via mineral carbonation of CaO + MgO
    4. **Construction aggregate** from residual silicates

    GDP Choices
    -----------
    - Magnetic separation: before or after serpentinization
    - Heat recovery: full vs supplemental heater
    - Carbonation pressure: low (10–50 bar) vs high (50–150 bar)
    """

    units = [
        # ══════════ Feed Preparation ══════════
        UnitOp(
            id="pump_slag",
            type="pump",
            params={"head_m": 1500.0, "efficiency": 0.75},
            inputs=["slag_water_feed"],
            outputs=["pressurized_slag"],
        ),
        UnitOp(
            id="pump_co2",
            type="pump",
            params={"head_m": 800.0, "efficiency": 0.80},
            inputs=["co2_feed"],
            outputs=["pressurized_co2"],
        ),

        # ══════════ LIMS Magnetic Separation (before serp) ══════════
        # Recovers Fe₃O₄, Cr₂O₃, MnO from raw slag
        UnitOp(
            id="lims_pre_serp",
            type="lims",
            params={
                "magnetic_recovery": 0.85,
                "field_T": 0.3,
            },
            inputs=["pressurized_slag"],
            outputs=["fe_concentrate_pre", "non_magnetic_slag"],
            optional=True,
            alternatives=["lims_position"],
        ),
        # Bypass: no pre-serp mag sep (send all to serpentinization)
        UnitOp(
            id="no_lims_pre",
            type="mixer",
            params={},
            inputs=["pressurized_slag"],
            outputs=["non_magnetic_slag"],
            optional=True,
            alternatives=["lims_position"],
        ),

        # ══════════ Heat Recovery Network ══════════
        UnitOp(
            id="hx_serp_preheat",
            type="heat_exchanger",
            params={
                "U_Wm2K": 500.0,
                "area_m2": 50.0,
                "dT_approach_K": 10.0,
                "type": "counter",
            },
            inputs=["non_magnetic_slag"],
            outputs=["heated_slag_feed"],
        ),

        # Optional aux heater
        UnitOp(
            id="aux_heater",
            type="heat_exchanger",
            params={
                "U_Wm2K": 1000.0,
                "area_m2": 20.0,
                "dT_approach_K": 5.0,
                "type": "counter",
            },
            inputs=["heated_slag_feed"],
            outputs=["reactor_feed"],
            optional=True,
            alternatives=["heat_strategy"],
        ),
        UnitOp(
            id="no_aux_heater",
            type="mixer",
            params={},
            inputs=["heated_slag_feed"],
            outputs=["reactor_feed"],
            optional=True,
            alternatives=["heat_strategy"],
        ),

        # ══════════ Serpentinization Reactor (Equilibrium) ══════════
        UnitOp(
            id="reactor_serpentinization",
            type="equilibrium_reactor",
            params={
                "residence_time_s": 14400.0,
                "T_C": 250.0,
                "p_bar": 100.0,
                "tank_volume_m3": 10.0,
                "equilibrium_phases": [
                    "Forsterite", "Fayalite", "Magnetite",
                    "Quartz", "Hematite", "Brucite",
                ],
                "gas_phases": [],
                "aqueous_elements": [
                    "H", "O", "Fe", "Si", "Mg",
                ],
                "database": "supcrtbl",
            },
            inputs=["reactor_feed"],
            outputs=["serp_product_hot"],
        ),

        # Hot product heat recovery
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

        # ══════════ H₂ Gas Separation ══════════
        UnitOp(
            id="separator_h2",
            type="separator",
            params={"recovery": 0.95, "split_fraction": 0.99},
            inputs=["serp_product_cooled"],
            outputs=["h2_gas_product", "serp_liquid"],
        ),

        # ══════════ CO₂ Mixing + Carbonation ══════════
        UnitOp(
            id="mixer_co2",
            type="mixer",
            params={},
            inputs=["pressurized_co2", "serp_liquid"],
            outputs=["carbonation_feed_raw"],
        ),
        UnitOp(
            id="hx_carb_preheat",
            type="heat_exchanger",
            params={
                "U_Wm2K": 800.0,
                "area_m2": 30.0,
                "dT_approach_K": 10.0,
                "type": "counter",
            },
            inputs=["carbonation_feed_raw"],
            outputs=["carbonation_feed"],
        ),

        # Carbonation reactor (Equilibrium)
        UnitOp(
            id="reactor_carbonation",
            type="equilibrium_reactor",
            params={
                "residence_time_s": 7200.0,
                "T_C": 185.0,
                "p_bar": 50.0,
                "tank_volume_m3": 8.0,
                "equilibrium_phases": [
                    "Forsterite", "Fayalite", "Magnetite",
                    "Magnesite", "Quartz", "Hematite", "Brucite",
                ],
                "gas_phases": ["H2O(g)", "CO2(g)", "H2(g)"],
                "aqueous_elements": [
                    "H", "O", "C", "Fe", "Si", "Mg", "Ca",
                ],
                "database": "supcrtbl",
            },
            inputs=["carbonation_feed"],
            outputs=["carb_product_hot"],
        ),

        # Carbonation product heat recovery
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

        # ══════════ Carbonate Solid Recovery ══════════
        UnitOp(
            id="separator_carbonate",
            type="separator",
            params={"recovery": 0.90, "split_fraction": 0.95},
            inputs=["carb_product_cooled"],
            outputs=["carbonate_solid_product", "carb_liquid_residue"],
        ),

        # ══════════ Waste mixer ══════════
        UnitOp(
            id="mixer_waste",
            type="mixer",
            params={},
            inputs=["carb_liquid_residue"],
            outputs=["liquid_waste"],
        ),
    ]

    streams = [
        # Feed streams
        Stream(
            name="slag_water_feed", phase="liquid",
            temperature_K=298.15, pressure_Pa=101325.0,
            composition_wt={
                "H2O": 800.0,
                "Fe2SiO4": 1.74,   # mol fayalite from FeO+SiO₂
                "CaO": 5.35,       # mol
                "MgO": 1.99,       # mol
                "SiO2": 1.09,      # mol free silica
                "Fe3O4": 0.33,     # mol pre-existing magnetite
            },
        ),
        Stream(
            name="co2_feed", phase="gas",
            temperature_K=298.15, pressure_Pa=101325.0,
            composition_wt={"CO2": 8.07},
        ),

        # Internal streams
        Stream(name="pressurized_slag", phase="liquid",
               temperature_K=298.15, pressure_Pa=1e7),
        Stream(name="pressurized_co2", phase="gas",
               temperature_K=298.15, pressure_Pa=5e6),
        Stream(name="fe_concentrate_pre", phase="solid",
               temperature_K=298.15, pressure_Pa=1e7),
        Stream(name="non_magnetic_slag", phase="liquid",
               temperature_K=298.15, pressure_Pa=1e7),
        Stream(name="heated_slag_feed", phase="liquid",
               temperature_K=473.15, pressure_Pa=1e7),
        Stream(name="reactor_feed", phase="liquid",
               temperature_K=523.15, pressure_Pa=1e7),
        Stream(name="serp_product_hot", phase="liquid",
               temperature_K=523.15, pressure_Pa=1e7),
        Stream(name="serp_product_cooled", phase="liquid",
               temperature_K=343.15, pressure_Pa=1e7),
        Stream(name="h2_gas_product", phase="gas",
               temperature_K=343.15, pressure_Pa=1e7),
        Stream(name="serp_liquid", phase="liquid",
               temperature_K=343.15, pressure_Pa=1e7),
        Stream(name="carbonation_feed_raw", phase="liquid",
               temperature_K=298.15, pressure_Pa=5e6),
        Stream(name="carbonation_feed", phase="liquid",
               temperature_K=423.15, pressure_Pa=5e6),
        Stream(name="carb_product_hot", phase="liquid",
               temperature_K=458.15, pressure_Pa=5e6),
        Stream(name="carb_product_cooled", phase="liquid",
               temperature_K=323.15, pressure_Pa=5e6),
        Stream(name="carbonate_solid_product", phase="solid",
               temperature_K=323.15, pressure_Pa=101325.0),
        Stream(name="carb_liquid_residue", phase="liquid",
               temperature_K=323.15, pressure_Pa=5e6),
        Stream(name="liquid_waste", phase="liquid",
               temperature_K=323.15, pressure_Pa=101325.0),
    ]

    return Superstructure(
        name="eaf_steel_slag",
        base_flowsheet=Flowsheet(
            name="eaf_steel_slag_base",
            units=units,
            streams=streams,
        ),
        disjunctions=[
            DisjunctionDef(
                name="lims_position",
                unit_ids=["lims_pre_serp", "no_lims_pre"],
                description="Magnetic separation before serpentinization "
                            "vs. bypass (Fe₃O₄ formed in-situ by serp)",
            ),
            DisjunctionDef(
                name="heat_strategy",
                unit_ids=["aux_heater", "no_aux_heater"],
                description="Supplemental heater vs full heat recovery only",
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
            "reactor_carbonation.T_C": (100.0, 200.0),
            "reactor_carbonation.p_bar": (10.0, 150.0),
            # HX sizing
            "hx_serp_preheat.area_m2": (10.0, 100.0),
            "hx_carb_preheat.area_m2": (5.0, 60.0),
            "hx_serp_recovery.area_m2": (10.0, 100.0),
            "hx_carb_recovery.area_m2": (5.0, 60.0),
            # Pumps
            "pump_slag.head_m": (500.0, 3000.0),
            "pump_co2.head_m": (100.0, 1500.0),
        },
        description=(
            "EAF steel slag valorization for iron recovery (LIMS, Cr/Mn), "
            "H₂ production (serpentinization of fayalite), and CO₂ "
            "sequestration (mineral carbonation of CaO/MgO). "
            "GDP choices: LIMS position (pre/post-serp), heat strategy, "
            "carbonation pressure. Throughput: 100,000 t/yr."
        ),
    )


# ═══════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════

GEO_SUPERSTRUCTURE_REGISTRY = {
    "steel_slag_h2_co2": steel_slag_h2_co2_superstructure,
    "olivine_carbonation_h2": olivine_carbonation_h2_superstructure,
    "eaf_steel_slag": eaf_steel_slag_superstructure,
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

