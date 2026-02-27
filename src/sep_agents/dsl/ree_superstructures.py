"""
Predefined REE Separation Superstructures
==========================================

Library of canonical REE separation superstructures for GDP optimization.
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


def lree_acid_leach_superstructure() -> Superstructure:
    """LREE recovery from acid-leach liquor.

    Superstructure choices
    ----------------------
    - **Separation method**: SX with D2EHPA *or* SX with PC88A  (XOR)
    - **Product form**: Oxalate precipitation *or* Hydroxide precipitation (XOR)
    - **Scrubbing stage**: Optional (on/off)

    Continuous tuning
    -----------------
    - Organic-to-aqueous ratio for selected SX unit
    - Reagent dosage for selected precipitator
    """
    units = [
        # ── SX alternatives (choose one) ─────────────────────────────
        UnitOp(
            id="sx_d2ehpa",
            type="solvent_extraction",
            params={
                "distribution_coeff": {"Nd+3": 5.0, "Ce+3": 2.0, "La+3": 1.0},
                "organic_to_aqueous_ratio": 1.5,
            },
            inputs=["feed"],
            outputs=["org_extract_d2", "aq_raffinate_d2"],
            optional=True,
            alternatives=["separation_method"],
        ),
        UnitOp(
            id="sx_pc88a",
            type="solvent_extraction",
            params={
                "distribution_coeff": {"Nd+3": 8.0, "Ce+3": 3.5, "La+3": 1.2},
                "organic_to_aqueous_ratio": 1.2,
            },
            inputs=["feed"],
            outputs=["org_extract_pc", "aq_raffinate_pc"],
            optional=True,
            alternatives=["separation_method"],
        ),

        # ── Scrubber (optional) ────────────────────────────────────
        UnitOp(
            id="scrubber",
            type="solvent_extraction",
            params={
                "distribution_coeff": {"Nd+3": 0.5, "Ce+3": 0.3, "La+3": 0.2},
                "organic_to_aqueous_ratio": 0.5,
            },
            inputs=["org_extract"],
            outputs=["scrubbed_org", "scrub_liquor"],
            optional=True,
        ),

        # ── Precipitation alternatives (choose one) ──────────────────
        UnitOp(
            id="oxalate_precip",
            type="precipitator",
            params={
                "residence_time_s": 3600.0,
                "reagent_dosage_gpl": 15.0,
            },
            inputs=["product_org"],
            outputs=["solid_oxalate", "barren_liquor_ox"],
            optional=True,
            alternatives=["product_form"],
        ),
        UnitOp(
            id="hydroxide_precip",
            type="precipitator",
            params={
                "residence_time_s": 1800.0,
                "reagent_dosage_gpl": 8.0,
            },
            inputs=["product_org"],
            outputs=["solid_hydroxide", "barren_liquor_oh"],
            optional=True,
            alternatives=["product_form"],
        ),
    ]

    streams = [
        Stream(
            name="feed",
            phase="liquid",
            temperature_K=298.15,
            pressure_Pa=101325.0,
            composition_wt={
                "H2O(aq)": 1000.0,
                "Nd+3": 15.0,
                "Ce+3": 20.0,
                "La+3": 10.0,
                "HCl(aq)": 50.0,
            },
        ),
        # SX outputs (D2EHPA path)
        Stream(name="org_extract_d2", phase="liquid"),
        Stream(name="aq_raffinate_d2", phase="liquid"),
        # SX outputs (PC88A path)
        Stream(name="org_extract_pc", phase="liquid"),
        Stream(name="aq_raffinate_pc", phase="liquid"),
        # Scrubber
        Stream(name="org_extract", phase="liquid"),
        Stream(name="scrubbed_org", phase="liquid"),
        Stream(name="scrub_liquor", phase="liquid"),
        # Product node (receives from scrubber output or directly from SX)
        Stream(name="product_org", phase="liquid"),
        # Precipitation outputs
        Stream(name="solid_oxalate", phase="solid"),
        Stream(name="barren_liquor_ox", phase="liquid"),
        Stream(name="solid_hydroxide", phase="solid"),
        Stream(name="barren_liquor_oh", phase="liquid"),
    ]

    return Superstructure(
        name="lree_acid_leach",
        base_flowsheet=Flowsheet(
            name="lree_acid_leach_base",
            units=units,
            streams=streams,
        ),
        disjunctions=[
            DisjunctionDef(
                name="separation_method",
                unit_ids=["sx_d2ehpa", "sx_pc88a"],
                description="Choose SX extractant: D2EHPA vs PC88A",
            ),
            DisjunctionDef(
                name="product_form",
                unit_ids=["oxalate_precip", "hydroxide_precip"],
                description="Product precipitation: oxalate vs hydroxide",
            ),
        ],
        fixed_units=[],  # No always-on units; all are in disjunctions or optional
        objective="minimize_opex",
        continuous_bounds={
            "sx_d2ehpa.organic_to_aqueous_ratio": (0.5, 3.0),
            "sx_pc88a.organic_to_aqueous_ratio": (0.5, 3.0),
            "oxalate_precip.reagent_dosage_gpl": (5.0, 30.0),
            "hydroxide_precip.reagent_dosage_gpl": (3.0, 20.0),
        },
        description="LREE recovery from HCl acid-leach liquor. "
                    "Choose between D2EHPA and PC88A extractants, "
                    "optional scrubbing stage, and oxalate vs hydroxide "
                    "product precipitation.",
    )


def simple_sx_precipitator_superstructure() -> Superstructure:
    """Minimal superstructure: SX → Precipitator with optional scrubber.

    Good for quick validation of the GDP solver.  Only 2 configurations
    (scrubber on/off).
    """
    units = [
        UnitOp(
            id="sx_1",
            type="solvent_extraction",
            params={
                "distribution_coeff": {"Nd+3": 5.0, "Ce+3": 2.0, "La+3": 1.0},
                "organic_to_aqueous_ratio": 1.5,
            },
            inputs=["feed"],
            outputs=["org_extract", "aq_raffinate"],
        ),
        UnitOp(
            id="scrubber",
            type="solvent_extraction",
            params={
                "distribution_coeff": {"Nd+3": 0.3, "Ce+3": 0.2, "La+3": 0.1},
                "organic_to_aqueous_ratio": 0.5,
            },
            inputs=["org_extract"],
            outputs=["scrubbed_org", "scrub_liquor"],
            optional=True,
        ),
        UnitOp(
            id="precipitator",
            type="precipitator",
            params={
                "residence_time_s": 3600.0,
                "reagent_dosage_gpl": 10.0,
            },
            inputs=["to_precip"],
            outputs=["solid_product", "barren_liquor"],
        ),
    ]

    streams = [
        Stream(
            name="feed", phase="liquid",
            temperature_K=298.15, pressure_Pa=101325.0,
            composition_wt={
                "H2O(aq)": 1000.0, "Nd+3": 15.0, "Ce+3": 20.0,
                "La+3": 10.0, "HCl(aq)": 50.0,
            },
        ),
        Stream(name="org_extract", phase="liquid"),
        Stream(name="aq_raffinate", phase="liquid"),
        Stream(name="scrubbed_org", phase="liquid"),
        Stream(name="scrub_liquor", phase="liquid"),
        Stream(name="to_precip", phase="liquid"),
        Stream(name="solid_product", phase="solid"),
        Stream(name="barren_liquor", phase="liquid"),
    ]

    return Superstructure(
        name="simple_sx_precip",
        base_flowsheet=Flowsheet(
            name="simple_sx_precip_base",
            units=units,
            streams=streams,
        ),
        disjunctions=[],
        fixed_units=["sx_1", "precipitator"],
        objective="minimize_opex",
        continuous_bounds={
            "sx_1.organic_to_aqueous_ratio": (0.5, 3.0),
            "precipitator.reagent_dosage_gpl": (5.0, 25.0),
        },
        description="Minimal SX → Precipitator with optional scrubber. "
                    "2 configurations for quick GDP validation.",
    )


# Registry of available superstructures
SUPERSTRUCTURE_REGISTRY = {
    "lree_acid_leach": lree_acid_leach_superstructure,
    "simple_sx_precip": simple_sx_precipitator_superstructure,
}


def list_superstructures() -> list[dict]:
    """List available predefined superstructures."""
    result = []
    for name, factory in SUPERSTRUCTURE_REGISTRY.items():
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
            "objective": ss.objective,
        })
    return result
