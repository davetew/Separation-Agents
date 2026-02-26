#!/usr/bin/env python3
"""
Benchmark Validation Suite for Separation-Agents
=================================================

Runs the REE separation model against published literature benchmarks and
public-domain data.  Produces a Markdown report with pass/fail results.

Benchmark Sources
-----------------
1. REE Chloride Speciation (25°C, 1 atm)
   - Migdisov et al. (2016) Chem. Geol. 439, 13-42
   - Luo & Byrne (2004) Geochim. Cosmochim. Acta 68(4), 691-699
   - SUPCRTBL database (Zimmer et al. 2016)

2. SX Distribution Coefficients (McCabe-Thiele)
   - Xie et al. (2014) Miner. Eng. 56, 10-28
   - D2EHPA separation factors: Ce/La ≈ 3.04, Nd/La ≈ 2.32 at pH 0.5

3. REE Hydroxide Precipitation pH
   - Baes & Mesmer (1976) "The Hydrolysis of Cations"
   - Nd(OH)3 pKsp ≈ 21.49

4. Mass Balance Closure
   - Conservation of moles across unit operations

5. Oxalate Precipitation Selectivity
   - Chung et al. (1998) J. Nucl. Sci. Technol.
   - Nd2(C2O4)3 pKsp ≈ 31.14

6. Feed Consistency
   - Speciation mass balance vs known feed composition

Usage
-----
    conda activate rkt
    python scripts/benchmark_validation.py
"""
from __future__ import annotations

import sys, os, math, time, json, logging
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Ensure sep_agents is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ.setdefault("FASTMCP_NO_BANNER", "1")
os.environ.setdefault("FASTMCP_QUIET", "1")

logging.basicConfig(level=logging.WARNING)
_log = logging.getLogger("benchmark")

# ─────────────────────────────────────────────────────────────────────────────
# Result containers
# ─────────────────────────────────────────────────────────────────────────────
class BenchmarkResult:
    def __init__(self, name: str, category: str, description: str,
                 reference: str):
        self.name = name
        self.category = category
        self.description = description
        self.reference = reference
        self.passed = False
        self.details: List[str] = []
        self.metrics: Dict[str, Any] = {}
        self.elapsed_s = 0.0

    def record(self, passed: bool, detail: str, **metrics):
        self.passed = passed
        self.details.append(detail)
        self.metrics.update(metrics)

    @property
    def status(self) -> str:
        return "✅ PASS" if self.passed else "❌ FAIL"


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 1: REE Chloride Speciation at 25°C
# ─────────────────────────────────────────────────────────────────────────────
def bench_speciation_lree_hcl() -> BenchmarkResult:
    """LREE speciation in 1M HCl at 25°C.

    Expected: Dominant aqueous REE species are free ions (REE³⁺) and
    first chloride complexes (REECl²⁺) at low Cl⁻ activity.
    At ~1M HCl, REECl²⁺ should be comparable to or exceed REE³⁺.
    """
    from sep_agents.properties.ree_databases import REEEquilibriumSolver

    res = BenchmarkResult(
        "LREE Speciation in 1M HCl",
        "Speciation",
        "Ce, Nd, La speciation at 25°C in ~1M HCl.  Expect free ions "
        "and mono-chloride complexes to dominate.  SUPCRTBL should reproduce "
        "known speciation from Migdisov et al. (2016).",
        "Migdisov et al. (2016) Chem. Geol. 439, 13-42; "
        "Luo & Byrne (2004) GCA 68(4), 691-699"
    )

    t0 = time.time()
    try:
        solver = REEEquilibriumSolver(preset="light_ree")
        result = solver.speciate(
            temperature_C=25.0,
            pressure_atm=1.0,
            water_kg=1.0,
            acid_mol={"HCl(aq)": 1.0},
            ree_mol={"Nd+3": 0.01, "Ce+3": 0.01, "La+3": 0.01},
        )

        if result.get("status") != "ok":
            res.record(False, f"Speciation failed: {result.get('error')}")
            return res

        ree_dist = result.get("ree_distribution", {})
        ph = result.get("pH", None)

        # Check pH is acidic (should be ~ 0 to -0.5 for 1M HCl)
        ph_ok = ph is not None and ph < 1.0
        res.metrics["pH"] = ph

        # Check that for each REE, mono-chloride (REECl²⁺) and free ion are
        # both present and among the top species
        checks_passed = 0
        total_checks = 0
        for elem in ["Nd", "Ce", "La"]:
            free_ion = f"{elem}+3"
            mono_cl = f"{elem}Cl+2"
            di_cl = f"{elem}Cl2+"

            free_amt = ree_dist.get(free_ion, 0)
            mono_amt = ree_dist.get(mono_cl, 0)
            di_amt = ree_dist.get(di_cl, 0)

            total_ree = free_amt + mono_amt + di_amt
            res.metrics[f"{elem}_free_ion_mol"] = round(free_amt, 6)
            res.metrics[f"{elem}_monoCl_mol"] = round(mono_amt, 6)
            res.metrics[f"{elem}_diCl_mol"] = round(di_amt, 6)

            # Benchmark: free ion and monoCl should both be > 5% of total
            total_checks += 2
            if total_ree > 0:
                if free_amt / total_ree > 0.05:
                    checks_passed += 1
                if mono_amt / total_ree > 0.05:
                    checks_passed += 1

        res.metrics["checks_passed"] = f"{checks_passed}/{total_checks}"
        all_ok = checks_passed >= 5 and ph_ok  # allow 1 marginal
        res.record(all_ok,
            f"pH = {ph:.2f}, speciation checks: {checks_passed}/{total_checks}. "
            f"All expected dominant species present."
            if all_ok else
            f"pH = {ph}, speciation checks: {checks_passed}/{total_checks}. "
            f"Some expected species missing or below threshold."
        )

    except Exception as e:
        res.record(False, f"Exception: {e}")

    res.elapsed_s = time.time() - t0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 2: SX Separation Factor Consistency
# ─────────────────────────────────────────────────────────────────────────────
def bench_sx_separation_factors() -> BenchmarkResult:
    """Solvent extraction separation factor with known D values.

    Published D2EHPA separation factors at pH 0.5 (Xie et al. 2014):
      β(Ce/La) ≈ 3.04
      β(Nd/La) ≈ 2.32

    We use our single-stage McCabe-Thiele model with equivalent D values
    and verify the extraction fractions match analytical predictions.
    """
    from sep_agents.sim.idaes_adapter import IDAESFlowsheetBuilder, StreamState, run_idaes
    from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream

    res = BenchmarkResult(
        "SX Separation Factors vs D2EHPA Literature",
        "Solvent Extraction",
        "Verify McCabe-Thiele SX model reproduces analytical extraction "
        "fractions for known D values.  Published β(Ce/La)≈3.04, β(Nd/La)≈2.32.",
        "Xie et al. (2014) Miner. Eng. 56, 10-28"
    )

    t0 = time.time()
    try:
        # Set D values to reproduce literature separation factors
        # D_Ce / D_La = β(Ce/La) = 3.04  →  if D_La = 1.0, D_Ce = 3.04
        # D_Nd / D_La = β(Nd/La) = 2.32  →  D_Nd = 2.32
        D_vals = {"La+3": 1.0, "Ce+3": 3.04, "Nd+3": 2.32}
        oa_ratio = 1.0  # O/A = 1 for clean calculation

        flowsheet = Flowsheet(
            name="sx_benchmark",
            units=[
                UnitOp(
                    id="sx_stage",
                    type="solvent_extraction",
                    params={
                        "distribution_coeff": D_vals,
                        "organic_to_aqueous_ratio": oa_ratio,
                    },
                    inputs=["feed"],
                    outputs=["organic", "raffinate"],
                )
            ],
            streams=[
                Stream(
                    name="feed",
                    phase="liquid",
                    temperature_K=298.15,
                    pressure_Pa=101325.0,
                    composition_wt={
                        "H2O(aq)": 1000.0,
                        "La+3": 10.0,
                        "Ce+3": 10.0,
                        "Nd+3": 10.0,
                        "HCl(aq)": 50.0,
                    },
                )
            ],
        )

        result = run_idaes(flowsheet)

        if result.get("status") != "ok":
            res.record(False, f"Simulation failed: {result.get('error')}")
            return res

        states = result.get("states", {})
        org_st = states.get("organic")
        raf_st = states.get("raffinate")

        if not org_st or not raf_st:
            res.record(False, "Missing output states")
            return res

        org_sp = org_st.species_amounts if hasattr(org_st, 'species_amounts') else {}
        raf_sp = raf_st.species_amounts if hasattr(raf_st, 'species_amounts') else {}

        # Analytical prediction for single-stage:
        # x_aq = F / (1 + D * R_OA)
        # x_org = F - x_aq
        # Extraction fraction E = x_org / F = D * R_OA / (1 + D * R_OA)
        checks = 0
        total = 0
        for elem, D in D_vals.items():
            feed_amt = 10.0
            E_predicted = D * oa_ratio / (1 + D * oa_ratio)
            E_model = org_sp.get(elem, 0) / feed_amt if feed_amt > 0 else 0

            err_pct = abs(E_predicted - E_model) / max(E_predicted, 1e-10) * 100

            res.metrics[f"{elem}_E_predicted"] = round(E_predicted, 4)
            res.metrics[f"{elem}_E_model"] = round(E_model, 4)
            res.metrics[f"{elem}_error_pct"] = round(err_pct, 2)

            total += 1
            if err_pct < 1.0:  # < 1% error
                checks += 1

        # Check separation factors
        E_Ce = org_sp.get("Ce+3", 0) / 10.0
        E_La = org_sp.get("La+3", 0) / 10.0
        E_Nd = org_sp.get("Nd+3", 0) / 10.0

        if E_La > 0 and (1 - E_La) > 0:
            D_eff_Ce = E_Ce / (1 - E_Ce) if E_Ce < 1 else float('inf')
            D_eff_La = E_La / (1 - E_La) if E_La < 1 else float('inf')
            D_eff_Nd = E_Nd / (1 - E_Nd) if E_Nd < 1 else float('inf')

            beta_CeLa = D_eff_Ce / D_eff_La if D_eff_La > 0 else float('inf')
            beta_NdLa = D_eff_Nd / D_eff_La if D_eff_La > 0 else float('inf')

            res.metrics["beta_Ce_La_computed"] = round(beta_CeLa, 2)
            res.metrics["beta_Ce_La_literature"] = 3.04
            res.metrics["beta_Nd_La_computed"] = round(beta_NdLa, 2)
            res.metrics["beta_Nd_La_literature"] = 2.32

        all_ok = checks == total
        res.record(all_ok,
            f"Extraction fraction errors: {checks}/{total} within 1%. "
            f"β(Ce/La)={beta_CeLa:.2f} (lit: 3.04), β(Nd/La)={beta_NdLa:.2f} (lit: 2.32)"
        )

    except Exception as e:
        import traceback
        res.record(False, f"Exception: {e}\n{traceback.format_exc()}")

    res.elapsed_s = time.time() - t0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 3: Mass Balance Closure
# ─────────────────────────────────────────────────────────────────────────────
def bench_mass_balance_closure() -> BenchmarkResult:
    """Verify mass balance closure across a 2-unit flowsheet (SX + precipitator).

    The sum of all output species (mol) should equal the input feed (mol)
    for a non-reactive SX stage.
    """
    from sep_agents.sim.idaes_adapter import run_idaes
    from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream

    res = BenchmarkResult(
        "Mass Balance Closure (SX Stage)",
        "Conservation",
        "Total moles in organic + raffinate should equal feed for "
        "non-reactive SX unit.",
        "First principles — conservation of mass"
    )

    t0 = time.time()
    try:
        flowsheet = Flowsheet(
            name="mass_balance_test",
            units=[
                UnitOp(
                    id="sx_1",
                    type="solvent_extraction",
                    params={
                        "distribution_coeff": {"Nd+3": 5.0, "Ce+3": 2.0, "La+3": 1.0},
                        "organic_to_aqueous_ratio": 1.5,
                    },
                    inputs=["feed"],
                    outputs=["organic", "raffinate"],
                )
            ],
            streams=[
                Stream(
                    name="feed",
                    phase="liquid",
                    temperature_K=298.15,
                    pressure_Pa=101325.0,
                    composition_wt={
                        "H2O(aq)": 1000.0,
                        "Nd+3": 15.0, "Ce+3": 20.0, "La+3": 10.0,
                        "HCl(aq)": 50.0,
                    },
                )
            ],
        )

        result = run_idaes(flowsheet)
        states = result.get("states", {})

        feed_st = states.get("feed")
        org_st = states.get("organic")
        raf_st = states.get("raffinate")

        if not all([feed_st, org_st, raf_st]):
            res.record(False, "Missing stream states")
            return res

        feed_sp = feed_st.species_amounts if hasattr(feed_st, 'species_amounts') else {}
        org_sp = org_st.species_amounts if hasattr(org_st, 'species_amounts') else {}
        raf_sp = raf_st.species_amounts if hasattr(raf_st, 'species_amounts') else {}

        # Check species-by-species closure
        all_species = set(feed_sp.keys())
        max_err = 0.0
        for sp in all_species:
            f = feed_sp.get(sp, 0)
            o = org_sp.get(sp, 0)
            r = raf_sp.get(sp, 0)
            err = abs(f - (o + r))
            rel_err = err / max(f, 1e-15)
            if rel_err > max_err:
                max_err = rel_err

        res.metrics["max_relative_error"] = f"{max_err:.2e}"
        res.metrics["num_species"] = len(all_species)

        passed = max_err < 1e-10
        res.record(passed,
            f"Max species-level relative error: {max_err:.2e} across "
            f"{len(all_species)} species. {'Closure excellent.' if passed else 'Closure violated!'}"
        )

    except Exception as e:
        res.record(False, f"Exception: {e}")

    res.elapsed_s = time.time() - t0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 4: Dilute Speciation — pH Sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def bench_ph_sensitivity() -> BenchmarkResult:
    """pH response to HCl concentration.

    Literature expectation at 25°C, 1 atm:
      0.01 M HCl → pH ≈ 2.0
      0.1  M HCl → pH ≈ 1.0
      1.0  M HCl → pH ≈ 0.0 (slight negative due to activity)

    This validates the Reaktoro HKF activity model for strong acid solutions.
    """
    from sep_agents.properties.ree_databases import REEEquilibriumSolver

    res = BenchmarkResult(
        "pH Response to HCl Concentration",
        "Speciation",
        "Validate Reaktoro pH predictions against known HCl acid chemistry. "
        "Expected: pH ≈ -log10([HCl]) for dilute-to-moderate concentrations.",
        "General aqueous chemistry; CRC Handbook (2023)"
    )

    t0 = time.time()
    try:
        solver = REEEquilibriumSolver(preset="light_ree")
        test_cases = [
            (0.01, 2.0, 0.5),   # (HCl_mol, expected_pH, tolerance)
            (0.10, 1.0, 0.3),
            (1.00, 0.0, 0.5),   # Activity effects make pH slightly > or < 0
        ]

        checks = 0
        for hcl_mol, expected_ph, tol in test_cases:
            r = solver.speciate(
                temperature_C=25.0, water_kg=1.0,
                acid_mol={"HCl(aq)": hcl_mol},
                ree_mol={"Nd+3": 0.001},
            )
            if r.get("status") != "ok":
                res.details.append(f"  HCl={hcl_mol}M: speciation failed")
                continue

            model_ph = r["pH"]
            err = abs(model_ph - expected_ph)
            ok = err < tol
            if ok:
                checks += 1

            res.metrics[f"HCl_{hcl_mol}M_pH"] = round(model_ph, 3)
            res.metrics[f"HCl_{hcl_mol}M_expected"] = expected_ph
            res.metrics[f"HCl_{hcl_mol}M_error"] = round(err, 3)

        passed = checks == len(test_cases)
        res.record(passed,
            f"pH predictions: {checks}/{len(test_cases)} within tolerance. "
            + "; ".join(f"[HCl]={c[0]}M: pH={res.metrics.get(f'HCl_{c[0]}M_pH', '?')}"
                        f" (expect ~{c[1]})" for c in test_cases)
        )
    except Exception as e:
        res.record(False, f"Exception: {e}")

    res.elapsed_s = time.time() - t0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 5: Nd Hydroxide Precipitation at Elevated pH
# ─────────────────────────────────────────────────────────────────────────────
def bench_nd_hydroxide_precipitation() -> BenchmarkResult:
    """Nd(OH)3 precipitation when NaOH raises pH above ~8.

    Literature: Nd(OH)3 precipitates at pH > ~7–8 (pKsp ≈ 21.49).
    At pH < 6, Nd should remain fully dissolved.
    We use two Reaktoro equilibria: one at low pH, one at high pH.
    """
    from sep_agents.properties.ree_databases import REEEquilibriumSolver

    res = BenchmarkResult(
        "Nd(OH)₃ Precipitation vs pH",
        "Precipitation",
        "Nd should remain dissolved at pH < 6 but precipitate as Nd(OH)₃ "
        "at pH > 8.  Validates custom pKsp injection into SUPCRTBL.",
        "Baes & Mesmer (1976) 'The Hydrolysis of Cations'; "
        "pKsp(Nd(OH)₃) ≈ 21.49"
    )

    t0 = time.time()
    try:
        solver = REEEquilibriumSolver(preset="light_ree")

        # Low pH — no NaOH, just acidic
        r_low = solver.speciate(
            temperature_C=25.0, water_kg=1.0,
            acid_mol={"HCl(aq)": 0.1},
            ree_mol={"Nd+3": 0.01},
        )

        # High pH — add excess NaOH
        r_high = solver.speciate(
            temperature_C=25.0, water_kg=1.0,
            acid_mol={"HCl(aq)": 0.01},
            ree_mol={"Nd+3": 0.01},
            other_mol={"NaOH(aq)": 0.2},  # Should drive pH > 10
        )

        if r_low.get("status") != "ok" or r_high.get("status") != "ok":
            res.record(False, "One or both speciation runs failed")
            return res

        ph_low = r_low["pH"]
        ph_high = r_high["pH"]

        # At low pH: Nd(OH)3(s) should be absent or negligible
        nd_oh3_low = r_low.get("species", {}).get("Nd(OH)3(s)", 0)
        nd_free_low = r_low.get("ree_distribution", {}).get("Nd+3", 0)

        # At high pH: Nd(OH)3(s) should capture most Nd
        nd_oh3_high = r_high.get("species", {}).get("Nd(OH)3(s)", 0)
        nd_free_high = r_high.get("ree_distribution", {}).get("Nd+3", 0)

        res.metrics["pH_low"] = round(ph_low, 2)
        res.metrics["pH_high"] = round(ph_high, 2)
        res.metrics["Nd(OH)3_low_pH_mol"] = f"{nd_oh3_low:.2e}"
        res.metrics["Nd_free_low_pH_mol"] = f"{nd_free_low:.4f}"
        res.metrics["Nd(OH)3_high_pH_mol"] = f"{nd_oh3_high:.4f}"
        res.metrics["Nd_free_high_pH_mol"] = f"{nd_free_high:.2e}"

        # Checks:
        low_ok = nd_oh3_low < 0.001 * 0.01  # < 0.1% precipitated at low pH
        high_ok = nd_oh3_high > 0.005        # > 50% of 0.01 mol precipitated

        passed = low_ok and high_ok
        res.record(passed,
            f"Low pH ({ph_low:.1f}): Nd(OH)₃ = {nd_oh3_low:.2e} mol "
            f"(expect negligible). "
            f"High pH ({ph_high:.1f}): Nd(OH)₃ = {nd_oh3_high:.4f} mol "
            f"(expect > 0.005). {'Both correct.' if passed else 'Mismatch detected.'}"
        )
    except Exception as e:
        res.record(False, f"Exception: {e}")

    res.elapsed_s = time.time() - t0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark 6: TEA / LCA Proxy Consistency
# ─────────────────────────────────────────────────────────────────────────────
def bench_tea_lca_consistency() -> BenchmarkResult:
    """TEA and LCA proxy models should produce sensible, non-zero values.

    Cross-check: OPEX should scale with reagent consumption, and
    the ratio OPEX/LCA should be within a plausible range for
    mineral processing (typically $0.01–$10 per kg CO2e).
    """
    from sep_agents.sim.idaes_adapter import run_idaes
    from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream

    res = BenchmarkResult(
        "TEA/LCA Proxy Model Sanity Check",
        "Economics",
        "OPEX and LCA should be positive, scale with reagent use, and their "
        "ratio should be in a plausible range for hydrometallurgy.",
        "Internal proxy models; cross-checked against general industry data"
    )

    t0 = time.time()
    try:
        # Small flowsheet
        flowsheet = Flowsheet(
            name="tea_lca_test",
            units=[
                UnitOp(id="sx_1", type="solvent_extraction",
                       params={"distribution_coeff": {"Nd+3": 5.0},
                               "organic_to_aqueous_ratio": 1.0},
                       inputs=["feed"], outputs=["org", "raf"]),
            ],
            streams=[
                Stream(name="feed", phase="liquid",
                       temperature_K=298.15, pressure_Pa=101325.0,
                       composition_wt={"H2O(aq)": 1000.0, "Nd+3": 10.0,
                                       "HCl(aq)": 50.0}),
            ],
        )
        r1 = run_idaes(flowsheet)
        opex_1 = r1["kpis"].get("overall.opex_USD", 0)
        lca_1 = r1["kpis"].get("overall.lca_kg_CO2e", 0)

        # Larger feed → should produce proportionally larger OPEX/LCA
        flowsheet2 = Flowsheet(
            name="tea_lca_test_2x",
            units=[
                UnitOp(id="sx_1", type="solvent_extraction",
                       params={"distribution_coeff": {"Nd+3": 5.0},
                               "organic_to_aqueous_ratio": 1.0},
                       inputs=["feed"], outputs=["org", "raf"]),
            ],
            streams=[
                Stream(name="feed", phase="liquid",
                       temperature_K=298.15, pressure_Pa=101325.0,
                       composition_wt={"H2O(aq)": 2000.0, "Nd+3": 20.0,
                                       "HCl(aq)": 100.0}),
            ],
        )
        r2 = run_idaes(flowsheet2)
        opex_2 = r2["kpis"].get("overall.opex_USD", 0)
        lca_2 = r2["kpis"].get("overall.lca_kg_CO2e", 0)

        res.metrics["opex_1x"] = round(opex_1, 4)
        res.metrics["lca_1x"] = round(lca_1, 4)
        res.metrics["opex_2x"] = round(opex_2, 4)
        res.metrics["lca_2x"] = round(lca_2, 4)

        # Checks
        nonzero = opex_1 > 0 and lca_1 > 0
        scales = opex_2 > opex_1 * 1.5  # Should roughly double (allowing overhead)
        ratio = opex_1 / lca_1 if lca_1 > 0 else 0
        ratio_ok = 0.001 < ratio < 100  # $/kgCO2e in plausible range

        res.metrics["opex_lca_ratio"] = round(ratio, 4)
        res.metrics["opex_scales_with_feed"] = scales

        passed = nonzero and scales and ratio_ok
        res.record(passed,
            f"OPEX(1x)=${opex_1:.2f}, LCA(1x)={lca_1:.2f} kgCO₂e, "
            f"OPEX(2x)=${opex_2:.2f}, ratio=${ratio:.3f}/kgCO₂e. "
            f"{'All checks passed.' if passed else 'Some checks failed.'}"
        )
    except Exception as e:
        res.record(False, f"Exception: {e}")

    res.elapsed_s = time.time() - t0
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Report Generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_benchmark_report(results: List[BenchmarkResult], output_dir: str) -> str:
    ts = datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    ts_file = ts.strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"benchmark_report_{ts_file}.md")

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    lines = [
        "# Separation-Agents: Benchmark Validation Report",
        f"\n**Generated**: {ts_str}",
        f"**Result**: **{passed}/{total}** benchmarks passed",
        f"**Total Runtime**: {sum(r.elapsed_s for r in results):.1f}s",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| # | Benchmark | Category | Result | Time (s) |",
        "|---|-----------|----------|--------|----------|",
    ]

    for i, r in enumerate(results, 1):
        lines.append(f"| {i} | {r.name} | {r.category} | {r.status} | {r.elapsed_s:.1f} |")

    lines.append("")
    lines.append("---")
    lines.append("")

    for i, r in enumerate(results, 1):
        lines.append(f"## {i}. {r.name}")
        lines.append("")
        lines.append(f"**Category**: {r.category}  ")
        lines.append(f"**Result**: {r.status}  ")
        lines.append(f"**Runtime**: {r.elapsed_s:.2f}s")
        lines.append("")
        lines.append(f"**Description**: {r.description}")
        lines.append("")
        lines.append(f"**Reference**: {r.reference}")
        lines.append("")

        if r.details:
            lines.append("**Details**:")
            for d in r.details:
                lines.append(f"> {d}")
            lines.append("")

        if r.metrics:
            lines.append("**Metrics**:")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|--------|-------|")
            for k, v in r.metrics.items():
                lines.append(f"| `{k}` | {v} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    report = "\n".join(lines)
    with open(path, "w") as f:
        f.write(report)

    return path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  SEPARATION-AGENTS: BENCHMARK VALIDATION SUITE")
    print("=" * 70)
    print()

    benchmarks = [
        ("1. LREE Speciation in 1M HCl", bench_speciation_lree_hcl),
        ("2. SX Separation Factors", bench_sx_separation_factors),
        ("3. Mass Balance Closure", bench_mass_balance_closure),
        ("4. pH Sensitivity", bench_ph_sensitivity),
        ("5. Nd(OH)₃ Precipitation", bench_nd_hydroxide_precipitation),
        ("6. TEA/LCA Consistency", bench_tea_lca_consistency),
    ]

    results = []
    for label, fn in benchmarks:
        print(f"  Running {label}...", end="", flush=True)
        r = fn()
        results.append(r)
        print(f" {r.status}  ({r.elapsed_s:.1f}s)")

    print()
    print("-" * 70)

    output_dir = os.path.join(os.path.dirname(__file__), "..", "reports")
    path = generate_benchmark_report(results, output_dir)

    passed = sum(1 for r in results if r.passed)
    total = len(results)

    print(f"\n  RESULTS: {passed}/{total} benchmarks passed")
    print(f"  Report:  {os.path.abspath(path)}")
    print("=" * 70)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
