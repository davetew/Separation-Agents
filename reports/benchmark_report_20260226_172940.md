# Separation-Agents: Benchmark Validation Report

**Generated**: 2026-02-26 17:29:40
**Result**: **6/6** benchmarks passed
**Total Runtime**: 5.7s

---

## Summary

| # | Benchmark | Category | Result | Time (s) |
|---|-----------|----------|--------|----------|
| 1 | LREE Speciation in 1M HCl | Speciation | ✅ PASS | 1.2 |
| 2 | SX Separation Factors vs D2EHPA Literature | Solvent Extraction | ✅ PASS | 0.5 |
| 3 | Mass Balance Closure (SX Stage) | Conservation | ✅ PASS | 0.4 |
| 4 | pH Response to HCl Concentration | Speciation | ✅ PASS | 1.6 |
| 5 | Nd(OH)₃ Precipitation vs pH | Precipitation | ✅ PASS | 1.2 |
| 6 | TEA/LCA Proxy Model Sanity Check | Economics | ✅ PASS | 0.8 |

---

## 1. LREE Speciation in 1M HCl

**Category**: Speciation  
**Result**: ✅ PASS  
**Runtime**: 1.23s

**Description**: Ce, Nd, La speciation at 25°C in ~1M HCl.  Expect free ions and mono-chloride complexes to dominate.  SUPCRTBL should reproduce known speciation from Migdisov et al. (2016).

**Reference**: Migdisov et al. (2016) Chem. Geol. 439, 13-42; Luo & Byrne (2004) GCA 68(4), 691-699

**Details**:
> pH = 0.06, speciation checks: 6/6. All expected dominant species present.

**Metrics**:

| Metric | Value |
|--------|-------|
| `pH` | 0.0649 |
| `Nd_free_ion_mol` | 0.002648 |
| `Nd_monoCl_mol` | 0.004462 |
| `Nd_diCl_mol` | 0.001948 |
| `Ce_free_ion_mol` | 0.002678 |
| `Ce_monoCl_mol` | 0.004512 |
| `Ce_diCl_mol` | 0.00197 |
| `La_free_ion_mol` | 0.002773 |
| `La_monoCl_mol` | 0.004671 |
| `La_diCl_mol` | 0.001723 |
| `checks_passed` | 6/6 |

---

## 2. SX Separation Factors vs D2EHPA Literature

**Category**: Solvent Extraction  
**Result**: ✅ PASS  
**Runtime**: 0.50s

**Description**: Verify McCabe-Thiele SX model reproduces analytical extraction fractions for known D values.  Published β(Ce/La)≈3.04, β(Nd/La)≈2.32.

**Reference**: Xie et al. (2014) Miner. Eng. 56, 10-28

**Details**:
> Extraction fraction errors: 3/3 within 1%. β(Ce/La)=3.04 (lit: 3.04), β(Nd/La)=2.32 (lit: 2.32)

**Metrics**:

| Metric | Value |
|--------|-------|
| `La+3_E_predicted` | 0.5 |
| `La+3_E_model` | 0.5 |
| `La+3_error_pct` | 0.0 |
| `Ce+3_E_predicted` | 0.7525 |
| `Ce+3_E_model` | 0.7525 |
| `Ce+3_error_pct` | 0.0 |
| `Nd+3_E_predicted` | 0.6988 |
| `Nd+3_E_model` | 0.6988 |
| `Nd+3_error_pct` | 0.0 |
| `beta_Ce_La_computed` | 3.04 |
| `beta_Ce_La_literature` | 3.04 |
| `beta_Nd_La_computed` | 2.32 |
| `beta_Nd_La_literature` | 2.32 |

---

## 3. Mass Balance Closure (SX Stage)

**Category**: Conservation  
**Result**: ✅ PASS  
**Runtime**: 0.38s

**Description**: Total moles in organic + raffinate should equal feed for non-reactive SX unit.

**Reference**: First principles — conservation of mass

**Details**:
> Max species-level relative error: 0.00e+00 across 5 species. Closure excellent.

**Metrics**:

| Metric | Value |
|--------|-------|
| `max_relative_error` | 0.00e+00 |
| `num_species` | 5 |

---

## 4. pH Response to HCl Concentration

**Category**: Speciation  
**Result**: ✅ PASS  
**Runtime**: 1.62s

**Description**: Validate Reaktoro pH predictions against known HCl acid chemistry. Expected: pH ≈ -log10([HCl]) for dilute-to-moderate concentrations.

**Reference**: General aqueous chemistry; CRC Handbook (2023)

**Details**:
> pH predictions: 3/3 within tolerance. [HCl]=0.01M: pH=2.001 (expect ~2.0); [HCl]=0.1M: pH=1.008 (expect ~1.0); [HCl]=1.0M: pH=0.067 (expect ~0.0)

**Metrics**:

| Metric | Value |
|--------|-------|
| `HCl_0.01M_pH` | 2.001 |
| `HCl_0.01M_expected` | 2.0 |
| `HCl_0.01M_error` | 0.001 |
| `HCl_0.1M_pH` | 1.008 |
| `HCl_0.1M_expected` | 1.0 |
| `HCl_0.1M_error` | 0.008 |
| `HCl_1.0M_pH` | 0.067 |
| `HCl_1.0M_expected` | 0.0 |
| `HCl_1.0M_error` | 0.067 |

---

## 5. Nd(OH)₃ Precipitation vs pH

**Category**: Precipitation  
**Result**: ✅ PASS  
**Runtime**: 1.17s

**Description**: Nd should remain dissolved at pH < 6 but precipitate as Nd(OH)₃ at pH > 8.  Validates custom pKsp injection into SUPCRTBL.

**Reference**: Baes & Mesmer (1976) 'The Hydrolysis of Cations'; pKsp(Nd(OH)₃) ≈ 21.49

**Details**:
> Low pH (1.0): Nd(OH)₃ = 0.00e+00 mol (expect negligible). High pH (13.2): Nd(OH)₃ = 0.0096 mol (expect > 0.005). Both correct.

**Metrics**:

| Metric | Value |
|--------|-------|
| `pH_low` | 1.01 |
| `pH_high` | 13.15 |
| `Nd(OH)3_low_pH_mol` | 0.00e+00 |
| `Nd_free_low_pH_mol` | 0.0083 |
| `Nd(OH)3_high_pH_mol` | 0.0096 |
| `Nd_free_high_pH_mol` | 0.00e+00 |

---

## 6. TEA/LCA Proxy Model Sanity Check

**Category**: Economics  
**Result**: ✅ PASS  
**Runtime**: 0.78s

**Description**: OPEX and LCA should be positive, scale with reagent use, and their ratio should be in a plausible range for hydrometallurgy.

**Reference**: Internal proxy models; cross-checked against general industry data

**Details**:
> OPEX(1x)=$10.21, LCA(1x)=55.92 kgCO₂e, OPEX(2x)=$20.26, ratio=$0.183/kgCO₂e. All checks passed.

**Metrics**:

| Metric | Value |
|--------|-------|
| `opex_1x` | 10.21 |
| `lca_1x` | 55.92 |
| `opex_2x` | 20.26 |
| `lca_2x` | 110.94 |
| `opex_lca_ratio` | 0.1826 |
| `opex_scales_with_feed` | True |

---
