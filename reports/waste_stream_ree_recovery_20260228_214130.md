# Waste Stream REE Recovery — Route Comparison Report

**Generated:** 2026-02-28 21:41  
**Framework:** Separation Agents (EO + GDP)  
**Solver:** IPOPT 3.14.19 with Big-M transformation

---

## Executive Summary

This report evaluates six candidate process topologies for recovering Rare Earth Elements (REEs) from three unconventional waste streams: coal fly ash leachate, acid mine drainage, and red mud leachate.

Each route is evaluated for **OPEX** (proxy operating cost) and **REE recovery**, using the Equation-Oriented (EO) solver for fixed topologies and Generalized Disjunctive Programming (GDP) for automatic topology selection.

---

## Coal Fly Ash Leachate (HCl leach of Class F fly ash)

**Feed REE concentration:** La=0.8, Ce=1.5, Nd=0.5 g/L  
**pH:** 1.5

> Coal fly ash from pulverised-coal power plants contains 200–400 ppm total REE, concentrated in the glassy aluminosilicate fraction. After HCl leaching at 90°C for 4 h, the resulting liquor has ~0.1% REE dissolved alongside significant Fe³⁺, Al³⁺, and Ca²⁺.

### Route Comparison

| Route | Topology | Status | OPEX (proxy) | Recovery KPIs | Time |
|-------|----------|--------|-------------|---------------|------|
| A | SX Only | ✅ | $5.58 | E_Ce=0.833, E_La=0.667, E_Nd=0.778, E_Pr=0.000 | 9.654s |
| B | SX → Precipitator | ✅ | $5.58 | ree_recovery=0.963, E_Ce=0.833, E_La=0.667, E_Nd=0.778 | 3.483s |
| C | IX → SX | ✅ | $15.58 | ree_recovery=0.007, E_Ce=0.833, E_La=0.667, E_Nd=0.778 | 3.537s |
| D | IX → Precipitator | ✅ | $10.58 | ree_recovery=0.007, ree_recovery=0.960 | 3.428s |
| E | SX → Precip → IX | ✅ | $15.58 | ree_recovery=0.476, ree_recovery=0.963, E_Ce=0.833, E_La=0.667 | 3.293s |
| F | **GDP** → ix_1+precip+sx_1 | ✅ | $0.00 | ree_recovery=1.000, ree_recovery=0.431, E_Ce=0.999, E_La=0.999 | 1.418s |

### Analysis

**Recommended route: F (GDP Auto-Select)** — highest aggregate recovery metric (4.428).

---

## Acid Mine Drainage (AMD) from Coal Mine Discharge

**Feed REE concentration:** La=0.05, Ce=0.1, Nd=0.03 g/L  
**pH:** 3.5

> Acidic runoff from coal mines in the Appalachian Basin contains dissolved REEs at 0.5–5 mg/L. Total flow rates can be enormous (100s of GPM), but REE concentration is ultra-low compared to Fe/Mn/Al. IX pre-concentration is typically essential.

### Route Comparison

| Route | Topology | Status | OPEX (proxy) | Recovery KPIs | Time |
|-------|----------|--------|-------------|---------------|------|
| A | SX Only | ✅ | $5.56 | E_Ce=0.833, E_La=0.667, E_Nd=0.778, E_Pr=0.000 | 3.36s |
| B | SX → Precipitator | ✅ | $5.56 | ree_recovery=0.964, E_Ce=0.833, E_La=0.667, E_Nd=0.778 | 3.279s |
| C | IX → SX | ✅ | $15.56 | ree_recovery=0.096, E_Ce=0.833, E_La=0.667, E_Nd=0.778 | 3.323s |
| D | IX → Precipitator | ✅ | $10.56 | ree_recovery=0.096, ree_recovery=0.960 | 3.368s |
| E | SX → Precip → IX | ✅ | $15.56 | ree_recovery=0.477, ree_recovery=0.964, E_Ce=0.833, E_La=0.667 | 3.298s |
| F | **GDP** → ix_1+precip+sx_1 | ✅ | $0.00 | ree_recovery=0.999, ree_recovery=0.402, E_Ce=0.999, E_La=0.999 | 1.384s |

### Analysis

**Recommended route: F (GDP Auto-Select)** — highest aggregate recovery metric (4.398).

⚠️ **Ultra-dilute feed** (<0.5 g/L total REE). IX pre-concentration (Routes C, D) is likely essential for economic viability. Direct SX on this dilute stream would require impractically large O/A ratios.

---

## Red Mud Leachate (HCl leach of Bayer process residue)

**Feed REE concentration:** La=2.0, Ce=3.5, Nd=1.2 g/L  
**pH:** 0.8

> Red mud — the alkaline residue from alumina refining — contains 500–1500 ppm REE, predominantly Ce, La, Nd. After acid leaching, the liquor has very high Fe and Al alongside moderate REE. The gangue-to-REE ratio can exceed 100:1, demanding selective separation chemistry.

### Route Comparison

| Route | Topology | Status | OPEX (proxy) | Recovery KPIs | Time |
|-------|----------|--------|-------------|---------------|------|
| A | SX Only | ✅ | $5.60 | E_Ce=0.833, E_La=0.667, E_Nd=0.778, E_Pr=0.000 | 3.29s |
| B | SX → Precipitator | ✅ | $5.60 | ree_recovery=0.963, E_Ce=0.833, E_La=0.667, E_Nd=0.778 | 3.28s |
| C | IX → SX | ✅ | $15.60 | ree_recovery=0.003, E_Ce=0.833, E_La=0.667, E_Nd=0.778 | 3.292s |
| D | IX → Precipitator | ✅ | $10.60 | ree_recovery=0.003, ree_recovery=0.960 | 3.268s |
| E | SX → Precip → IX | ✅ | $15.60 | ree_recovery=0.476, ree_recovery=0.963, E_Ce=0.833, E_La=0.667 | 3.264s |
| F | **GDP** → ix_1+precip+sx_1 | ✅ | $0.00 | ree_recovery=1.000, ree_recovery=0.457, E_Ce=0.999, E_La=0.999 | 1.384s |

### Analysis

**Recommended route: F (GDP Auto-Select)** — highest aggregate recovery metric (4.454).

✅ **Moderate-to-high REE concentration** (>3 g/L). Direct SX (Route A or B) is viable. Adding precipitation improves product purity at modest additional OPEX.

---

## Methodology

### Simulation Backend
All fixed-topology routes (A–E) were solved using the **Equation-Oriented (EO)** backend (`run_eo()`), which formulates the entire flowsheet as a single Pyomo `ConcreteModel` solved simultaneously by IPOPT.

Route F uses **Generalized Disjunctive Programming (GDP)** via `solve_gdp_eo()`, which wraps each optional unit in a Pyomo `Disjunct` and applies the Big-M transformation before solving with IPOPT.

### Unit Models

| Unit | Model | Key Parameters |
|------|-------|----------------|
| SX | McCabe-Thiele | D(La)=2.0, D(Ce)=5.0, D(Nd)=3.5, O/A=1.0 |
| Precipitator | Recovery-fraction | τ=3600s, reagent=10 g/L |
| IX | Competitive Langmuir | K(La)=1.0, K(Ce)=2.5, K(Nd)=2.0, V_bed=1 m³ |

### Subprocess Isolation
Each route is executed in a separate Python subprocess to prevent IPOPT AMPL-ASL handler corruption between successive solves.

### Limitations
- Distribution coefficients are fixed empirical values, not pH-dependent
- No gangue species (Fe, Al) modelled in EO backend (SM backend supports these via Reaktoro)
- OPEX is a proxy metric; absolute values should not be used for feasibility engineering
- No recycle streams

---

*Report generated by Separation Agents — waste_stream_ree_recovery.py*