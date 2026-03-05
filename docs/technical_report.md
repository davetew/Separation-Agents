# Technical Report: REE Separation Process Modeling in Separation-Agents

**Authors:** Step Function LLC  
**Repository:** `Separation-Agents`  
**Date:** February 2026

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [REE Separation: Process Overview](#2-ree-separation-process-overview)
   - 2.1 [Ore Types and Feed Preparation](#21-ore-types-and-feed-preparation)
   - 2.2 [Hydrometallurgical Processing Stages](#22-hydrometallurgical-processing-stages)
   - 2.3 [Solvent Extraction (SX)](#23-solvent-extraction-sx)
   - 2.4 [Selective Precipitation and Crystallization](#24-selective-precipitation-and-crystallization)
3. [Modeling Architecture](#3-modeling-architecture)
   - 3.1 [Sequential-Modular Flowsheet Solver](#31-sequential-modular-flowsheet-solver)
   - 3.2 [Domain-Specific Language (DSL)](#32-domain-specific-language-dsl)
   - 3.3 [Unit Operation Models](#33-unit-operation-models)
   - 3.4 [Equation-Oriented (EO) Flowsheet Solver](#34-equation-oriented-eo-flowsheet-solver)
   - 3.5 [JAX-Based Equilibrium Solver](#35-jax-based-equilibrium-solver)
   - 3.6 [Generalized Disjunctive Programming (GDP)](#36-generalized-disjunctive-programming-gdp)
4. [Thermodynamic Property Models](#4-thermodynamic-property-models)
   - 4.1 [Reaktoro and the SUPCRTBL Database](#41-reaktoro-and-the-supcrtbl-database)
   - 4.2 [REE Database Configuration](#42-ree-database-configuration)
   - 4.3 [Custom Species Injection](#43-custom-species-injection)
   - 4.4 [Limitations of the Thermodynamic Model](#44-limitations-of-the-thermodynamic-model)
5. [Techno-Economic Analysis (TEA)](#5-techno-economic-analysis-tea)
   - 5.1 [OPEX Estimation Model](#51-opex-estimation-model)
   - 5.2 [Life Cycle Assessment (LCA)](#52-life-cycle-assessment-lca)
   - 5.3 [JAX-Based Differentiable Cost Estimation](#53-jax-based-differentiable-cost-estimation)
6. [Bayesian Optimization with BoTorch](#6-bayesian-optimization-with-botorch)
   - 6.1 [Problem Formulation](#61-problem-formulation)
   - 6.2 [Gaussian Process Surrogate](#62-gaussian-process-surrogate)
   - 6.3 [Acquisition Function](#63-acquisition-function)
   - 6.4 [BO Loop Implementation](#64-bo-loop-implementation)
7. [Report Generation and Automated Analysis](#7-report-generation-and-automated-analysis)
8. [References](#8-references)

---

## 1. Introduction

The separation of Rare Earth Elements (REEs) from one another and from gangue minerals is among the most challenging problems in industrial hydrometallurgy.  The 17 REE elements (15 lanthanides + Y + Sc) share nearly identical ionic radii, outer-shell electronic configurations, and aqueous complex chemistries.  Their separation therefore requires extremely selective multi-stage processes that must be both thermodynamically and kinetically favorable.

This repository — **Separation-Agents** — implements a multi-agent, LLM-guided simulation and optimization framework for REE separation flowsheets.  It couples:

- **Reaktoro** for rigorous Gibbs-energy-minimization equilibrium speciation,
- **IDAES/Pyomo** for sequential-modular flowsheet simulation,
- **BoTorch** (built on GPyTorch and PyTorch) for Bayesian Optimization of continuous design variables,
- **FastMCP** for exposing all functionality as Model Context Protocol (MCP) tools consumable by LLM agents.

The remainder of this document describes the general REE separation process, the technical modeling approach, the provenance and limitations of the property models, and the BO optimization methodology employed.

---

## 2. REE Separation: Process Overview

### 2.1 Ore Types and Feed Preparation

REEs occur in three broad ore categories:

| Ore Type | Primary REEs | Typical Processing |
|----------|-------------|-------------------|
| Bastnäsite (CO₃F⁻) | La, Ce, Pr, Nd (LREEs) | Acid roast → HCl leach |
| Monazite (PO₄³⁻) | Ce, La, Nd, Th | Caustic crack → acid dissolve |
| Ion-adsorption clays | Dy, Y, Tb (HREEs) | In-situ (NH₄)₂SO₄ leach |
| Xenotime (PO₄³⁻) | Y, Dy, Er, Yb | Acid digestion |

In all cases the initial "leach liquor" is an acidic aqueous solution containing dissolved REE ions alongside gangue elements (Fe, Al, Ca, Si, Mn, Mg) and the leaching acid (HCl, H₂SO₄, or HNO₃).  The role of **speciation analysis** is to determine the distribution of REE ions among their dissolved complex forms (e.g., Nd³⁺, NdCl²⁺, NdCl₂⁺, NdCl₃(aq), NdCl₄⁻) at a given pH, temperature, and ionic strength.

### 2.2 Hydrometallurgical Processing Stages

A complete flowsheet typically consists of three to five stages:

```
Leach Liquor
    ↓
┌─────────────────┐
│  Impurity       │  ← removal of Fe, Al, Th by selective precipitation at pH 3–4
│  Removal        │     or solvent extraction with TBP/D2EHPA
└────────┬────────┘
         ↓
┌─────────────────┐
│  Group           │  ← SX using D2EHPA or PC88A to split LREE / MREE / HREE
│  Separation      │     groups based on distribution coefficient gradients
└────────┬────────┘
         ↓
┌─────────────────┐
│  Individual      │  ← multi-stage counter-current SX: 50–200 stages per
│  REE Separation  │     adjacent pair (e.g., Nd/Pr requires ~100 stages)
└────────┬────────┘
         ↓
┌─────────────────┐
│  Precipitation   │  ← conversion to oxide/carbonate/oxalate for sale
│  & Calcination   │
└─────────────────┘
```

### 2.3 Solvent Extraction (SX)

Solvent extraction is the workhorse of REE separation.  An organic phase (kerosene + extractant) contacts the aqueous feed in mixer-settlers.  The extractant selectively complexes with certain REE ions based on the **distribution coefficient** $D$:

$$D_i = \frac{[\text{M}_i]_{\text{org}}}{[\text{M}_i]_{\text{aq}}}$$

where $[\text{M}_i]$ denotes the concentration of metal $i$ in each phase.  For a single ideal stage with organic-to-aqueous volume ratio $R_{O/A}$:

$$x_{\text{aq},i} = \frac{F_i}{1 + D_i \cdot R_{O/A}}$$
$$x_{\text{org},i} = F_i - x_{\text{aq},i}$$

where $F_i$ is the feed amount of species $i$.  Common extractants and their selectivity series:

| Extractant | Selectivity Series | Typical Use |
|-----------|-------------------|-------------|
| D2EHPA | Lu > Yb > … > La (HREE-first) | HREE / LREE group split |
| PC88A/P507 | Same sense as D2EHPA, higher selectivity | Individual HREE separation |
| Cyanex 272 | HREE > LREE, good Nd/Pr | Nd/Pr split |
| HEH[EHP] | Strong HREE preference | High-purity Y, Dy |

The **separation factor** between adjacent elements A and B is:

$$\beta_{A/B} = \frac{D_A}{D_B}$$

Typical values of $\beta$ for adjacent lanthanides are only 1.5–3.0, necessitating 50–200 counter-current stages.

### 2.4 Selective Precipitation and Crystallization

An alternative to SX for certain separations is **selective precipitation**, exploiting differences in solubility products ($K_{sp}$).  For example:

- **Oxalate precipitation**: REE₂(C₂O₄)₃ precipitates have $pK_{sp}$ values ranging from ~25 (La) to ~32 (Sm), enabling fractionation by controlled oxalic acid addition.
- **Hydroxide precipitation**: REE(OH)₃ precipitates at pH values that vary by element ($pK_{sp}$ from ~20 to ~24), though the pH overlap is large.
- **Double sulfate precipitation**: Na₂SO₄ selectively precipitates LREEs as NaREE(SO₄)₂·xH₂O, leaving HREEs in solution.

---

## 3. Modeling Architecture

### 3.1 Sequential-Modular Flowsheet Solver

The solver uses a **sequential-modular** approach via the `IDAESFlowsheetBuilder` class ([`src/sep_agents/sim/idaes_adapter.py`](../src/sep_agents/sim/idaes_adapter.py)).  Key characteristics:

1. **Topological ordering**: Units are sorted so that each unit's input streams are produced by previously solved units.
2. **Stream state propagation**: Each unit receives fully-specified `StreamState` objects (T, P, species amounts in mol, pH, Eh) as inputs and produces updated `StreamState` objects as outputs.
3. **Unit-level solvers**: Each unit type dispatches to a specialized solver method (SX, reactor, crystallizer, separator, mixer, mill, ion exchange, or passthrough).
4. **No iteration on recycles**: The current implementation does not support recycle loops; all streams flow forward in a DAG topology.

```python
# Pseudocode for sequential-modular solve
states = {feed.name: StreamState.from_dsl_stream(feed) for feed in flowsheet.streams}
for unit in topological_sort(flowsheet.units):
    inlets = {name: states[name] for name in unit.inputs}
    outlets = unit_solver(unit, inlets)
    states.update(outlets)
```

### 3.2 Domain-Specific Language (DSL)

Flowsheets are defined using a Pydantic-based DSL with three core objects:

- **`Stream`**: Feed stream definition (temperature, pressure, composition in wt or mol fractions, PSD, pH, etc.)
- **`UnitOp`**: A processing unit with `id`, `type`, `params`, `inputs`, and `outputs`
- **`Flowsheet`**: Container holding `name`, a list of `units`, and a list of `streams`

The DSL is serializable to/from YAML, enabling LLM agents to synthesize flowsheet definitions from natural language descriptions.

### 3.3 Unit Operation Models

Both SM and EO backends implement the following unit models:

| Unit Type | SM Solver | EO Block | Model Basis |
|-----------|-----------|----------|-------------|
| `solvent_extraction` | `_solve_solvent_extraction` | `build_sx_stage` | McCabe-Thiele with species-specific $D$ values |
| `precipitator` / `crystallizer` | `_solve_crystallizer` | `build_precipitator` | SM: Reaktoro equilibrium; EO: recovery-fraction model |
| `ion_exchange` | `_solve_ion_exchange` | `build_ix_column` | Competitive Langmuir isotherm with slack decomposition |
| `reactor` / `leach_reactor` | `_solve_reactor` | — | Reaktoro Gibbs energy minimization (SM only) |
| `separator` / `magnetic_separator` | `_solve_separator` | — | Split fraction, cyclone, or flotation models |
| `mixer` | `_solve_mixer` | — | Mass-weighted mixing |
| `mill` / `crusher` | `_solve_mill` | — | Passthrough with energy consumption KPI |

#### Solvent Extraction Model

The SX model implements single-stage extraction using:

$$\text{amt}_{\text{aq},i} = \frac{\text{amt}_{\text{feed},i}}{1 + D_i \cdot R_{O/A}}$$

where:
- $D_i$ is obtained from a user-supplied dictionary mapping species names to distribution coefficients, or a single scalar applied to all non-aqueous species
- $R_{O/A}$ is the organic-to-aqueous volume ratio
- Aqueous background species (H₂O, H⁺, OH⁻, Cl⁻, HCl(aq), Na⁺, Ca²⁺) are excluded from extraction ($D = 0$)

This is a simplification — real SX systems exhibit pH-dependent, non-ideal $D$ values governed by complexation equilibria.  The model is appropriate for initial flowsheet screening.

#### Reactor / Equilibrium Model

The reactor model delegates to Reaktoro's Gibbs-energy minimization solver:

1. Construct a `ChemicalState` from the inlet `StreamState` species amounts
2. Override T/P from unit parameters if specified
3. Optionally inject reagents (e.g., `reagent_dosage_gpl`)
4. Call `EquilibriumSolver.solve(state)` to find the global Gibbs energy minimum
5. Extract the equilibrium species distribution, pH, and Eh

This provides rigorous equilibrium speciation but does not capture kinetic limitations.

#### Crystallizer / Precipitator Model

The crystallizer combines Reaktoro equilibrium with a phase-separation step:

1. Solve equilibrium (reusing the reactor solver)
2. Partition species into **solid** (species without charge or `(aq)` suffix) and **aqueous** (species with `(aq)`, charged ions)
3. Route each phase to the corresponding outlet

### 3.4 Equation-Oriented (EO) Flowsheet Solver

The EO backend ([`src/sep_agents/sim/eo_flowsheet.py`](../src/sep_agents/sim/eo_flowsheet.py)) formulates the entire flowsheet as a single Pyomo `ConcreteModel` solved simultaneously by IPOPT.

#### Formulation

The EO formulation uses a **flow-composition-temperature-pressure (FcTP)** approach:
- **State variable**: `flow_mol_comp[j]` — molar flow rate per component
- **Unit blocks**: Each unit is a Pyomo `Block` containing `Var`, `Constraint`, and `Expression` objects
- **Inter-unit linking**: Equality constraints connect upstream outlet flows to downstream inlet flows

```python
from sep_agents.sim.eo_flowsheet import run_eo
from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream

result = run_eo(flowsheet, objective="minimize_opex")
# result["status"] == "ok"
# result["kpis"] contains OPEX, LCA, recovery metrics
# result["model"] is the solved Pyomo model
```

#### EO Unit Models

**1. Solvent Extraction (`sx_eo.py`)**: McCabe-Thiele with Pyomo `Var` for $D_j$ and $R_{O/A}$:

$$\text{raffinate}_j = \frac{\text{feed}_j}{1 + D_j \cdot R_{O/A}}, \quad \text{organic}_j = \text{feed}_j - \text{raffinate}_j$$

**2. Precipitator (`precipitator_eo.py`)**: Recovery-fraction model for numerical stability:

$$\text{solid}_j = R_j \cdot \text{feed}_j, \quad \text{barren}_j = (1 - R_j) \cdot \text{feed}_j$$

Avoids Ksp/softplus formulations that cause overflow during gradient computation.

**3. Ion Exchange (`ix_eo.py`)**: Competitive Langmuir isotherm with capped recovery:

$$q_j = q_{\max} \cdot \frac{K_j \cdot c_j}{1 + \sum_{k} K_k \cdot c_k}$$

Recovery is capped at $\leq 1.0$ using slack decomposition: `raw_frac = recovery + slack`, preventing numerical infeasibility when theoretical loading exceeds feed.

#### Advantages over SM

| Feature | SM | EO |
|---------|----|----|  
| Speed (3-unit flowsheet) | ~10–20s | ~1.8s |
| Gradient-based optimization | No (black-box) | Yes (IPOPT) |
| GDP compatibility | Enumeration only | Native Pyomo.GDP |
| Recycles | Not supported | Not supported (planned) |

### 3.5 JAX-Based Equilibrium Solver

To enable end-to-end differentiable flowsheet optimization, a pure-JAX implementation of chemical equilibrium via constrained Gibbs energy minimization (GEM) is provided ([`src/sep_agents/sim/jax_equilibrium.py`](../src/sep_agents/sim/jax_equilibrium.py)).

#### Formulation

The solver minimizes the total Gibbs energy of the system:

$$\min_{n} G(T, P, \mathbf{n}) = \sum_i n_i \mu_i(T, P, \mathbf{n})$$

subject to elemental mass balances $\mathbf{A}\mathbf{n} = \mathbf{b}$ and non-negativity $n_i \ge 0$.

Key differences and advantages compared to the Reaktoro backend:
1. **Fully Differentiable**: Gradients of equilibrium states (pH, Eh, species amounts) with respect to inputs (T, P, feed compositions) can be obtained directly via `jax.grad` or `jax.jacfwd`.
2. **Built-in Thermodynamics**: Implements the extended Debye-Hückel (B-dot) activity model for aqueous species, natively evaluating temperature-dependent parameters (A, B, Bdot coefficients).
3. **Advanced State Corrections**: Natively computes temperature and pressure phase corrections using the Helgeson-Kirkham-Flowers (HKF) equations of state for aqueous species, Holland-Powell thermal properties for minerals, and Peng-Robinson fugacity coefficients for gases.
4. **Log-Space Optimization**: Solves the constrained GEM entirely in log-space ($y_i = \log(n_i + \epsilon)$) using `jaxopt.LBFGSB` to guarantee strict positivity and differentiability without manual constraint projection.

### 3.6 Generalized Disjunctive Programming (GDP)

The GDP module ([`src/sep_agents/opt/gdp_eo.py`](../src/sep_agents/opt/gdp_eo.py)) enables **superstructure optimization** — simultaneously selecting the best process topology and optimizing continuous operating parameters.

#### Superstructure Formulation

A superstructure is a network containing:
- **Fixed units**: Always present in the flowsheet
- **Optional units**: May be active or bypassed (binary decision)
- **XOR disjunctions**: Exactly one of a set of alternatives must be active

These are encoded using Pyomo.GDP's `Disjunct` and `Disjunction` objects:

```python
from sep_agents.dsl.schemas import Superstructure, DisjunctionDef
from sep_agents.opt.gdp_eo import solve_gdp_eo

ss = Superstructure(
    name="opt",
    base_flowsheet=flowsheet,
    fixed_units=["sx_1"],
    disjunctions=[DisjunctionDef(name="sep", unit_ids=["sx_alt", "ix_alt"])],
    objective="maximize_recovery",
)
result = solve_gdp_eo(ss)
# result.active_units, result.bypassed_units, result.objective_value
```

#### Big-M Transformation

The GDP model is converted to a standard MINLP using the Big-M relaxation:

1. Each `Disjunct` (unit active / unit bypassed) contains the full set of unit constraints or passthrough constraints
2. `Disjunction` constraints enforce XOR selection
3. `TransformationFactory('gdp.bigm').apply_to(model)` relaxes the disjuncts into linear inequalities with binary indicator variables
4. IPOPT solves the resulting NLP relaxation

#### Bypass Semantics

When a unit is bypassed, its outlet flows are set equal to its inlet flows via passthrough constraints:

$$\text{outlet}_j = \text{inlet}_j \quad \forall j \in \text{components}$$

This ensures mass balance closure regardless of topology selection.

#### Supported Objectives

| Objective | Description |
|-----------|-------------|
| `minimize_opex` | Minimize total proxy OPEX across active units |
| `maximize_recovery` | Maximize total REE recovery fraction |
| `none` | Feasibility solve only |

---

## 4. Thermodynamic Property Models

### 4.1 Reaktoro and the SUPCRTBL Database

All thermodynamic calculations are performed using [Reaktoro](https://reaktoro.org/) (v2.x), a C++/Python library for chemical thermodynamics and kinetics.  Reaktoro solves the **constrained Gibbs energy minimization** problem:

$$\min_{n} G(T, P, \mathbf{n}) = \sum_i n_i \mu_i(T, P, \mathbf{n})$$

subject to elemental mass balance constraints $\mathbf{A}\mathbf{n} = \mathbf{b}$, where:
- $n_i$ is the amount (mol) of species $i$
- $\mu_i$ is the chemical potential of species $i$
- $\mathbf{A}$ is the formula matrix (elements × species)
- $\mathbf{b}$ is the element amount vector

The implementation uses the **SUPCRTBL** database by default.  SUPCRTBL (SUPCRT with revised Berman–Brown–Helgeson data) is a comprehensive thermodynamic database for aqueous species, minerals, and gases.  It is derived from:

> **Provenance**: Zimmer, K., et al. (2016). *SUPCRTBL: A revised and extended thermodynamic dataset and software package of SUPCRT92.* Computers & Geosciences, 90, 97-111.

Key characteristics:
- **300+ REE aqueous species** covering all 14 stable lanthanides plus Y and Sc
- Supports chloride, sulfate, nitrate, fluoride, carbonate, and hydroxide complexes
- Temperature range: 0–600°C; pressure range: 1–5000 bar
- Activity coefficient model: extended Debye-Hückel (HKF equation of state for aqueous species)

### 4.2 REE Database Configuration

The `ree_databases.py` module ([`src/sep_agents/properties/ree_databases.py`](../src/sep_agents/properties/ree_databases.py)) provides pre-configured Reaktoro `ChemicalSystem` objects optimized for REE separation:

| Preset | Elements | Typical Application |
|--------|----------|-------------------|
| `light_ree` | La, Ce, Pr, Nd + base + gangue | Bastnäsite / monazite processing |
| `heavy_ree` | Sm–Lu, Y, Sc + base + gangue | Ion-adsorption clay, xenotime |
| `full_ree` | All 16 REE + base + gangue | Full rare earth circuit |
| Custom | User-selected elements | Specific separations |

Each preset automatically includes:
- **Base elements**: H, O, Na, K, Cl, C, S, N, P, F
- **Gangue elements**: Fe, Al, Ca, Si, Mg, Mn
- **Phases**: Aqueous (all species via `rkt.speciate()`), compatible minerals, and gas phase

### 4.3 Custom Species Injection

A significant challenge for REE process modeling is that many REE precipitates (hydroxides, oxalates) are not present in the SUPCRTBL database.  The `build_ree_system()` function dynamically injects these species using published solubility products:

#### REE Hydroxide Injection

For each REE hydroxide, the standard Gibbs energy of formation ($G^0_{\text{product}}$) is derived from the dissolution equilibrium:

$$\text{REE(OH)}_3\text{(s)} \rightleftharpoons \text{REE}^{3+}\text{(aq)} + 3\text{OH}^-\text{(aq)}$$

$$\Delta G^0_{\text{rxn}} = -RT \cdot pK_{sp} \cdot \ln(10)$$

$$G^0_{\text{product}} = \Delta G^0_{\text{rxn}} + G^0_{\text{REE}^{3+}} + 3 \cdot G^0_{\text{OH}^-}$$

where $G^0$ values for the aqueous ions are taken from the SUPCRTBL database at $T = 298.15$ K, $P = 1$ bar.

| Species | $pK_{sp}$ | Source |
|---------|-----------|--------|
| La(OH)₃(s) | 20.7 | Baes & Mesmer (1976) |
| Ce(OH)₃(s) | 19.7 | Baes & Mesmer (1976) |
| Pr(OH)₃(s) | 23.47 | Estimated from ionic radius trend |
| Nd(OH)₃(s) | 21.49 | Baes & Mesmer (1976) |
| Sm(OH)₃(s) | 22.08 | Estimated |

#### REE Oxalate Injection

A more sophisticated approach is used for oxalate species.  A **pseudo-element "Ox"** is defined to prevent Reaktoro's Gibbs minimizer from converting oxalic acid carbon into thermodynamically-favorable inorganic species (CO₂, CH₄, graphite):

1. Aqueous oxalate species (C₂O₄²⁻, HC₂O₄⁻, H₂C₂O₄(aq)) are defined with $G^0$ values derived from NBS tables and acid dissociation constants:
   - $pK_{a2} = 4.14$ for HC₂O₄⁻ → H⁺ + C₂O₄²⁻
   - $pK_{a1} = 1.25$ for H₂C₂O₄ → H⁺ + HC₂O₄⁻

2. Solid REE₂(C₂O₄)₃ precipitates are injected using the dissolution equilibrium:

$$\text{REE}_2\text{(C}_2\text{O}_4\text{)}_3\text{(s)} \rightleftharpoons 2\text{REE}^{3+} + 3\text{C}_2\text{O}_4^{2-}$$

| Species | $pK_{sp}$ | Source |
|---------|-----------|--------|
| La₂(C₂O₄)₃(s) | 25.0 | Martell & Smith (2004) |
| Ce₂(C₂O₄)₃(s) | 28.0 | Martell & Smith (2004) |
| Pr₂(C₂O₄)₃(s) | 30.82 | Estimated from trend |
| Nd₂(C₂O₄)₃(s) | 31.14 | Martell & Smith (2004) |
| Sm₂(C₂O₄)₃(s) | 32.0 | Estimated |

The increasing $pK_{sp}$ from La to Sm means heavier LREEs precipitate more readily, enabling selective fractionation by controlled oxalate addition.

### 4.4 Limitations of the Thermodynamic Model

| Limitation | Impact | Mitigation Path | Status |
|-----------|--------|----------------|--------|
| **Constant $G^0$** for injected species | Temperature dependence neglected | $C_p(T)$ polynomials from NIST | Open |
| **No SX thermodynamic model** | $D$ values are empirical inputs | COSMO-RS or D-pH correlations | Open |
| **No kinetic model** | All reactors assume equilibrium | Residence-time conversion models | Open |
| **No activity model for organics** | Organic-phase non-ideality ignored | UNIFAC or NRTL | Open |
| **Limited mineral database** | Only injected REE solids available | Expand precipitate types | Open |
| ~~Single-stage SX only~~ | ~~Cascades not modeled~~ | ~~McCabe-Thiele cascade solver~~ | ✅ Resolved (EO) |
| **Molar mass fallbacks** | Unknown species default to 100 g/mol | Expand lookup tables | Open |
| **No recycle convergence** | Forward-only stream topology | Tear-stream iteration in EO | Open |

---

## 5. Techno-Economic Analysis (TEA)

### 5.1 OPEX Estimation Model

The OPEX model ([`src/sep_agents/cost/tea.py`](../src/sep_agents/cost/tea.py)) computes proxy operating costs by summing:

1. **Reagent costs**: Each feed stream species with a known price is costed at its molar consumption rate:

$$\text{Cost}_{\text{reagent}} = \sum_{s \in \text{feeds}} \sum_{i \in \text{species}} n_i \cdot \frac{M_i}{1000} \cdot p_i$$

where $n_i$ is mol consumed, $M_i$ is molar mass (g/mol), and $p_i$ is USD/kg.

2. **Pumping energy**: Water volumes are costed at $0.05/m³ as a proxy for pumping energy.

3. **Unit energy consumption**: Heuristic power models for each unit type:
   - **Crystallizer/Precipitator**: kWh = (residence_time / 3600) × 5.0 + (reagent_dosage × 0.1)
   - **Solvent extraction**: kWh = 2.0 × number_of_stages
   - Electricity price: $0.08/kWh

#### Reagent Price Table

| Reagent | Price (USD/kg) | Source |
|---------|---------------|--------|
| HCl(aq) | 0.20 | Industrial bulk pricing proxy |
| H₂SO₄(aq) | 0.15 | Byproduct pricing |
| HNO₃(aq) | 0.35 | Industrial bulk |
| NaOH(aq) | 0.50 | Chlor-alkali output |
| Oxalic acid | 1.20 | Industrial grade |
| D2EHPA | 8.50 | Specialty extractant |
| PC88A | 12.00 | Specialty extractant |
| Cyanex 272 | 25.00 | Specialty extractant |

### 5.2 Life Cycle Assessment (LCA)

The LCA model ([`src/sep_agents/cost/lca.py`](../src/sep_agents/cost/lca.py)) mirrors the TEA structure but uses **cradle-to-gate CO₂ equivalent emission factors** in place of prices:

| Reagent | kg CO₂e/kg | Basis |
|---------|-----------|-------|
| HCl(aq) | 1.10 | Energy-intensive synthesis |
| NaOH(aq) | 1.20 | Chlor-alkali process |
| HNO₃(aq) | 2.50 | N₂O emissions during production |
| NH₃(aq) | 2.10 | Haber-Bosch process |
| Grid electricity | 0.45 kg/kWh | Global average grid mix |

These are **proxy values** suitable for screening-level comparisons.  For publication-grade LCA, process-specific emission factors from ecoinvent or GREET databases should be used.

### 5.3 JAX-Based Differentiable Cost Estimation

To directly interface with gradient-descent and Bayesian Optimization frameworks, a fully differentiable Techno-Economic Analysis (TEA) module is implemented ([`src/sep_agents/cost/jax_tea.py`](../src/sep_agents/cost/jax_tea.py)). This completely pure-JAX approach allows for backpropagating gradients of the Equivalent Annualized Cost (EAC) directly to flowsheet operating parameters (e.g., $d(EAC)/d(reagent\_dosage)$) in a single backward pass.

#### Capabilities

The JAX cost module calculates proxy CAPEX and OPEX for the lifecycle of a flowsheet:
1. **Mining Cost**: Functions evaluating overall plant throughput ($tpd$), depth multipliers ($cost/m$), and strip ratios mapping the penalty of waste-to-ore handling.
2. **Comminution Cost**: Predicts crushing and milling capacity parameters utilizing a differentiable adaptation of the Bond Work Index equation to scale baseline electricity requirements.
3. **Leaching Cost**: Models the core hydrometallurgical extraction block, combining residence time-driven CAPEX with temperature-dependent heating penalties (DT over ambient) and linear acid-dosing costs.
4. **Processing Cost**: Approximates solvent extraction stages via a scalable sizing proxy anchored to flow rate $m^3/h$, adding reagent bounds for precipitation circuits.

#### Key Cost Assumptions & Proxy References

*   **Mining Scale**: CAPEX proxy scales sub-linearly with plant capacity (the 0.6 rule of thumb approximation; $Cost \propto Capacity^{0.6}$), while OPEX evaluates linearly per ton of feed ore. Base operating assumption is 330 operational days a year.
*   **Comminution Hardness**: Hardness scales energy usage from a 10 kWh/t baseline dynamically adjusted by the Bond Work Index multiplying over a local electricity cost proxy ($0.08 / kWh).
*   **Reagent Valuation**: Modeled identically to the base TEA, assuming bulk industrial scale pricing (e.g. Acid: $150 / ton, SX stages scaling dynamically). 

> **Provenance**: The parameters utilized by `jax_tea.py` serve as synthetic proxy placeholders designed to reflect broad, heuristic-level estimations comparable to preliminary O'Hara methods for feasibility studies. 

---

## 6. Bayesian Optimization with BoTorch

### 6.1 Problem Formulation

The optimization problem is formulated as:

$$\min_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$$

where:
- $\mathbf{x}$ is a vector of continuous design variables (e.g., organic-to-aqueous ratio, reagent dosage, temperature)
- $\mathcal{X} = [\mathbf{lb}, \mathbf{ub}]$ is the box-constrained feasible region
- $f(\mathbf{x})$ is the black-box objective (e.g., OPEX from a full flowsheet simulation)

This is a **derivative-free** optimization problem because each $f(\mathbf{x})$ evaluation requires solving the sequential-modular flowsheet (Reaktoro calls, mass balances, TEA/LCA), and no analytical gradient is available.

### 6.2 Gaussian Process Surrogate

The implementation ([`src/sep_agents/opt/bo.py`](../src/sep_agents/opt/bo.py)) uses BoTorch's `SingleTaskGP` — an exact Gaussian Process with:

- **Prior mean**: Constant (zero after Y-standardization)
- **Kernel**: Matérn 5/2 with automatic relevance determination (ARD), i.e., one length-scale per dimension
- **Likelihood**: Gaussian with learned noise variance
- **Hyperparameter fitting**: Marginal log-likelihood maximization via `fit_gpytorch_mll` (L-BFGS-B optimizer)

The GP provides a posterior predictive distribution:

$$f(\mathbf{x}) \mid \mathcal{D}_n \sim \mathcal{N}\big(\mu_n(\mathbf{x}),\; \sigma^2_n(\mathbf{x})\big)$$

where $\mathcal{D}_n = \{(\mathbf{x}_i, y_i)\}_{i=1}^n$ is the dataset of prior evaluations.

#### Y-Standardization

Before fitting, the target values are standardized:

$$\tilde{y}_i = \frac{y_i - \bar{y}}{s_y}$$

This improves GP hyperparameter fitting stability, especially for objectives whose magnitudes vary significantly across the feasible domain.

For minimization problems, the normalized targets are negated ($\tilde{y} \to -\tilde{y}$) so that the acquisition function always operates in maximization mode.

### 6.3 Acquisition Function

The optimizer uses **Log Expected Improvement (LogEI)**:

$$\text{LogEI}(\mathbf{x}) = \log\Big(\mathbb{E}\big[\max(f(\mathbf{x}) - f^*_{\text{best}}, \; 0)\big]\Big)$$

LogEI was introduced by Ament et al. (2024) as a numerically-stable replacement for standard EI.  It avoids the catastrophic cancellation and underflow issues that arise when $\sigma_n(\mathbf{x})$ is very small or $f^*_{\text{best}}$ is far from $\mu_n(\mathbf{x})$.

> **Reference**: Ament, S., et al. (2024). *Unexpected Improvements to Expected Improvement for Bayesian Optimization.* NeurIPS 2024. [arXiv:2310.20708](https://arxiv.org/abs/2310.20708)

The acquisition function is optimized via multi-start L-BFGS-B with 5 restarts and 20 raw samples for initialization.

### 6.4 BO Loop Implementation

The optimization loop proceeds as follows:

```
┌───────────────────────────────────────────────┐
│ 1. INITIALIZATION (Latin Hypercube Sampling)  │
│    Generate n_initial points in [0,1]^d       │
│    Evaluate objective at each point            │
├───────────────────────────────────────────────┤
│ 2. SEQUENTIAL BO ITERATIONS                    │
│    For i = 1 to n_iters:                       │
│      a. Standardize Y → (Y - μ) / σ           │
│      b. Fit SingleTaskGP to (X, Y_norm)        │
│      c. Construct LogEI acquisition function   │
│      d. Optimize LogEI → candidate x*          │
│      e. Evaluate objective f(x*)               │
│      f. Augment dataset: D ← D ∪ (x*, f(x*))  │
├───────────────────────────────────────────────┤
│ 3. RETURN best (x, f(x)) from all evaluations │
└───────────────────────────────────────────────┘
```

**Design choices:**

| Parameter | Default Value | Rationale |
|-----------|--------------|-----------|
| `n_initial` | 5 | Provides adequate space coverage for low-d (2–5) problems |
| `n_iters` | 15 | Balances exploration/exploitation for flowsheet-cost objectives |
| Initial sampling | Latin Hypercube (scipy `qmc.LatinHypercube`) | Better space-filling than random |
| Candidate optimization | 5 restarts, 20 raw samples | Adequate for smooth acquisition landscapes |
| Working space | $[0, 1]^d$ (normalized) | Ensures numerical stability; physical rescaling at evaluation |

**Limitations:**
- No constraint handling (all constraints must be embedded in the objective as penalty terms)
- No multi-objective support (single scalar objective only)
- No batch-parallel evaluations (sequential acquisition only)
- GP scaling: computational cost is $O(n^3)$ per iteration, practical up to ~500 evaluations

---

## 7. Report Generation and Automated Analysis

The automated report generator ([`src/sep_agents/report.py`](../src/sep_agents/report.py)) produces a self-contained Markdown document after each workflow execution:

1. **Process Flow Diagram**: Matplotlib-rendered PNG with color-coded nodes (feeds, units, products), edge-to-edge arrows with stream-name labels, and species/mass annotations
2. **Stream State Table**: Temperature, pressure, flow rate, pH, and top species for each stream, with type classification (Feed/Internal/Product)
3. **Mass Balance**: Separation of output into valuable REE product vs waste (water, residual ions)
4. **Output-Normalized Metrics**: OPEX and LCA per kg of REE product, estimated REE market value per kg of input ore, and net value
5. **Optimization Results**: Baseline vs optimized comparison with convergence history

Reports are timestamped to prevent overwriting: `analysis_report_YYYYMMDD_HHMMSS.md`.

---

## 8. References

1. **Reaktoro**: Leal, A.M.M. (2015). *Reaktoro: A unified framework for modeling chemically reactive systems.* [reaktoro.org](https://reaktoro.org/)

2. **SUPCRTBL**: Zimmer, K., Zhang, Y., Lu, P., Chen, Y., Zhang, G., Dalkilic, M., & Zhu, C. (2016). *SUPCRTBL: A revised and extended thermodynamic dataset and software package of SUPCRT92.* Computers & Geosciences, 90, 97–111.

3. **BoTorch**: Balandat, M., et al. (2020). *BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization.* NeurIPS 2020. [arxiv.org/abs/1910.06403](https://arxiv.org/abs/1910.06403)

4. **LogEI**: Ament, S., et al. (2024). *Unexpected Improvements to Expected Improvement for Bayesian Optimization.* NeurIPS 2024. [arxiv.org/abs/2310.20708](https://arxiv.org/abs/2310.20708)

5. **IDAES**: Miller, D.C., et al. (2018). *Next Generation Multi-Scale Process Systems Engineering Framework.* Computer Aided Chemical Engineering, 44, 2209–2214.

6. **REE Hydroxide Ksp**: Baes, C.F., & Mesmer, R.E. (1976). *The Hydrolysis of Cations.* Wiley.

7. **REE Oxalate Ksp**: Martell, A.E., & Smith, R.M. (2004). *Critical Stability Constants.* Plenum Press. (via NIST Critically Selected Stability Constants Database)

8. **REE Separation Chemistry**: Xie, F., Zhang, T.A., Dreisinger, D., & Doyle, F. (2014). *A critical review on solvent extraction of rare earths from aqueous solutions.* Minerals Engineering, 56, 10–28.

9. **GPyTorch**: Gardner, J., et al. (2018). *GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration.* NeurIPS 2018.

---

*This document describes the modeling approach as implemented in the repository at the time of writing.  The proxy TEA/LCA models and thermodynamic extensions are intended for screening-level analysis and should be validated against plant data before engineering design use.*
