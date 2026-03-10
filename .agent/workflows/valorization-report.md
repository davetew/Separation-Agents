---
description: Generate a comprehensive LaTeX technical report summarizing the valorization analysis
---

# Valorization Report Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given completed simulation and optimization results (from `/process-optimization`), this workflow produces a comprehensive LaTeX technical report.

> [!CAUTION]
> The report **MUST** contain all **11 required sections** listed below. Do not skip or merge sections.
> Required packages: `natbib`, `siunitx`, `enumitem`, `caption`, `longtable`, `booktabs`, `tikz`, `float`.
> Use `\SI{}{}` / `\num{}` from `siunitx` for all quantities and large numbers.
> Compile with `pdflatex` (run twice). Output to `docs/<resource>_<YYYY-MM-DD_HHMM>/`.

> [!IMPORTANT]
> **Mandatory consistency verification — perform BEFORE compiling the final PDF:**
>
> 1. **Topology↔Revenue consistency**: The revenue table, waterfall chart, and levelized economics MUST only include products/credits from the **optimal topology**. If T1 excludes a unit (e.g., CO₂ carbonation, V leach), the revenue section MUST NOT include that unit's products. Products from excluded units should appear only in the trade-study (topology ranking) table under alternative topologies.
> 2. **Number traceability**: Every numeric figure (revenue/t, OPEX/t, CAPEX, NPV, IRR, payback) must be consistent across: topology table → revenue table → levelized economics table → waterfall chart → discussion section → executive summary. If one number changes, ALL must be updated.
> 3. **Cross-document consistency**: If both a report and presentation exist, they must agree on all numeric results. Check all shared metrics after edits.

---

## Step 1: Gather All Input Data

Collect the following from previous workflow stages:

- **Feed characterization** (`/resource-characterization`): composition, speciation, value streams
- **Superstructure definition** (`/superstructure-selection`): candidate topologies, disjunctions
- **Optimization results** (`/process-optimization`): ranked trade-study table, optimal topology KPIs, TEA results, BoTorch results, cost sensitivities
- **PFD images**: process flow diagram PNG/SVG files from `reports/`
- **Stream state tables**: from IDAES simulation results

---

## Step 2: Create the LaTeX Document

Create a timestamped subdirectory under the docs directory and place the `.tex` file there:

```bash
# Generate the output directory name: <resource>_<YYYY-MM-DD_HHMM>
OUTDIR="reports/<resource_name>_$(date +%Y-%m-%d_%H%M)"
mkdir -p "$OUTDIR"
```

Use the Eagle Mine report (`docs/eagle_mine_valorization_report.tex`) as a structural template. Create a new `.tex` file inside that directory named descriptively (e.g., `$OUTDIR/<resource_name>_valorization_report.tex`).

### Required Packages

```latex
\documentclass[11pt,a4paper]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage[numbers]{natbib}
\usepackage{xcolor}
\usepackage{siunitx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usepackage{float}
\usepackage{enumitem}
\usepackage{caption}
\usepackage{longtable}
```

### Required Sections

The report **must** contain the following sections (adapt content to the specific resource):

#### 1. Title & Abstract
- Descriptive title: "Techno-Economic Assessment of [Resource] Valorization"
- Abstract: 1 paragraph summarizing the resource, process, key results (recoveries, net value), and methodology

#### 2. Introduction and Opportunity
- Resource context (location, operator, geological setting)
- Why valorization is attractive (residual value, waste reduction, environmental benefit)
- Scope of the study

#### 3. Raw Material Composition
- Mineralogy table (from `/resource-characterization`)
- Metal grades table
- Bulk chemistry notes

#### 4. Process Description
- One subsection per unit operation stage
- **TikZ process flow diagram** showing all unit operations, streams, and products
- **PFD must show all reagent and utility inputs** (water, acids, bases, precipitants) as distinct nodes feeding into the appropriate unit operations. Use `violet!12` fill and dashed gray arrows to distinguish reagent feeds from process streams.
- Use the Eagle Mine PFD code as a template, adapting node labels and connections
- Key reactions (chemical equations in `equation` environment)

> [!IMPORTANT]
> **Modeling fidelity rule:** All leach and precipitation unit operations **MUST** use `equilibrium_reactor` (Gibbs energy minimization) — not empirical `leach_reactor` or `precipitator` components. Each equilibrium unit must specify `aqueous_elements`, `equilibrium_phases`, `database`, `reagent_name`, and `reagent_dosage_gpl`. This ensures dissolution extents and precipitation yields are predicted from first principles rather than assumed.

#### 5. Modeling Approach
- **Three-tier hierarchy table**: JAX GEM (differentiable speciation), IDAES-PSE (GDP superstructure), JAX TEA + BoTorch (cost optimization)
- Subsection on each modeling tier:
  - JAX GEM solver for equilibrium speciation (SUPCRTBL/HKF database)
  - IDAES Sequential-Modular or EO solver for GDP
  - JAX TEA cost model (EAC formula: CRF × CAPEX, where CRF = r(1+r)^n / ((1+r)^n − 1))
  - BoTorch Bayesian optimization setup

#### 6. GDP Superstructure Analysis
- Superstructure description (disjunctions, optional units)
- Topology enumeration results
- **Topology ranking table** (all feasible topologies ranked by net value)
- Discussion of optimal vs. sub-optimal topologies

#### 7. Cost Model Assumptions
- Plant-level parameters table (throughput, operating days, discount rate, project life)
- Commodity prices table

#### 8. Results
- **Simulation-derived parameters table** (pH, conversions, recoveries, sources)
- **Revenue table** (production volumes, prices, annual revenue per stream)
- **Levelized economics table**: Gross Revenue − OPEX − EAC = **Levelized Net Value** (all in $/t feedstock). Include: CAPEX, project life, discount rate, CRF, EAC derivation.
- **NPV tornado chart** (TikZ figure):
  - One-at-a-time ±30% perturbation of **top 6** parameters
  - Each y-axis label must include the **nominal base value** (e.g., "Cr₂O₃ price ($9,000/t)")
  - Do NOT perturb multiple parameters simultaneously
  - Use `red!60` for adverse, `blue!50` for favorable
  - Invert colors for cost/rate parameters (higher discount rate = adverse)
- **CAPEX breakdown table** (by process area, M USD and % share)
- **OPEX breakdown table** (by cost category, each row must include the **nominal specific unit cost**: e.g., "HCl @ $0.30/kg", "Electricity @ $0.08/kWh", "Labor — 8 FTE @ $35/hr", "Maintenance — 1.1% of CAPEX"). Show M USD/yr and % share.
- **BoTorch optimization table** (base case vs. optimal design variables)
- Narrative: NPV sensitivity interpretation, dominant risk factors

#### 9. Discussion
- Modeling fidelity assessment
- Economic drivers
- Key risks and uncertainties

#### 10. Conclusions
- Numbered list of key findings
- Recommendation for next steps (pilot testing, detailed engineering, etc.)

#### 11. References
- Formatted bibliography (numbered style)
- Include: thermodynamic database sources, cost modeling references, resource-specific literature

---

## Step 3: Generate TikZ Process Flow Diagram

Create a TikZ PFD within the document. Follow this style:

```latex
\begin{tikzpicture}[
  node distance=1.4cm and 1.8cm,
  unit/.style={rectangle, draw, rounded corners, minimum width=2.2cm,
               minimum height=0.7cm, align=center, fill=blue!8, font=\small},
  stream/.style={-{Stealth[length=2.5mm]}, thick},
  product/.style={rectangle, draw, rounded corners, minimum width=1.8cm,
                  minimum height=0.6cm, align=center, fill=green!12,
                  font=\small\itshape},
]
  % Nodes and edges adapted from the specific flowsheet
\end{tikzpicture}
```

> [!IMPORTANT]
> **PFD process correctness rules:**
> - LIMS (magnetic separator) must operate on **dry crushed feed** containing pre-existing magnetite (Fe₃O₄). It is NOT effective on dissolved ions, wüstite (FeO), or paramagnetic minerals.
> - Place LIMS **before** the water mixer — not after slurry formation.
> - Ensure all unit operation placements reflect physical/chemical feasibility.

---

## Step 4: Compile the Report

// turbo
```bash
cd <repo_root>/docs && pdflatex -interaction=nonstopmode <report_name>.tex
```

Run twice to resolve cross-references:
// turbo
```bash
cd <repo_root>/docs && pdflatex -interaction=nonstopmode <report_name>.tex
```

Check the log for errors. Fix any issues and recompile until clean.

---

## Step 5: Verify Output

Confirm:
- [ ] PDF generated successfully in `docs/<resource>_<YYYY-MM-DD_HHMM>/`
- [ ] All 11 required sections present
- [ ] All tables render correctly
- [ ] TikZ PFD displays properly
- [ ] Cross-references and hyperlinks work
- [ ] No undefined references or overfull hbox warnings (critical)
- [ ] `siunitx` used for all quantities

---

## Output

- `reports/<resource_name>_<YYYY-MM-DD_HHMM>/<resource_name>_valorization_report.tex` — LaTeX source
- `reports/<resource_name>_<YYYY-MM-DD_HHMM>/<resource_name>_valorization_report.pdf` — Compiled PDF