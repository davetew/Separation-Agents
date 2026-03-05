---
description: Generate a comprehensive LaTeX technical report summarizing the valorization analysis
---

# Valorization Report Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given completed simulation and optimization results (from `/process-optimization`), this workflow produces a comprehensive LaTeX technical report.

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

Use the Eagle Mine report (`docs/eagle_mine_valorization_report.tex`) as a structural template. Create a new `.tex` file in `docs/` named descriptively (e.g., `docs/<resource_name>_valorization_report.tex`).

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
- Use the Eagle Mine PFD code as a template, adapting node labels and connections
- Key reactions (chemical equations in `equation` environment)

#### 5. Modeling Approach
- **Three-tier hierarchy table**: Reaktoro GEM, IDAES-PSE, analytical/empirical models
- Subsection on each modeling tier:
  - IDAES Sequential-Modular or EO solver
  - Reaktoro thermodynamic backend (database, limitations)
  - JAX TEA cost model (EAC formula)
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
- **Itemized CAPEX/OPEX table** (by process stage)
- **Revenue table** (production volumes, prices, annual revenue per stream)
- **Levelized economics table** (EAC, total revenue, net value, $/t feedstock)
- **BoTorch optimization table** (base case vs. optimal design variables)
- **Cost sensitivity table** (top-5 cost drivers via JAX ∂EAC/∂param)

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
  node distance=1.8cm and 2.4cm,
  unit/.style={rectangle, draw, rounded corners, minimum width=2.8cm,
               minimum height=0.8cm, align=center, fill=blue!8, font=\small},
  stream/.style={-{Stealth[length=3mm]}, thick},
  product/.style={rectangle, draw, rounded corners, minimum width=2.2cm,
                  minimum height=0.6cm, align=center, fill=green!12,
                  font=\small\itshape},
]
  % Nodes and edges adapted from the specific flowsheet
\end{tikzpicture}
```

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
- [ ] PDF generated successfully in `docs/`
- [ ] All tables render correctly
- [ ] TikZ PFD displays properly
- [ ] Cross-references and hyperlinks work
- [ ] No undefined references or overfull hbox warnings (critical)

---

## Output

- `docs/<resource_name>_valorization_report.tex` — LaTeX source
- `docs/<resource_name>_valorization_report.pdf` — Compiled PDF
