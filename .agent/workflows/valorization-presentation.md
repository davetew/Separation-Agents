---
description: Generate a LaTeX Beamer presentation summarizing a valorization study
---

# Valorization Presentation Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given completed valorization analysis and/or report (from `/valorization-report` or `/process-optimization`), this workflow produces a LaTeX Beamer slide deck for stakeholder or academic presentation.

---

## Step 1: Gather Results

Collect key results from previous workflow stages. The presentation distills the technical report into visual, high-impact slides. Focus on:

- Resource description (1 slide)
- Problem statement / opportunity (1 slide)
- Process flow diagram (1–2 slides)
- Key simulation results (2 slides)
- Economics summary (2 slides)
- Optimization results (1–2 slides)
- Conclusions (1 slide)

Target: **15–25 slides** total.

---

## Step 2: Create the Beamer Document

Create a new `.tex` file in `docs/` named `<resource_name>_valorization_presentation.tex`.

### Beamer Preamble

```latex
\documentclass[aspectratio=169]{beamer}

% Theme
\usetheme{metropolis}  % Clean, modern theme
\usepackage{appendixnumberbeamer}

% Packages
\usepackage{booktabs}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning,shapes.geometric}
\usepackage{graphicx}

\sisetup{group-separator={,}, group-minimum-digits=4}

\title{Techno-Economic Assessment of\\[4pt] \textbf{[Resource Name]} Valorization}
\subtitle{[Subtitle: e.g., Metal Recovery, CO₂ Mineralization, and H₂ Production]}
\author{Step Function LLC}
\institute{Separation-Agents Framework}
\date{\today}
```

### Required Slides

#### Title Slide
```latex
\begin{frame}
\titlepage
\end{frame}
```

#### Outline Slide
```latex
\begin{frame}{Outline}
\tableofcontents
\end{frame}
```

#### Section 1: Background & Opportunity (2–3 slides)
- Slide: Resource description (source, location, throughput)
- Slide: Opportunity statement (residual value, environmental benefit)
- Slide: Raw material composition table (condensed from report)

#### Section 2: Process Design (3–4 slides)
- Slide: Simplified TikZ PFD (fewer details than report version)
- Slide(s): Key process stages with reactions (1 slide per major stage)
- Slide: Modeling methodology overview (three-tier hierarchy, 1 slide)

#### Section 3: GDP Analysis (2–3 slides)
- Slide: Superstructure overview (disjunctions, topology count)
- Slide: Topology ranking table (top 5, condensed)
- Slide: Optimal topology description with key differentiators

#### Section 4: Results (3–4 slides)
- Slide: Simulation-derived KPIs (key recoveries, conversions)
- Slide: Economics summary (CAPEX/OPEX bar chart or table)
- Slide: Revenue breakdown (pie chart or stacked bar)
- Slide: **Big number slide** — Net value per tonne feedstock (large, bold, centered)

#### Section 5: Optimization & Sensitivity (2–3 slides)
- Slide: BoTorch results (base vs. optimal, improvement)
- Slide: Cost sensitivity (top-5 drivers, horizontal bar chart or table)
- Optional: Pareto front or convergence plot if available

#### Section 6: Conclusions (1–2 slides)
- Slide: Key findings (numbered list, max 5 items)
- Slide: Next steps / recommendations

#### Acknowledgments / References (1 slide)
- Key references only (3–5 most important)

---

## Step 3: Design Guidelines

Follow these design principles for effective Beamer slides:

1. **One key message per slide** — Avoid overloading
2. **Large fonts for numbers** — Use `\Huge` or `\LARGE` for headline KPIs
3. **Consistent units** — Use `siunitx` throughout
4. **Minimal text** — Bullet points, not paragraphs
5. **Color coding** — Use green for positive metrics (revenue, value), red for costs/risks
6. **TikZ diagrams** should be simplified versions of the report PFD
7. **Tables** should use `booktabs` with max 5–6 rows visible per slide

### Big Number Slide Template

```latex
\begin{frame}{Net Value}
\centering
\vspace{2cm}
{\Huge\textbf{\$XXX/t}}\\[12pt]
{\large Net value per tonne of [resource]}\\[8pt]
{\normalsize Revenue: \$YYY/t \quad Cost: \$ZZZ/t}
\end{frame}
```

---

## Step 4: Compile the Presentation

// turbo
```bash
cd <repo_root>/docs && pdflatex -interaction=nonstopmode <presentation_name>.tex
```

Run twice for cross-references:
// turbo
```bash
cd <repo_root>/docs && pdflatex -interaction=nonstopmode <presentation_name>.tex
```

---

## Step 5: Verify Output

Confirm:
- [ ] PDF generated successfully
- [ ] All slides render correctly in 16:9 aspect ratio
- [ ] TikZ PFD displays properly
- [ ] Tables fit within slide boundaries (no overflows)
- [ ] Fonts are legible (minimum 14pt for body text)
- [ ] Total slide count is within 15–25 range

---

## Output

- `docs/<resource_name>_valorization_presentation.tex` — Beamer source
- `docs/<resource_name>_valorization_presentation.pdf` — Compiled PDF
