---
description: Generate a LaTeX Beamer presentation summarizing a valorization study
---

# Valorization Presentation Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given completed valorization analysis and/or report (from `/valorization-report` or `/process-optimization`), this workflow produces a LaTeX Beamer slide deck for stakeholder or academic presentation, using the **Step Function branded slide format**.

---

## Step 1: Gather Results

Collect key results from previous workflow stages. The presentation distills the technical report into visual, high-impact slides. Focus on:

- Feed characterization YAML / composition tables
- Process flow diagram details (unit operations, streams, conditions)
- GDP superstructure trade-study results (topology ranking)
- BoTorch optimization results (if available)
- TEA results: CAPEX, OPEX, EAC, revenue breakdown, net value per tonne
- Major technical and economic assumptions
- JAX cost-sensitivity analysis (top cost drivers)
- Any generated figures or plots

Target: **15–25 slides** total.

---

## Step 2: Create the Beamer Document

Create a timestamped subdirectory under the presentations directory and place the `.tex` file there:

```bash
# Generate the output directory name: <resource>_<YYYY-MM-DD_HHMM>
OUTDIR="/Users/davidtew/stepfunc/Github/Separation-Agents/presentations/<resource_name>_$(date +%Y-%m-%d_%H%M)"
mkdir -p "$OUTDIR"
```

Create `<resource_name>_valorization_presentation.tex` inside that directory.

### 2.1 — Document Class and Branding

Use the Step Function branding package located at:

```
/Users/davidtew/stepfunc/Presentations/formatting/
```

The document preamble **MUST** follow this structure:

```latex
\documentclass[aspectratio=169]{beamer}

% --- Step Function Branding (replaces any generic Beamer theme) ---
% DO NOT use \usetheme{metropolis}, \usetheme{Madrid}, or any other theme.
% The sf-branding package provides all theming (colors, fonts, title page,
% frame headers, footer, logo, and closing slide).
\usepackage{sf-branding}

% --- Additional packages as needed ---
\usepackage{appendixnumberbeamer}
\usepackage{amsmath,amssymb}
\usepackage{siunitx}
\sisetup{group-separator={,}, group-minimum-digits=4}
\usepackage{graphicx}
% ... (add project-specific packages here)

\title{Techno-Economic Assessment of\\[4pt] \textbf{[Resource Name]} Valorization}
\subtitle{[Subtitle: e.g., Metal Recovery · CO₂ Mineralization · H₂ Production]}
\author{Step Function}
\date{\today}
```

**Key rules:**
- **DO NOT** use `\usetheme{...}` — the `sf-branding` package handles all theming.
- **DO NOT** redefine any colors already provided by the branding (StepDark, StepBlue, StepElectric, StepGreen, StepGrey, StepRed, StepYellow, StepOrange, StepPurple, StepTeal).
- **DO NOT** redefine `\setbeamertemplate{title page}`, `\setbeamertemplate{frametitle}`, or `\setbeamertemplate{footline}` — these are set by the branding.
- You **MAY** define additional project-specific colors for data visualization (e.g. `revenue`, `cost`) that don't clash with brand colors.

### 2.2 — Available Branding Commands

The `sf-branding` package provides these commands — **use them**:

| Command | Purpose |
|---|---|
| `\StepLogo[scale]` | Renders the Step Function logo (auto-placed in frame headers and title page) |
| `\closingSlide` | Inserts a branded "Thank you / Questions?" closing frame |
| `\framesub{text}` | Shorthand for `\framesubtitle{\textit{text}}` |
| `\draft` | Overlays a red "DRAFT!" watermark on the current frame |

### 2.3 — Brand Color Palette

Use these colors for consistent visual identity:

| Color | Hex | Use |
|---|---|---|
| `StepDark` | `#0B1120` | Background (auto-set) |
| `StepTeal` | `#22D3EE` | Frame titles, structure (auto-set) |
| `StepGreen` | `#00CC66` | Bullet items, positive values (auto-set) |
| `StepBlue` | `#0C5394` | Logo fill |
| `StepElectric` | `#3B82F6` | Accent/highlight |
| `StepGrey` | `#64748B` | Subtitles, secondary text |
| `StepRed` | `#EF4444` | Warnings, cost items |
| `StepYellow` | `#F59E0B` | Caution, attention |
| `StepOrange` | `#D97706` | Secondary accent |
| `StepPurple` | `#9333EA` | Tertiary accent |

---

## Step 3: Slide Structure

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

#### Section 2: Process Design (4-5 slides)
- Slide: Simplified TikZ PFD (fewer details than report version)
- Slide(s): Stream table, with pressures, temperatures, flow rates and compositions
- Slide(s): Key process stages with reactions (1 slide per major stage)
- Slide: Modeling methodology overview (three-tier hierarchy, 1 slide)

#### Section 3: GDP Analysis (2–3 slides)
- Slide: Superstructure overview (disjunctions, topology count)
- Slide: Topology ranking table (top 5, condensed)
- Slide: Optimal topology description with key differentiators

#### Section 4: Results (4-5 slides)
- Slide: Simulation-derived KPIs (key recoveries, conversions)
- Slide: Economics summary (CAPEX/OPEX bar chart or table)
- Slide: Revenue breakdown (pie chart or stacked bar)
- Slide: **Big number slide** — Net value per tonne feedstock (large, bold, centered)
- Slide: Major economic and technical assumptions that drive the economics

#### Section 5: Optimization & Sensitivity (2–3 slides)
- Slide: BoTorch results (base vs. optimal, improvement)
- Slide: Cost sensitivity (top-5 drivers, horizontal bar chart or table)
- Optional: Pareto front or convergence plot if available

#### Section 6: Conclusions (1–2 slides)
- Slide: Key findings (numbered list, max 5 items)
- Slide: Next steps / recommendations

#### Closing Slide
Always end with the branded closing slide:
```latex
\closingSlide
```

---

## Step 4: Design Guidelines

Follow these design principles for effective Beamer slides on the dark background:

1. **One key message per slide** — Avoid overloading
2. **Large fonts for numbers** — Use `\Huge` or `\LARGE` for headline KPIs
3. **Consistent units** — Use `siunitx` throughout
4. **Minimal text** — Bullet points, not paragraphs
5. **Color coding** — Use `StepGreen` for positive metrics (revenue, value), `StepRed` for costs/risks
6. **TikZ diagrams** should be simplified versions of the report PFD
7. **Tables** should use `booktabs` with max 5–6 rows visible per slide

### TikZ Styling Guidance

For process flow diagrams and other TikZ graphics on the dark `StepDark` background:
- Use `white` or `StepGrey` for text labels
- Use `StepElectric!15` or `StepTeal!15` for unit-operation box fills
- Use `StepGreen!15` for product boxes
- Use `StepOrange!15` for feed boxes
- Use `white` or `StepGrey` for arrows and stream lines (`draw=StepGrey` or `draw=white`)
- Ensure all TikZ text is explicitly set to `text=white`

### Big Number Slide Template

```latex
\begin{frame}{Net Value}
\centering
\vspace{2cm}
{\Huge\textbf{\textcolor{StepGreen}{\$XXX/t}}}\\[12pt]
{\large Net value per tonne of [resource]}\\[8pt]
{\normalsize
\textcolor{StepGreen}{Revenue: \$YYY/t} \quad
\textcolor{StepRed}{Cost: \$ZZZ/t}}
\end{frame}
```

---

## Step 5: Compile the Presentation

Because `sf-branding` uses `fontspec` (for the logo font), the document **MUST** be compiled with `lualatex`, **NOT** `pdflatex`.

Ensure the formatting directory is on the TeX search path at compile time.

// turbo
```bash
cd "$OUTDIR" && \
  TEXINPUTS="/Users/davidtew/stepfunc/Presentations/formatting//:$TEXINPUTS" \
  lualatex -interaction=nonstopmode <presentation_name>.tex
```

Run twice for cross-references:
// turbo
```bash
cd "$OUTDIR" && \
  TEXINPUTS="/Users/davidtew/stepfunc/Presentations/formatting//:$TEXINPUTS" \
  lualatex -interaction=nonstopmode <presentation_name>.tex
```

---

## Step 6: Verify Output

Confirm:
- [ ] PDF generated successfully
- [ ] Title slide shows Step Function logo and dark branded background
- [ ] All slides render correctly in 16:9 aspect ratio
- [ ] TikZ PFD displays properly on dark background
- [ ] Tikz PFD fits within the slide boundaries (no overflows)
- [ ] Tikz PFD unit operations are not located on top of each other
- [ ] Tables fit within slide boundaries (no overflows)
- [ ] Fonts are legible (minimum 14pt for body text)
- [ ] Total slide count is within 15–25 range
- [ ] Closing slide uses branded `\closingSlide`
- [ ] Footer shows "Step Function Confidential | page#"

---

## Output

All output is saved to the timestamped subdirectory:

```
/Users/davidtew/stepfunc/Github/Separation-Agents/presentations/<resource_name>_YYYY-MM-DD_HHMM/
├── <resource_name>_valorization_presentation.tex   ← Beamer source
└── <resource_name>_valorization_presentation.pdf   ← Compiled PDF
```

Example: `/Users/davidtew/stepfunc/Github/Separation-Agents/presentations/steel_slag_2026-03-05_1130/`