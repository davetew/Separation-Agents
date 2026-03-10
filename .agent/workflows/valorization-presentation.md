---
description: Generate a LaTeX Beamer presentation summarizing a valorization study
---

# Valorization Presentation Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

Given completed valorization analysis and/or report (from `/valorization-report` or `/process-optimization`), this workflow produces a LaTeX Beamer slide deck for stakeholder or academic presentation, using the **Step Function branded slide format**.

> [!CAUTION]
> **Critical constraints — violations will produce incorrect output:**
> - **DO NOT** use `\usetheme{metropolis}` or any other Beamer theme — use `\usepackage{sf-branding}` ONLY
> - **DO NOT** compile with `pdflatex` — use `lualatex` with `TEXINPUTS` set to `/Users/davidtew/stepfunc/Presentations/formatting//`
> - **DO NOT** save output to `docs/` — save to `presentations/<resource>_YYYY-MM-DD_HHMM/`
> - **DO** end with `\closingSlide` (not a custom closing frame)
> - **DO** include a big number slide using the template in Step 3
> - **DO** include `\tableofcontents` outline slide
> - **DO** style all TikZ for dark background: `draw=white` or `draw=StepColor!60`, `fill=StepColor!15`
> - **DO** use `text=black` inside light-filled boxes (e.g., `fill=StepElectric!15`); use `text=white` only for labels outside boxes or on dark fills
> - **DO** size PFD nodes compactly (max `minimum width=1.6cm`, `node distance=0.8cm and 1.1cm`, `font=\tiny`) to fit on slide

> [!IMPORTANT]
> **Mandatory consistency verification — perform BEFORE compiling the final PDF:**
>
> 1. **Topology↔Revenue consistency**: The revenue table, waterfall chart, and big-number headline MUST only include products/credits from the **optimal topology**. If T1 excludes a unit (e.g., CO₂ carbonation, V leach), the revenue slides MUST NOT include that unit's products or credits. Products from excluded units should appear only in *alternative* topology rows in the trade-study table.
> 2. **Topology↔PFD consistency**: The PFD diagram must match the identified optimal topology. Bypassed units should use the `skip` style, active units the `unit` style. No active units should appear in the PFD that are bypassed in the optimal topology.
> 3. **Number consistency**: Every numeric figure (revenue/t, OPEX/t, CAPEX, NPV, IRR, payback) must trace back to Table → Waterfall → Big-Number → Summary. If any single number changes, ALL downstream appearances must be updated.
> 4. **Cross-document consistency**: If both a report and presentation exist, they must agree on all numeric results. Compile and spot-check both after edits.

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
- **PFD must show all reagent and utility inputs** (water, HCl, NaOH, NH₄Cl, etc.) as distinct nodes feeding into the appropriate unit operations. Use `StepPurple!15` fill and dashed arrows (`rin` style) to distinguish reagent feeds from process streams.
- Slide(s): Stream table, with pressures, temperatures, flow rates and compositions
- Slide(s): Key process stages with reactions (1 slide per major stage)
- Slide: Modeling methodology overview (three-tier hierarchy, 1 slide)

> [!IMPORTANT]
> All leach and precipitation units must be modeled via `equilibrium_reactor` (Gibbs energy minimization). Do not use empirical `leach_reactor` or `precipitator`. Ensure the modeling methodology slide reflects this.

#### Section 3: GDP Analysis (2–3 slides)
- Slide: Superstructure overview (disjunctions, topology count)
- Slide: Topology ranking table (top 5, condensed)
- Slide: Optimal topology description with key differentiators

#### Section 4: Results (5–7 slides)
- Slide: Simulation-derived KPIs (key recoveries, conversions)
- Slide: Revenue breakdown (table or stacked bar)
- Slide: **Levelized net value** — big number (Revenue − OPEX − EAC) with a TikZ **waterfall bridge chart** showing each revenue stream (+Cr₂O₃, +V₂O₅, etc.) then cost deductions (−OPEX, −EAC) to final net value
- Slide: Key economic assumptions (plant parameters + commodity prices)

#### Section 5: Sensitivity & Cost Structure (4–5 slides)
- Slide: **NPV tornado chart** — one-at-a-time ±30% perturbation, **top 6 parameters**, each y-axis label must show the **nominal base value** in parentheses (e.g., "Cr₂O₃ price ($9,000/t)"). Do NOT perturb multiple parameters simultaneously. Use `StepRed` for adverse, `StepGreen!80` for favorable. Show base NPV dashed line. Invert colors for cost/rate parameters where increase = adverse.
- Slide: **CAPEX breakdown** — horizontal bar chart by process area (e.g., leach + precip, feed prep, LIMS, utilities, engineering). Show M USD and % share.
- Slide: **OPEX breakdown** — horizontal bar chart by cost category. Each label must include the **nominal specific unit cost** (e.g., "HCl @ $0.30/kg", "Electricity @ $0.08/kWh", "Labor — 8 FTE @ $35/hr", "Maintenance — 1.1% of CAPEX"). Show M USD/yr and % share.
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
- Use `text=black` inside light-filled boxes (15% opacity fills are too light for white text)
- Use `text=white` only for labels placed directly on the dark slide background
- **Node sizing**: Use `minimum width=1.6cm`, `minimum height=0.5cm`, `font=\tiny` to ensure PFD fits on slide
- **Node spacing**: Use `node distance=0.8cm and 1.1cm` for compact layout

### Big Number Slide Template

```latex
\begin{frame}{Levelized Net Value}
\centering
{\Large\textbf{\textcolor{StepGreen}{\$XXX/t}} levelized net value of production}\\[4pt]
% === Waterfall (bridge) chart ===
\begin{tikzpicture}[y=0.009cm]
  \pgfmathsetmacro{\bw}{0.42}
  % Revenue bars (cumulative, green shades)
  \fill[StepGreen!70] (1-\bw,0) rectangle (1+\bw,<rev1>);
  \node[above, font=\tiny, text=white] at (1,<rev1>) {+<rev1>};
  \node[below, font=\tiny, text=white] at (1,-15) {Product 1};
  % ... additional revenue bars stacked upward ...
  % Cost bars (deductions, red shades)
  \fill[StepRed!70] (5.2-\bw,<top-opex>) rectangle (5.2+\bw,<top>);
  \node[right, font=\tiny, text=white] at (5.2+\bw,<mid>) {-OPEX};
  \fill[StepRed!50] (6.4-\bw,<net>) rectangle (6.4+\bw,<top-opex>);
  \node[right, font=\tiny, text=white] at (6.4+\bw,<mid2>) {-EAC};
  % Net value bar
  \fill[StepElectric] (7.8-\bw,0) rectangle (7.8+\bw,<net>);
  \node[above, font=\scriptsize\bfseries, text=white] at (7.8,<net>) {<net>};
  % Y-axis, connectors, etc.
\end{tikzpicture}
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