---
description: End-to-end valorization analysis — chains all sub-workflows from raw material to final report
---

# Valorize — Full Pipeline Orchestrator

// turbo-all

This is the **top-level orchestrator** for valorization analysis. It chains all sub-workflows in sequence to take a raw material from initial characterization through GDP optimization to a polished LaTeX report and Beamer presentation.

**Usage**: The user invokes `/valorize` and provides a raw material description. The agent runs the full pipeline, pausing at key decision points for user review.

---

## Python Environment

> **All Python commands in this pipeline MUST be executed inside the `rkt` conda environment.**

Use one of these patterns for every Python invocation:

```bash
# For scripts:
conda run --no-capture-output -n rkt python3 <script.py>

# For inline Python:
conda run --no-capture-output -n rkt python3 -c "<code>"

# For pip installs (if needed):
conda run -n rkt pip install <package>
```

The `rkt` environment contains: `reaktoro`, `jax`, `jaxlib`, `pyomo`, `idaes-pse`, `botorch`, and the `sep_agents` package.

**Never use the system Python or base conda environment for simulation or optimization.**

---

## Inputs Required from User

Before starting, ensure you have:
- **Raw material description**: source, mineralogy, metal assay, throughput (t/d or t/yr)
- **Objective** (optional): defaults to `maximize_value_per_kg_ore`; alternatives: `minimize_lca`, `maximize_recovery`
- **Commodity prices** (optional): override defaults if the user has specific market data

If the user provides incomplete information, request the minimum required fields before proceeding.

---

## Stage 1: Resource Characterization

Follow all steps in `/resource-characterization`:

1. Parse the user's raw material description into composition data
2. Run equilibrium speciation via `speciate_ree_stream` / `run_speciation`
3. Identify all viable value streams (metals, REE, H₂, CO₂, iron, acid)
4. Assign commodity prices
5. Generate `feed_characterization.yaml` and a Markdown summary

### 🔄 Gate: User Review

**Pause and present the characterization results to the user.** Ask:
- "Does this composition and speciation look correct?"
- "Are there additional value streams to consider?"
- "Any commodity price adjustments?"

**Wait for user confirmation before proceeding.**

---

## Stage 2: Superstructure Selection

Follow all steps in `/superstructure-selection`:

1. Query the superstructure registry for pre-built candidates
2. Evaluate applicability of each to the identified value streams
3. If needed, construct new superstructure(s) using the DSL
4. Enumerate valid topologies via `enumerate_configurations()`
5. Produce a topology summary table

### 🔄 Gate: User Review

**Pause and present the superstructure candidates to the user.** Ask:
- "Do these superstructure candidates cover the value streams adequately?"
- "Any process alternatives to add or remove?"
- "Proceed with optimization of [N] candidates?"

**Wait for user confirmation before proceeding.**

---

## Stage 3: Process Optimization

Follow all steps in `/process-optimization`:

1. Run GDP topology optimization (enumerative or EO solver)
2. Simulate top topologies via IDAES
3. Compute TEA (JAX itemized costs, EAC)
4. Estimate revenue from all product streams
5. Run BoTorch continuous optimization (if applicable)
6. Compute cost sensitivities via JAX autodiff
7. Assemble trade-study comparison table
8. Generate Markdown reports with PFDs

**Auto-continue** — no user gate (results are presented in the final report).

---

## Stage 4: Technical Report

Follow all steps in `/valorization-report`:

1. Gather all results from Stages 1–3
2. Create LaTeX technical report using the Eagle Mine template structure
3. Include all tables, TikZ PFD, equations, and discussion sections
4. Compile with `pdflatex` (run twice for cross-references)
5. Verify clean PDF output

**Auto-continue** to presentation generation.

---

## Stage 5: Presentation

Follow all steps in `/valorization-presentation`:

1. Distill the technical report into 15–25 Beamer slides
2. Create title, background, PFD, results, economics, sensitivity, conclusions slides
3. Use the `metropolis` Beamer theme
4. Compile with `pdflatex`
5. Verify clean PDF output

---

## Stage 6: Final Delivery

Present all outputs to the user:

1. **Feed characterization** — `feed_characterization.yaml`
2. **Trade-study table** — ranked topology comparison (Markdown)
3. **Optimal process** — description of the best topology with key KPIs
4. **Technical report** — `docs/<resource>_valorization_report.pdf`
5. **Presentation** — `docs/<resource>_valorization_presentation.pdf`
6. **Process flow diagrams** — PNG/SVG files in `reports/`

Summarize the key finding in a single sentence:
> **"Processing [resource] at [throughput] t/d yields a net value of $[X]/t via [primary value streams], with a total project NPV of $[Y]M over [Z] years."**

---

## Error Recovery

If any stage fails:
- **Speciation failure**: Check species names against the Reaktoro database; try alternative presets
- **GDP solver infeasible**: Relax constraints, check stream connectivity, verify unit parameters
- **BoTorch no improvement**: Report base-case results; the base topology may already be near-optimal
- **LaTeX compilation error**: Fix the specific error, recompile; common issues are special characters in labels or missing packages
- **Missing data**: Return to the user with specific questions rather than using unsupported defaults
