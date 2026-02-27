"""
Report Generator
================
Produces a self-contained Markdown report after a flowsheet analysis,
including a rendered PNG process flow diagram, stream state tables, and
output-normalized economic and environmental metrics.
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import os

# Approximate molar masses (g/mol) for output mass estimation
_MOLAR_MASS = {
    "Nd+3": 144.24, "Ce+3": 140.12, "La+3": 138.91, "Pr+3": 140.91,
    "Y+3": 88.91, "Dy+3": 162.50, "Gd+3": 157.25, "Sm+3": 150.36,
    "Eu+3": 151.96, "Ho+3": 164.93, "Er+3": 167.26, "Yb+3": 173.05,
    "NdCl+2": 179.69, "CeCl+2": 175.57, "LaCl+2": 174.36,
    "NdCl2+": 215.14, "CeCl2+": 211.02, "LaCl2+": 209.81,
    "NdCl3(aq)": 250.60, "CeCl3(aq)": 246.47, "LaCl3(aq)": 245.26,
    "H2O(aq)": 18.015, "H2O": 18.015, "HCl(aq)": 36.46,
    "Na+": 22.99, "Cl-": 35.45, "Ca+2": 40.08, "Fe+3": 55.85,
    "C2O4-2": 88.02, "H+": 1.008, "OH-": 17.008,
    "NdCl4-": 286.05, "CeCl4-": 281.93, "LaCl4-": 280.71,
    "NdOH+2": 161.25, "CeOH+2": 157.13, "LaOH+2": 155.92,
    "PrCl+2": 176.36, "PrCl2+": 211.81, "PrCl3(aq)": 247.27,
    "PrCl4-": 282.72, "Pr+3": 140.91,
}

# Approximate market prices for REE oxides (USD/kg metal equivalent, 2024 proxy)
_REE_VALUE_USD_PER_KG = {
    "Nd": 150.0, "Pr": 100.0, "Dy": 350.0, "Tb": 1500.0,
    "Ce": 2.0, "La": 2.0, "Y": 10.0, "Sm": 3.0, "Eu": 30.0,
    "Gd": 40.0, "Ho": 70.0, "Er": 35.0, "Yb": 20.0, "Lu": 800.0,
    "Sc": 3500.0,
}

# REE element symbols for identifying product species
_REE_ELEMENTS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
                 "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Y", "Sc"}

# Species that are "waste" (non-valuable solvents, acids, water, common ions)
_WASTE_SPECIES = {"H2O(aq)", "H2O", "HCl(aq)", "H+", "OH-", "Cl-", "Na+",
                  "Ca+2", "SO4-2", "NO3-", "K+", "Mg+2"}


def _is_ree_species(name: str) -> bool:
    return any(el in name for el in _REE_ELEMENTS)


def _ree_element(name: str) -> Optional[str]:
    """Extract the REE element symbol from a species name."""
    for el in sorted(_REE_ELEMENTS, key=len, reverse=True):
        if el in name:
            return el
    return None


def _species_mass_kg(species_amounts: Dict[str, float]) -> float:
    total = 0.0
    for sp, mol in species_amounts.items():
        mm = _MOLAR_MASS.get(sp, 100.0)
        total += mol * mm / 1000.0
    return total


def _ree_mass_kg(species_amounts: Dict[str, float]) -> float:
    total = 0.0
    for sp, mol in species_amounts.items():
        if _is_ree_species(sp):
            mm = _MOLAR_MASS.get(sp, 144.0)
            total += mol * mm / 1000.0
    return total


def _waste_mass_kg(species_amounts: Dict[str, float]) -> float:
    total = 0.0
    for sp, mol in species_amounts.items():
        if sp in _WASTE_SPECIES or not _is_ree_species(sp):
            mm = _MOLAR_MASS.get(sp, 100.0)
            total += mol * mm / 1000.0
    return total


def _ree_value_usd(species_amounts: Dict[str, float]) -> float:
    """Estimate the market value of REE species in the stream."""
    total = 0.0
    for sp, mol in species_amounts.items():
        el = _ree_element(sp)
        if el and el in _REE_VALUE_USD_PER_KG:
            mm = _MOLAR_MASS.get(sp, 144.0)
            mass_kg = mol * mm / 1000.0
            total += mass_kg * _REE_VALUE_USD_PER_KG[el]
    return total


def _top_species(species_amounts: Dict[str, float], n: int = 5) -> List[tuple]:
    ranked = sorted(species_amounts.items(), key=lambda kv: kv[1], reverse=True)
    return [(sp, amt) for sp, amt in ranked[:n] if amt > 1e-6]


def _get_species_amounts(st) -> Dict[str, float]:
    if hasattr(st, "species_amounts"):
        return st.species_amounts
    elif isinstance(st, dict):
        return st.get("species_amounts", {})
    return {}


def _render_flowsheet_png(flowsheet, states, output_dir: str, timestamp: str) -> str:
    """Render a process flow diagram as a PNG using matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch

    produced = {s for u in flowsheet.units for s in u.outputs}
    consumed = {s for u in flowsheet.units for s in u.inputs}

    # Build topology: feeds -> (unit, intermediates, unit, ...) -> products
    # We lay out in a two-row staggered grid:
    #   Row 0 (y=0): feed(s), unit(s), product(s) — main spine
    #   Row 1 (y=-offset): secondary product outlets branch down

    feeds = [s.name for s in flowsheet.streams if s.name not in produced]
    unit_ids = [u.id for u in flowsheet.units]
    unit_map = {u.id: u for u in flowsheet.units}

    # Collect intermediates
    intermediates = {}
    for u in flowsheet.units:
        for out in u.outputs:
            if out in consumed:
                intermediates[out] = u.id

    # Collect products per unit
    products_by_unit = {}
    for u in flowsheet.units:
        for out in u.outputs:
            if out not in consumed:
                products_by_unit.setdefault(u.id, []).append(out)

    # Assign x positions along the main spine
    # Key rule: intermediates (consumed by downstream) stay on the main spine,
    # terminal products branch off vertically.
    x_step = 4.5
    node_pos = {}  # name -> (x, y)
    x = 0.0

    # Feeds
    for f in feeds:
        node_pos[f] = (x, 0.0)
        x += x_step

    # Units and their outputs
    for uid in unit_ids:
        node_pos[uid] = (x, 0.0)
        unit_obj = unit_map[uid]

        # Separate outputs into those consumed downstream vs terminal products
        spine_outs = [o for o in unit_obj.outputs if o in consumed]  # intermediates
        branch_outs = [o for o in unit_obj.outputs if o not in consumed]  # products

        # Place branches below the unit, staggered horizontally
        for j, p in enumerate(branch_outs):
            node_pos[p] = (x + x_step * (0.3 + 0.8 * j), -3.2 * (j + 1))

        x += x_step

    # Collect all edges
    edges = []
    for u in flowsheet.units:
        for inp in u.inputs:
            src = inp if inp in node_pos else intermediates.get(inp, inp)
            if src in node_pos:
                edges.append((src, u.id, inp))
        for out in u.outputs:
            if out in node_pos:
                edges.append((u.id, out, out))

    # Figure sizing
    all_x = [p[0] for p in node_pos.values()]
    all_y = [p[1] for p in node_pos.values()]
    fig_w = max(14, (max(all_x) - min(all_x)) + 8)
    fig_h = max(6, (max(all_y) - min(all_y)) + 6)

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
    ax.set_xlim(min(all_x) - 3, max(all_x) + 3)
    ax.set_ylim(min(all_y) - 3, max(all_y) + 3)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Process Flow Diagram — {flowsheet.name}\n{timestamp}",
                 fontsize=14, fontweight="bold", pad=15)

    bw, bh = 3.0, 1.6  # box width, height

    def _node_type(name):
        if name in [f for f in feeds]:
            return "feed"
        elif name in unit_ids:
            return "unit"
        else:
            return "product"

    _colors = {
        "feed":    ("#4a90d9", "#2c5f8a", "white"),
        "unit":    ("#f5a623", "#c17d12", "white"),
        "product": ("#7ed321", "#3d7a0a", "white"),
    }

    # Draw boxes
    for name, (cx, cy) in node_pos.items():
        ntype = _node_type(name)
        face, edge, txtc = _colors[ntype]
        rounding = "round,pad=0.2" if ntype != "unit" else "square,pad=0.15"
        box = mpatches.FancyBboxPatch(
            (cx - bw / 2, cy - bh / 2), bw, bh,
            boxstyle=rounding, facecolor=face, edgecolor=edge, lw=2
        )
        ax.add_patch(box)

        # Label inside box
        if ntype == "unit":
            label = f"{name}\n({unit_map[name].type})"
        elif ntype == "feed":
            label = f"{name}\n(Feed)"
        else:
            label = f"{name}\n(Product)"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=11, fontweight="bold", color=txtc)

        # Stream annotation below box
        if name in states:
            sp = _get_species_amounts(states[name])
            top = _top_species(sp, n=2)
            mass = _species_mass_kg(sp)
            if top:
                ann_parts = [f"{s}: {a:.1f} mol" for s, a in top]
                ann = "\n".join(ann_parts)
                ann += f"\n({mass:.1f} kg total)"
            else:
                ann = f"({mass:.1f} kg total)"
            ax.text(cx, cy - bh / 2 - 0.2, ann,
                    ha="center", va="top", fontsize=8, color="#444",
                    linespacing=1.3)

    # Draw arrows between boxes (connecting edge-to-edge, not center-to-center)
    for src, dst, stream_label in edges:
        sx, sy = node_pos[src]
        dx, dy = node_pos[dst]

        # Determine connection points on box edges
        if abs(dx - sx) > abs(dy - sy):
            # Horizontal dominant
            if dx > sx:
                x1, y1 = sx + bw / 2, sy
                x2, y2 = dx - bw / 2, dy
            else:
                x1, y1 = sx - bw / 2, sy
                x2, y2 = dx + bw / 2, dy
        else:
            # Vertical dominant
            if dy > sy:
                x1, y1 = sx, sy + bh / 2
                x2, y2 = dx, dy - bh / 2
            else:
                x1, y1 = sx, sy - bh / 2
                x2, y2 = dx, dy + bh / 2

        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-|>",
            mutation_scale=18,
            lw=2.0,
            color="#555",
            connectionstyle="arc3,rad=0.0" if abs(dy - sy) < 0.1 else "arc3,rad=0.2",
        )
        ax.add_patch(arrow)

        # Stream label on the arrow
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        offset_y = 0.3 if abs(dy - sy) < 0.1 else 0.0
        offset_x = 0.0 if abs(dy - sy) < 0.1 else 0.5
        ax.text(mx + offset_x, my + offset_y, stream_label,
                ha="center", va="bottom", fontsize=9,
                fontstyle="italic", color="#888",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="white",
                          edgecolor="#ddd", alpha=0.85))

    plt.tight_layout()
    img_name = f"flowsheet_diagram_{timestamp.replace(' ', '_').replace(':', '').replace('-', '')}.png"
    img_path = os.path.join(output_dir, img_name)
    fig.savefig(img_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return img_path


def generate_report(
    request_text: str,
    flowsheet,
    result: Dict[str, Any],
    states: Dict[str, Any],
    baseline_kpis: Dict[str, float],
    optimized_kpis: Optional[Dict[str, float]] = None,
    opt_params: Optional[Dict[str, float]] = None,
    opt_history: Optional[List[Dict]] = None,
    output_dir: str = "reports",
) -> tuple:
    """Generate a structured Markdown analysis report.

    Returns
    -------
    tuple of (str, str)
        (report_markdown, output_file_path)
    """
    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    ts_file = timestamp.strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"analysis_report_{ts_file}.md")

    # Identify terminal products and feeds
    produced = {s for u in flowsheet.units for s in u.outputs}
    consumed = {s for u in flowsheet.units for s in u.inputs}
    terminal_products = {o for u in flowsheet.units for o in u.outputs if o not in consumed}
    feed_names = [s.name for s in flowsheet.streams if s.name not in produced]

    # ── Compute mass balances ────────────────────────────────────────────
    # Feed mass
    feed_total_mass_kg = 0.0
    feed_ree_mass_kg = 0.0
    for f in feed_names:
        if f in states:
            sp = _get_species_amounts(states[f])
            feed_total_mass_kg += _species_mass_kg(sp)
            feed_ree_mass_kg += _ree_mass_kg(sp)

    # Product mass split: valuable (REE) vs waste
    product_valuable_kg = 0.0
    product_waste_kg = 0.0
    product_total_kg = 0.0
    product_value_usd = 0.0

    for name in terminal_products:
        if name in states:
            sp = _get_species_amounts(states[name])
            ree_kg = _ree_mass_kg(sp)
            waste_kg = _waste_mass_kg(sp)
            product_valuable_kg += ree_kg
            product_waste_kg += waste_kg
            product_total_kg += _species_mass_kg(sp)
            product_value_usd += _ree_value_usd(sp)

    opex = baseline_kpis.get("overall.opex_USD", 0)
    lca = baseline_kpis.get("overall.lca_kg_CO2e", 0)
    recovery = baseline_kpis.get("overall.recovery", 0)

    lines = []

    # ── Header ───────────────────────────────────────────────────────────
    lines.append("# REE Separation Process — Analysis Report")
    lines.append(f"\n**Generated**: {ts_str}")
    lines.append(f"**Flowsheet**: `{flowsheet.name}`")
    lines.append("")

    # ── Section 1: Request ───────────────────────────────────────────────
    lines.append("## 1. Analysis Request")
    lines.append("")
    lines.append(request_text.strip())
    lines.append("")

    # ── Section 2: System Description ────────────────────────────────────
    lines.append("## 2. System Description")
    lines.append("")
    lines.append(f"The flowsheet `{flowsheet.name}` consists of "
                 f"**{len(flowsheet.units)}** unit operation(s) and "
                 f"**{len(flowsheet.streams)}** defined feed stream(s).")
    lines.append("")

    for unit in flowsheet.units:
        params_str = ", ".join(f"`{k}={v}`" for k, v in (unit.params or {}).items()
                               if not isinstance(v, dict))
        lines.append(f"- **{unit.id}** (`{unit.type}`): {params_str}")
    lines.append("")

    # ── Section 3: Process Flow Diagram ──────────────────────────────────
    lines.append("## 3. Process Flowsheet")
    lines.append("")
    try:
        img_path = _render_flowsheet_png(flowsheet, states, output_dir, ts_str)
        rel_img = os.path.basename(img_path)
        lines.append(f"![Process Flow Diagram]({rel_img})")
    except Exception as e:
        lines.append(f"*Flowsheet rendering failed: {e}*")
    lines.append("")

    # ── Section 4: Stream State Table ────────────────────────────────────
    lines.append("## 4. Stream States")
    lines.append("")
    lines.append("| Stream | Type | T (K) | P (Pa) | Flow (mol) | Mass (kg) | pH | Top Species (mol) |")
    lines.append("|--------|------|------:|-------:|-----------:|----------:|---:|-------------------|")

    for name, st in states.items():
        if hasattr(st, "temperature_K"):
            temp = f"{st.temperature_K:.1f}"
            pres = f"{st.pressure_Pa:.0f}"
            flow = f"{st.flow_mol:.1f}"
            ph = f"{st.pH:.2f}" if st.pH is not None else "—"
        elif isinstance(st, dict):
            temp = f"{st.get('temperature_K', 298.15):.1f}"
            pres = f"{st.get('pressure_Pa', 101325):.0f}"
            flow = f"{st.get('flow_mol', 0):.1f}"
            ph_val = st.get('pH')
            ph = f"{ph_val:.2f}" if ph_val is not None else "—"
        else:
            continue

        sp_amounts = _get_species_amounts(st)
        mass = _species_mass_kg(sp_amounts)
        top = _top_species(sp_amounts, n=3)
        top_str = ", ".join(f"{sp} ({amt:.1f})" for sp, amt in top)

        # Classify stream type
        if name in feed_names:
            stype = "**Feed**"
        elif name in terminal_products:
            stype = "Product"
        else:
            stype = "Internal"

        lines.append(f"| {name} | {stype} | {temp} | {pres} | {flow} | {mass:.2f} | {ph} | {top_str} |")

    lines.append("")

    # ── Section 5: Output-Specific Performance ───────────────────────────
    lines.append("## 5. Output-Specific Performance")
    lines.append("")

    # Mass balance table
    lines.append("### Mass Balance")
    lines.append("")
    lines.append("| Category | Mass (kg) | Fraction |")
    lines.append("|----------|----------:|---------:|")
    lines.append(f"| Feed (total input) | {feed_total_mass_kg:.2f} | 100.0% |")
    lines.append(f"| Feed (REE content) | {feed_ree_mass_kg:.2f} | {feed_ree_mass_kg / feed_total_mass_kg * 100:.2f}% |" if feed_total_mass_kg > 0 else "| Feed (REE content) | 0.00 | — |")
    lines.append(f"| **Product (valuable REE)** | **{product_valuable_kg:.2f}** | — |")
    lines.append(f"| Product (waste/residual) | {product_waste_kg:.2f} | — |")
    lines.append(f"| Product (total) | {product_total_kg:.2f} | — |")
    lines.append("")

    # Output-normalized economics
    lines.append("### Economic & Environmental Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|------:|")
    lines.append(f"| Overall Recovery | {recovery * 100:.1f}% |")

    if product_valuable_kg > 1e-6:
        lines.append(f"| **OPEX / kg REE product** | **${opex / product_valuable_kg:.4f}/kg** |")
        lines.append(f"| **LCA / kg REE product** | **{lca / product_valuable_kg:.4f} kg CO₂e/kg** |")

    if product_total_kg > 1e-6:
        lines.append(f"| OPEX / kg total product | ${opex / product_total_kg:.4f}/kg |")
        lines.append(f"| LCA / kg total product | {lca / product_total_kg:.4f} kg CO₂e/kg |")

    if feed_total_mass_kg > 1e-6:
        lines.append(f"| **Estimated REE value / kg ore** | **${product_value_usd / feed_total_mass_kg:.4f}/kg ore** |")
        lines.append(f"| OPEX / kg ore (input) | ${opex / feed_total_mass_kg:.6f}/kg ore |")
        lines.append(f"| Net value / kg ore | ${(product_value_usd - opex) / feed_total_mass_kg:.4f}/kg ore |")

    lines.append(f"| REE product value (absolute) | ${product_value_usd:.2f} |")
    lines.append(f"| OPEX (absolute) | ${opex:.2f} |")
    lines.append(f"| LCA (absolute) | {lca:.2f} kg CO₂e |")
    lines.append("")

    # Per-unit recoveries
    unit_kpis = {k: v for k, v in baseline_kpis.items() if ".recovery" in k and "overall" not in k}
    if unit_kpis:
        lines.append("### Per-Unit Recovery")
        lines.append("")
        lines.append("| Unit | Recovery |")
        lines.append("|------|----------|")
        for k, v in unit_kpis.items():
            lines.append(f"| {k.replace('.recovery', '')} | {v * 100:.1f}% |")
        lines.append("")

    # ── Section 6: Optimization Results ──────────────────────────────────
    if optimized_kpis or opt_params:
        lines.append("## 6. Optimization Results (BoTorch)")
        lines.append("")

        if opt_params:
            lines.append("### Optimal Parameters")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|------:|")
            for k, v in opt_params.items():
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")

        if optimized_kpis:
            opt_opex = optimized_kpis.get("overall.opex_USD", opex)
            opt_lca = optimized_kpis.get("overall.lca_kg_CO2e", lca)
            pct_opex = ((opex - opt_opex) / opex * 100) if opex > 0 else 0

            lines.append("### Baseline vs. Optimized")
            lines.append("")
            lines.append("| Metric | Baseline | Optimized | Δ |")
            lines.append("|--------|----------|-----------|---|")
            lines.append(f"| OPEX | ${opex:.2f} | ${opt_opex:.2f} | {pct_opex:+.1f}% |")
            lines.append(f"| LCA | {lca:.2f} kg CO₂e | {opt_lca:.2f} kg CO₂e | — |")
            if product_valuable_kg > 1e-6:
                lines.append(f"| OPEX/kg REE | ${opex / product_valuable_kg:.4f} | ${opt_opex / product_valuable_kg:.4f} | — |")
            if feed_total_mass_kg > 1e-6:
                lines.append(f"| Net value/kg ore | ${(product_value_usd - opex) / feed_total_mass_kg:.4f} | ${(product_value_usd - opt_opex) / feed_total_mass_kg:.4f} | — |")
            lines.append("")

        if opt_history:
            lines.append("### Convergence History")
            lines.append("")
            lines.append("| Iteration | Best OPEX ($) |")
            lines.append("|----------:|--------------:|")
            for h in opt_history:
                lines.append(f"| {h['iter']} | {h['best_y']:.4f} |")
            lines.append("")

    # ── Write ────────────────────────────────────────────────────────────
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    return report, output_path

# ═══════════════════════════════════════════════════════════════════════
# GDP Superstructure Report
# ═══════════════════════════════════════════════════════════════════════

def generate_gdp_report(
    gdp_result,
    superstructure_name: str = "",
    output_dir: str = "reports",
    database: str = "light_ree",
) -> tuple:
    """Generate a comprehensive Markdown report for GDP optimization.

    Includes the topology ranking **plus** full analysis-report detail
    (flowsheet PNG, stream states, mass balance, $/kg metrics) for the
    best configuration.

    Parameters
    ----------
    gdp_result : GDPResult
        Output from :func:`optimize_superstructure`.
    superstructure_name : str
        Name of the superstructure that was optimized.
    output_dir : str
        Directory to write the report file.
    database : str
        Reaktoro database preset (for re-evaluation of best config).

    Returns
    -------
    tuple of (str, str)
        (report_markdown, output_file_path)
    """
    from .opt.gdp_builder import build_sub_flowsheet
    from .dsl.ree_superstructures import SUPERSTRUCTURE_REGISTRY
    from .sim.idaes_adapter import IDAESFlowsheetBuilder

    ts = datetime.now()
    ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")
    ts_file = ts.strftime("%Y%m%d_%H%M%S")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gdp_report_{ts_file}.md")

    lines = [
        "# REE Separation — GDP Superstructure Optimization Report",
        "",
        f"**Generated**: {ts_str}  ",
        f"**Superstructure**: {superstructure_name}  ",
        f"**Objective**: {gdp_result.objective}  ",
        f"**Configurations Evaluated**: {gdp_result.num_configs_evaluated}  ",
        f"**Total Runtime**: {gdp_result.total_elapsed_s:.1f}s",
        "",
        "---",
        "",
    ]

    # ── Section 1: Topology Ranking ──────────────────────────────────
    lines.append("## 1. Topology Ranking")
    lines.append("")
    lines.append("| Rank | Active Units | OPEX ($) | LCA (kgCO₂e) | Recovery | Status |")
    lines.append("|:----:|-------------|--------:|-------------:|---------:|:------:|")

    successful = sorted(
        [r for r in gdp_result.all_results if r.status == "ok"],
        key=lambda r: r.objective_value,
    )
    failed = [r for r in gdp_result.all_results if r.status != "ok"]

    for rank, ev in enumerate(successful, 1):
        active = ", ".join(sorted(ev.config.active_unit_ids))
        opex = ev.kpis.get("overall.opex_USD", 0)
        lca = ev.kpis.get("overall.lca_kg_CO2e", 0)
        rec = ev.kpis.get("overall.recovery", 0)
        badge = "🏆" if rank == 1 else "✅"
        lines.append(
            f"| {rank} {badge} | {active} | {opex:.2f} | {lca:.2f} | "
            f"{rec:.2%} | OK |"
        )

    for ev in failed:
        active = ", ".join(sorted(ev.config.active_unit_ids))
        lines.append(f"| — | {active} | — | — | — | ❌ {ev.error[:40]} |")

    lines.append("")

    # ── Re-evaluate best for full stream states ──────────────────────
    best = gdp_result.best
    flowsheet = None
    states = {}
    best_kpis = {}

    if best and superstructure_name in SUPERSTRUCTURE_REGISTRY:
        try:
            ss = SUPERSTRUCTURE_REGISTRY[superstructure_name]()
            flowsheet = build_sub_flowsheet(ss, best.config)

            # Apply optimized params if any
            if best.optimized_params:
                unit_map = {u.id: u for u in flowsheet.units}
                for key, val in best.optimized_params.items():
                    parts = key.split(".", 1)
                    if len(parts) == 2 and parts[0] in unit_map:
                        unit_map[parts[0]].params[parts[1]] = val

            builder = IDAESFlowsheetBuilder(database_name=database)
            sim = builder.build_and_solve(flowsheet)
            if sim.get("status") == "ok":
                states = sim.get("states", {})
                best_kpis = sim.get("kpis", {})
        except Exception as e:
            _log.warning("Re-evaluation for report failed: %s", e)

    if not best_kpis and best:
        best_kpis = best.kpis

    # ── Section 2: Best Configuration Summary ────────────────────────
    if best:
        lines.append("---")
        lines.append("")
        lines.append("## 2. Best Configuration — 🏆")
        lines.append("")
        lines.append(f"**Active units**: {', '.join(sorted(best.config.active_unit_ids))}  ")
        if best.config.bypassed_unit_ids:
            lines.append(f"**Bypassed units**: {', '.join(sorted(best.config.bypassed_unit_ids))}  ")
        if best.config.stage_choices:
            stages = ", ".join(f"{k}={v}" for k, v in best.config.stage_choices.items())
            lines.append(f"**Stage choices**: {stages}  ")
        lines.append("")

        if flowsheet:
            for unit in flowsheet.units:
                params_str = ", ".join(
                    f"`{k}={v}`" for k, v in (unit.params or {}).items()
                    if not isinstance(v, dict)
                )
                lines.append(f"- **{unit.id}** (`{unit.type}`): {params_str}")
            lines.append("")

        if best.optimized_params:
            lines.append("### Optimized Continuous Parameters")
            lines.append("")
            lines.append("| Parameter | Value |")
            lines.append("|-----------|------:|")
            for k, v in best.optimized_params.items():
                lines.append(f"| `{k}` | {v:.4f} |")
            lines.append("")

    # ── Section 3: Process Flowsheet (PNG) ───────────────────────────
    if flowsheet and states:
        lines.append("## 3. Process Flowsheet")
        lines.append("")
        try:
            img_path = _render_flowsheet_png(flowsheet, states, output_dir, ts_str)
            rel_img = os.path.basename(img_path)
            lines.append(f"![Process Flow Diagram]({rel_img})")
        except Exception as e:
            lines.append(f"*Flowsheet rendering failed: {e}*")
        lines.append("")

    # ── Section 4: Stream State Table ────────────────────────────────
    if states and flowsheet:
        produced = {s for u in flowsheet.units for s in u.outputs}
        consumed = {s for u in flowsheet.units for s in u.inputs}
        terminal_products = {o for u in flowsheet.units for o in u.outputs if o not in consumed}
        feed_names = [s.name for s in flowsheet.streams if s.name not in produced]

        lines.append("## 4. Stream States")
        lines.append("")
        lines.append("| Stream | Type | T (K) | P (Pa) | Flow (mol) | Mass (kg) | pH | Top Species (mol) |")
        lines.append("|--------|------|------:|-------:|-----------:|----------:|---:|-------------------|")

        for name, st in states.items():
            if hasattr(st, "temperature_K"):
                temp = f"{st.temperature_K:.1f}"
                pres = f"{st.pressure_Pa:.0f}"
                flow = f"{st.flow_mol:.1f}"
                ph = f"{st.pH:.2f}" if st.pH is not None else "—"
            elif isinstance(st, dict):
                temp = f"{st.get('temperature_K', 298.15):.1f}"
                pres = f"{st.get('pressure_Pa', 101325):.0f}"
                flow = f"{st.get('flow_mol', 0):.1f}"
                ph_val = st.get('pH')
                ph = f"{ph_val:.2f}" if ph_val is not None else "—"
            else:
                continue

            sp_amounts = _get_species_amounts(st)
            mass = _species_mass_kg(sp_amounts)
            top = _top_species(sp_amounts, n=3)
            top_str = ", ".join(f"{sp} ({amt:.1f})" for sp, amt in top)

            if name in feed_names:
                stype = "**Feed**"
            elif name in terminal_products:
                stype = "Product"
            else:
                stype = "Internal"

            lines.append(f"| {name} | {stype} | {temp} | {pres} | {flow} | {mass:.2f} | {ph} | {top_str} |")

        lines.append("")

        # ── Section 5: Mass Balance & Metrics ────────────────────────
        feed_total_mass_kg = 0.0
        feed_ree_mass_kg = 0.0
        for f in feed_names:
            if f in states:
                sp = _get_species_amounts(states[f])
                feed_total_mass_kg += _species_mass_kg(sp)
                feed_ree_mass_kg += _ree_mass_kg(sp)

        product_valuable_kg = 0.0
        product_waste_kg = 0.0
        product_total_kg = 0.0
        product_value_usd = 0.0

        for name in terminal_products:
            if name in states:
                sp = _get_species_amounts(states[name])
                product_valuable_kg += _ree_mass_kg(sp)
                product_waste_kg += _waste_mass_kg(sp)
                product_total_kg += _species_mass_kg(sp)
                product_value_usd += _ree_value_usd(sp)

        opex = best_kpis.get("overall.opex_USD", 0)
        lca = best_kpis.get("overall.lca_kg_CO2e", 0)
        recovery = best_kpis.get("overall.recovery", 0)

        lines.append("## 5. Output-Specific Performance")
        lines.append("")

        # Mass balance table
        lines.append("### Mass Balance")
        lines.append("")
        lines.append("| Category | Mass (kg) | Fraction |")
        lines.append("|----------|----------:|---------:|")
        lines.append(f"| Feed (total input) | {feed_total_mass_kg:.2f} | 100.0% |")
        if feed_total_mass_kg > 0:
            lines.append(f"| Feed (REE content) | {feed_ree_mass_kg:.2f} | {feed_ree_mass_kg / feed_total_mass_kg * 100:.2f}% |")
        else:
            lines.append("| Feed (REE content) | 0.00 | — |")
        lines.append(f"| **Product (valuable REE)** | **{product_valuable_kg:.2f}** | — |")
        lines.append(f"| Product (waste/residual) | {product_waste_kg:.2f} | — |")
        lines.append(f"| Product (total) | {product_total_kg:.2f} | — |")
        lines.append("")

        # Output-normalized economics
        lines.append("### Economic & Environmental Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|------:|")
        lines.append(f"| Overall Recovery | {recovery * 100:.1f}% |")

        if product_valuable_kg > 1e-6:
            lines.append(f"| **OPEX / kg REE product** | **${opex / product_valuable_kg:.4f}/kg** |")
            lines.append(f"| **LCA / kg REE product** | **{lca / product_valuable_kg:.4f} kg CO₂e/kg** |")
        if product_total_kg > 1e-6:
            lines.append(f"| OPEX / kg total product | ${opex / product_total_kg:.4f}/kg |")
            lines.append(f"| LCA / kg total product | {lca / product_total_kg:.4f} kg CO₂e/kg |")
        if feed_total_mass_kg > 1e-6:
            lines.append(f"| **Estimated REE value / kg ore** | **${product_value_usd / feed_total_mass_kg:.4f}/kg ore** |")
            lines.append(f"| OPEX / kg ore (input) | ${opex / feed_total_mass_kg:.6f}/kg ore |")
            lines.append(f"| Net value / kg ore | ${(product_value_usd - opex) / feed_total_mass_kg:.4f}/kg ore |")
        lines.append(f"| REE product value (absolute) | ${product_value_usd:.2f} |")
        lines.append(f"| OPEX (absolute) | ${opex:.2f} |")
        lines.append(f"| LCA (absolute) | {lca:.2f} kg CO₂e |")
        lines.append("")

        # Per-unit recoveries
        unit_kpis = {k: v for k, v in best_kpis.items()
                     if ".recovery" in k and "overall" not in k}
        if unit_kpis:
            lines.append("### Per-Unit Recovery")
            lines.append("")
            lines.append("| Unit | Recovery |")
            lines.append("|------|----------|")
            for k, v in unit_kpis.items():
                lines.append(f"| {k.replace('.recovery', '')} | {v * 100:.1f}% |")
            lines.append("")

    # ── Section 6: All Configurations (Detail) ───────────────────────
    if len(successful) > 1:
        lines.append("---")
        lines.append("")
        lines.append("## 6. All Configurations (Comparison)")
        lines.append("")
        for rank, ev in enumerate(successful, 1):
            active = ", ".join(sorted(ev.config.active_unit_ids))
            bypassed = ", ".join(sorted(ev.config.bypassed_unit_ids)) or "none"
            badge = " 🏆" if rank == 1 else ""
            lines.append(f"### Config #{rank}{badge}: [{active}]")
            lines.append("")
            lines.append(f"- **Bypassed**: {bypassed}")
            lines.append(f"- **Objective**: {ev.objective_value:.4f}")
            lines.append(f"- **Runtime**: {ev.elapsed_s:.2f}s")

            if ev.kpis:
                lines.append("")
                lines.append("| KPI | Value |")
                lines.append("|-----|------:|")
                for k, v in sorted(ev.kpis.items()):
                    val = f"{v:.4f}" if isinstance(v, float) else str(v)
                    lines.append(f"| `{k}` | {val} |")
            lines.append("")

    # ── Write ────────────────────────────────────────────────────────
    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)

    return report, output_path
