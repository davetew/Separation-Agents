"""
Superstructure Flowsheet Visualizer
=====================================

Generates publication-quality process flow diagrams (PFDs) from
:class:`Superstructure` or YAML definitions.  Each unit operation is
drawn with a shape that reflects its type, streams are shown as
directed edges, and GDP disjunctions are highlighted with dashed
enclosures.

Heat exchangers are drawn with the standard shell-and-tube symbol
(circle with two crossing streamlines) and paired preheat/recovery
HXs are connected with dashed heat-coupling arcs.

Usage
-----
>>> from sep_agents.vis.flowsheet_vis import visualize_superstructure
>>> visualize_superstructure("eaf_steel_slag", save="eaf_pfd.png")

Or from CLI:
    python -m sep_agents.vis.flowsheet_vis eaf_steel_slag -o eaf_pfd.png
"""
from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Arc
import matplotlib.patheffects as pe
import networkx as nx
import numpy as np

from ..dsl.schemas import DisjunctionDef, Flowsheet, Superstructure, UnitOp

_log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# Unit type → visual style mapping
# ═══════════════════════════════════════════════════════════════════════

UNIT_STYLES: Dict[str, Dict[str, Any]] = {
    "pump":                  {"shape": "trapezoid",  "color": "#4A90D9", "icon": "P"},
    "heat_exchanger":        {"shape": "hx",         "color": "#E8913A", "icon": "HX"},
    "mixer":                 {"shape": "diamond",    "color": "#7B7B7B", "icon": "M"},
    "separator":             {"shape": "diamond",    "color": "#9B59B6", "icon": "S"},
    "lims":                  {"shape": "hexagon",    "color": "#2ECC71", "icon": "⊕"},
    "equilibrium_reactor":   {"shape": "rectangle",  "color": "#E74C3C", "icon": "≡R"},
    "stoichiometric_reactor":{"shape": "rectangle",  "color": "#C0392B", "icon": "SR"},
    "leach_reactor":         {"shape": "rectangle",  "color": "#D35400", "icon": "LR"},
    "precipitator":          {"shape": "rectangle",  "color": "#8E44AD", "icon": "PP"},
    "mill":                  {"shape": "circle",     "color": "#34495E", "icon": "⚙"},
    "cyclone":               {"shape": "circle",     "color": "#1ABC9C", "icon": "CY"},
    "thickener":             {"shape": "trapezoid",  "color": "#16A085", "icon": "TH"},
    "flotation_bank":        {"shape": "rectangle",  "color": "#2980B9", "icon": "FL"},
    "solvent_extraction":    {"shape": "rectangle",  "color": "#F39C12", "icon": "SX"},
    "ion_exchange":          {"shape": "rectangle",  "color": "#27AE60", "icon": "IX"},
    "crystallizer":          {"shape": "hexagon",    "color": "#8E44AD", "icon": "CR"},
}

DEFAULT_STYLE = {"shape": "rectangle", "color": "#95A5A6", "icon": "?"}

# Disjunction group colors (cycled)
DISJUNCTION_COLORS = [
    "#E74C3C40", "#3498DB40", "#2ECC7140",
    "#F39C1240", "#9B59B640", "#1ABC9C40",
]
DISJUNCTION_BORDER_COLORS = [
    "#E74C3C", "#3498DB", "#2ECC71",
    "#F39C12", "#9B59B6", "#1ABC9C",
]

# Heat coupling arc style
HEAT_COUPLING_COLOR = "#E8913A"


# ═══════════════════════════════════════════════════════════════════════
# Heat Exchanger Pairing
# ═══════════════════════════════════════════════════════════════════════

def _detect_hx_pairs(
    superstructure: Superstructure,
) -> List[Tuple[str, str, str]]:
    """Auto-detect paired heat exchangers.

    Matches by naming convention:
        hx_{group}_preheat  ↔  hx_{group}_recovery
    Also honours an explicit ``heat_pair`` param in unit params.

    Returns list of (cold_side_id, hot_side_id, group_label).
    """
    hx_units = {
        u.id: u for u in superstructure.base_flowsheet.units
        if u.type == "heat_exchanger"
    }

    pairs: List[Tuple[str, str, str]] = []
    paired_ids: Set[str] = set()

    # 1. Explicit heat_pair params
    for uid, u in hx_units.items():
        partner = u.params.get("heat_pair")
        if partner and partner in hx_units and uid not in paired_ids:
            pairs.append((uid, partner, _hx_group_label(uid)))
            paired_ids.add(uid)
            paired_ids.add(partner)

    # 2. Naming convention:  hx_{group}_preheat ↔ hx_{group}_recovery
    preheat_re = re.compile(r"^hx_(.+?)_preheat$")
    for uid in hx_units:
        if uid in paired_ids:
            continue
        m = preheat_re.match(uid)
        if m:
            group = m.group(1)
            recovery_id = f"hx_{group}_recovery"
            if recovery_id in hx_units and recovery_id not in paired_ids:
                # preheat = cold side (heating feed), recovery = hot side (cooling product)
                pairs.append((uid, recovery_id, group.replace("_", " ")))
                paired_ids.add(uid)
                paired_ids.add(recovery_id)

    return pairs


def _hx_group_label(uid: str) -> str:
    """Extract a human-readable group label from an HX unit ID."""
    label = uid.replace("hx_", "").replace("_preheat", "").replace("_recovery", "")
    return label.replace("_", " ")


# ═══════════════════════════════════════════════════════════════════════
# Shape drawing helpers
# ═══════════════════════════════════════════════════════════════════════

def _draw_hx_symbol(
    ax: plt.Axes,
    x: float, y: float,
    color: str,
    is_optional: bool = False,
    r: float = 0.38,
    is_paired: bool = False,
    is_utility: bool = False,
    cold_stream: str = "",
    hot_stream: str = "",
) -> None:
    """Draw a proper shell-and-tube heat exchanger symbol.

    Two streamlines cross through a circle, representing the two
    fluid streams exchanging energy.
    """
    linestyle = "--" if is_optional else "-"
    edgecolor = "#333333"

    # Main circle (shell)
    circle = plt.Circle(
        (x, y), r,
        facecolor=color, edgecolor=edgecolor,
        linewidth=1.5, linestyle=linestyle,
        alpha=0.85, zorder=3,
    )
    ax.add_patch(circle)

    # Two crossing streamlines through the HX
    # Horizontal line (process stream — the one in the graph edges)
    lw = 2.0
    ax.plot(
        [x - r * 0.95, x + r * 0.95], [y, y],
        color="white", linewidth=lw, solid_capstyle="round",
        zorder=3.5,
    )
    # Diagonal line (counter-stream / utility)
    ax.plot(
        [x - r * 0.7, x + r * 0.7], [y + r * 0.7, y - r * 0.7],
        color="white", linewidth=lw, solid_capstyle="round",
        zorder=3.5,
    )

    # Small arrowheads on the diagonal to show counter-flow direction
    arr_size = r * 0.25
    # Arrow at the bottom-right of diagonal (hot stream entering from top)
    ax.annotate(
        "", xy=(x + r * 0.55, y - r * 0.55),
        xytext=(x + r * 0.15, y - r * 0.15),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=1.5),
        zorder=3.6,
    )

    # "HX" icon
    ax.text(
        x, y, "HX",
        ha="center", va="center",
        fontsize=8, fontweight="bold", color="white",
        zorder=4,
        path_effects=[pe.withStroke(linewidth=2, foreground="#333333")],
    )

    # If utility (unpaired standalone HX like aux_heater), add utility label
    if is_utility and not is_paired:
        ax.text(
            x, y + r + 0.12, "utility",
            ha="center", va="bottom",
            fontsize=5, color="#E8913A",
            fontstyle="italic", fontweight="bold",
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.08", facecolor="#FFF3E0",
                      edgecolor="#E8913A", alpha=0.8, linewidth=0.5),
        )


def _draw_heat_coupling(
    ax: plt.Axes,
    pos_cold: Tuple[float, float],
    pos_hot: Tuple[float, float],
    group_label: str,
    cold_input: str = "",
    cold_output: str = "",
    hot_input: str = "",
    hot_output: str = "",
) -> None:
    """Draw a dashed heat-coupling arc between paired HXs.

    Shows the energy flow from hot-side recovery HX to cold-side
    preheat HX with a curved arrow and 'Q' label.
    """
    x1, y1 = pos_cold
    x2, y2 = pos_hot

    # Determine arc direction (route above or below the main flow)
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    # Offset the arc above or below based on relative positions
    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)

    # Arc height proportional to distance
    arc_height = max(0.8, dist * 0.3)

    # Draw curved dashed arrow from hot to cold (energy flows hot → cold)
    ax.annotate(
        "",
        xy=(x1, y1 + 0.4),  # cold side (receives Q)
        xytext=(x2, y2 + 0.4),  # hot side (releases Q)
        arrowprops=dict(
            arrowstyle="-|>",
            color=HEAT_COUPLING_COLOR,
            lw=2.0,
            linestyle="--",
            shrinkA=8, shrinkB=8,
            connectionstyle=f"arc3,rad=-0.3",
        ),
        zorder=2.5,
    )

    # "Q" label at the midpoint of the arc
    # Offset the label above the arc
    label_x = mid_x
    label_y = mid_y + arc_height * 0.5 + 0.15

    ax.text(
        label_x, label_y,
        f"Q ({group_label})",
        ha="center", va="bottom",
        fontsize=6, fontweight="bold",
        color=HEAT_COUPLING_COLOR,
        zorder=5,
        bbox=dict(
            boxstyle="round,pad=0.12",
            facecolor="#FFF3E0",
            edgecolor=HEAT_COUPLING_COLOR,
            alpha=0.85,
            linewidth=1.0,
        ),
    )

    # Stream annotations near each HX showing which streams exchange heat
    anno_fontsize = 5
    anno_color = "#B06000"

    if hot_input:
        ax.text(
            x2, y2 - 0.55,
            f"hot: {hot_input.replace('_', ' ')}",
            ha="center", va="top",
            fontsize=anno_fontsize, color=anno_color,
            fontstyle="italic", zorder=5,
        )
    if cold_input:
        ax.text(
            x1, y1 - 0.55,
            f"cold: {cold_input.replace('_', ' ')}",
            ha="center", va="top",
            fontsize=anno_fontsize, color=anno_color,
            fontstyle="italic", zorder=5,
        )


def _draw_unit_node(
    ax: plt.Axes,
    x: float,
    y: float,
    unit_id: str,
    unit_type: str,
    style: Dict[str, Any],
    is_optional: bool = False,
    w: float = 1.4,
    h: float = 0.7,
    is_paired_hx: bool = False,
    is_utility_hx: bool = False,
) -> None:
    """Draw a single unit operation node at (x, y)."""
    shape = style["shape"]
    color = style["color"]
    icon = style["icon"]

    linestyle = "--" if is_optional else "-"
    linewidth = 1.5
    edgecolor = "#333333"

    if shape == "hx":
        # Use the proper two-stream HX symbol
        _draw_hx_symbol(
            ax, x, y, color,
            is_optional=is_optional,
            is_paired=is_paired_hx,
            is_utility=is_utility_hx,
        )
        # Label below
        label = unit_id.replace("_", "\n")
        ax.text(
            x, y - 0.38 - 0.18, label,
            ha="center", va="top",
            fontsize=6, color="#333333",
            zorder=4,
        )
        return

    if shape == "rectangle":
        rect = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.05",
            facecolor=color, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
            alpha=0.85, zorder=3,
        )
        ax.add_patch(rect)

    elif shape == "circle":
        circle = plt.Circle(
            (x, y), h / 2,
            facecolor=color, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
            alpha=0.85, zorder=3,
        )
        ax.add_patch(circle)

    elif shape == "diamond":
        s = h * 0.6
        diamond = plt.Polygon(
            [(x, y + s), (x + s, y), (x, y - s), (x - s, y)],
            facecolor=color, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
            alpha=0.85, zorder=3,
        )
        ax.add_patch(diamond)

    elif shape == "hexagon":
        r = h / 2
        angles = np.linspace(0, 2 * np.pi, 7)[:-1]
        hex_pts = [(x + r * 1.2 * np.cos(a), y + r * np.sin(a)) for a in angles]
        hexagon = plt.Polygon(
            hex_pts,
            facecolor=color, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
            alpha=0.85, zorder=3,
        )
        ax.add_patch(hexagon)

    elif shape == "trapezoid":
        trap = plt.Polygon(
            [
                (x - w * 0.3, y + h / 2),
                (x + w * 0.3, y + h / 2),
                (x + w / 2, y - h / 2),
                (x - w / 2, y - h / 2),
            ],
            facecolor=color, edgecolor=edgecolor,
            linewidth=linewidth, linestyle=linestyle,
            alpha=0.85, zorder=3,
        )
        ax.add_patch(trap)

    # Icon text
    ax.text(
        x, y, icon,
        ha="center", va="center",
        fontsize=9, fontweight="bold", color="white",
        zorder=4,
        path_effects=[pe.withStroke(linewidth=2, foreground="#333333")],
    )

    # Label below
    label = unit_id.replace("_", "\n")
    ax.text(
        x, y - h / 2 - 0.18, label,
        ha="center", va="top",
        fontsize=6, color="#333333",
        zorder=4,
    )


def _draw_disjunction_box(
    ax: plt.Axes,
    positions: Dict[str, Tuple[float, float]],
    unit_ids: List[str],
    name: str,
    description: str,
    color_fill: str,
    color_border: str,
    w: float = 1.4,
    h: float = 0.7,
) -> None:
    """Draw a dashed enclosure around disjunction units."""
    xs = [positions[uid][0] for uid in unit_ids if uid in positions]
    ys = [positions[uid][1] for uid in unit_ids if uid in positions]

    if not xs:
        return

    pad = 0.5
    x_min, x_max = min(xs) - w / 2 - pad, max(xs) + w / 2 + pad
    y_min, y_max = min(ys) - h / 2 - pad * 0.8, max(ys) + h / 2 + pad * 0.8

    rect = FancyBboxPatch(
        (x_min, y_min), x_max - x_min, y_max - y_min,
        boxstyle="round,pad=0.15",
        facecolor=color_fill,
        edgecolor=color_border,
        linewidth=2.0, linestyle="--",
        zorder=1,
    )
    ax.add_patch(rect)

    ax.text(
        (x_min + x_max) / 2, y_max + 0.05,
        f"⊻ {name}",
        ha="center", va="bottom",
        fontsize=7, fontweight="bold",
        color=color_border,
        zorder=5,
    )


# ═══════════════════════════════════════════════════════════════════════
# Graph layout
# ═══════════════════════════════════════════════════════════════════════

def _build_graph(superstructure: Superstructure) -> nx.DiGraph:
    """Build a networkx DiGraph from the superstructure."""
    G = nx.DiGraph()
    units = superstructure.base_flowsheet.units

    producers: Dict[str, str] = {}
    consumers: Dict[str, List[str]] = {}

    for u in units:
        G.add_node(u.id, type=u.type, optional=u.optional)
        for out_s in u.outputs:
            producers[out_s] = u.id
        for in_s in u.inputs:
            consumers.setdefault(in_s, []).append(u.id)

    all_inputs = {s for u in units for s in u.inputs}
    all_outputs = {s for u in units for s in u.outputs}
    feed_streams = all_inputs - all_outputs
    product_streams = all_outputs - all_inputs

    for fs in feed_streams:
        G.add_node(f"feed:{fs}", type="__feed__", optional=False)
        for uid in consumers.get(fs, []):
            G.add_edge(f"feed:{fs}", uid, stream=fs)

    for ps in product_streams:
        G.add_node(f"prod:{ps}", type="__product__", optional=False)
        if ps in producers:
            G.add_edge(producers[ps], f"prod:{ps}", stream=ps)

    for stream_name in producers:
        if stream_name in consumers:
            src = producers[stream_name]
            for dst in consumers[stream_name]:
                G.add_edge(src, dst, stream=stream_name)

    return G


def _hierarchical_layout(
    G: nx.DiGraph,
    superstructure: Superstructure,
) -> Dict[str, Tuple[float, float]]:
    """Compute a left-to-right hierarchical layout."""
    try:
        topo_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(G, k=2.5, iterations=50)

    layers: Dict[str, int] = {}
    for node in topo_order:
        preds = list(G.predecessors(node))
        if not preds:
            layers[node] = 0
        else:
            layers[node] = max(layers.get(p, 0) for p in preds) + 1

    layer_groups: Dict[int, List[str]] = {}
    for node, layer in layers.items():
        layer_groups.setdefault(layer, []).append(node)

    disj_units: Dict[str, Set[str]] = {}
    for dj in superstructure.disjunctions:
        for uid in dj.unit_ids:
            disj_units.setdefault(dj.name, set()).add(uid)

    positions: Dict[str, Tuple[float, float]] = {}
    x_spacing = 2.5
    y_spacing = 2.0

    for layer_idx in sorted(layer_groups.keys()):
        nodes = layer_groups[layer_idx]
        x = layer_idx * x_spacing

        disj_members = set()
        for dname, members in disj_units.items():
            overlap = members & set(nodes)
            if len(overlap) > 1:
                disj_members |= overlap

        non_disj = [n for n in nodes if n not in disj_members]
        disj_list = [n for n in nodes if n in disj_members]
        ordered = non_disj + disj_list
        n = len(ordered)

        for i, node in enumerate(ordered):
            y = (n - 1) / 2.0 * y_spacing - i * y_spacing
            positions[node] = (x, y)

    return positions


# ═══════════════════════════════════════════════════════════════════════
# Main visualization function
# ═══════════════════════════════════════════════════════════════════════

def visualize_superstructure(
    superstructure_or_name: Any,
    save: Optional[str] = None,
    figsize: Tuple[float, float] = (20, 10),
    dpi: int = 150,
    show_stream_labels: bool = True,
    title: Optional[str] = None,
    components_dir: Optional[str] = None,
) -> plt.Figure:
    """Visualize a superstructure as a process flow diagram.

    Parameters
    ----------
    superstructure_or_name : Superstructure | str
        A Pydantic Superstructure object, or a name/path for YAML loading.
    save : str, optional
        Output file path (e.g. ``"pfd.png"``, ``"pfd.svg"``).
    figsize : tuple
        Figure size in inches.
    dpi : int
        Resolution for raster output.
    show_stream_labels : bool
        If True, label edges with stream names.
    title : str, optional
        Custom title (default: superstructure name).
    components_dir : str, optional
        Path to component YAML directory for loading.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Resolve input
    if isinstance(superstructure_or_name, str):
        from ..dsl.yaml_loader import load_superstructure
        ss = load_superstructure(superstructure_or_name, components_dir)
    else:
        ss = superstructure_or_name

    _log.info("Visualizing superstructure '%s'", ss.name)

    # Build graph and layout
    G = _build_graph(ss)
    positions = _hierarchical_layout(G, ss)

    # Detect heat exchanger pairs
    hx_pairs = _detect_hx_pairs(ss)
    paired_hx_ids: Set[str] = set()
    for cold_id, hot_id, _ in hx_pairs:
        paired_hx_ids.add(cold_id)
        paired_hx_ids.add(hot_id)

    # Index units
    unit_map = {u.id: u for u in ss.base_flowsheet.units}

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="white")
    ax.set_facecolor("#FAFAFA")
    ax.set_aspect("equal")

    # ── Draw disjunction boxes first (behind everything) ─────────
    for i, dj in enumerate(ss.disjunctions):
        ci = i % len(DISJUNCTION_COLORS)
        _draw_disjunction_box(
            ax, positions, dj.unit_ids, dj.name, dj.description,
            DISJUNCTION_COLORS[ci], DISJUNCTION_BORDER_COLORS[ci],
        )

    # ── Draw heat coupling arcs between paired HXs ───────────────
    for cold_id, hot_id, group_label in hx_pairs:
        if cold_id in positions and hot_id in positions:
            cold_u = unit_map.get(cold_id)
            hot_u = unit_map.get(hot_id)
            _draw_heat_coupling(
                ax,
                positions[cold_id], positions[hot_id],
                group_label,
                cold_input=cold_u.inputs[0] if cold_u and cold_u.inputs else "",
                hot_input=hot_u.inputs[0] if hot_u and hot_u.inputs else "",
            )

    # ── Draw edges (streams) ─────────────────────────────────────
    for u, v, data in G.edges(data=True):
        stream_name = data.get("stream", "")
        if u not in positions or v not in positions:
            continue

        x1, y1 = positions[u]
        x2, y2 = positions[v]

        is_feed_edge = u.startswith("feed:")
        is_prod_edge = v.startswith("prod:")

        edge_color = "#2C3E50"
        if is_feed_edge:
            edge_color = "#27AE60"
        elif is_prod_edge:
            edge_color = "#E74C3C"

        ax.annotate(
            "",
            xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_color,
                lw=1.5,
                shrinkA=20, shrinkB=20,
                connectionstyle="arc3,rad=0.05",
            ),
            zorder=2,
        )

        if show_stream_labels and stream_name:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            offset = 0.12
            angle = math.atan2(y2 - y1, x2 - x1)
            lx = mx - offset * math.sin(angle)
            ly = my + offset * math.cos(angle)

            label = stream_name.replace("_", " ")
            if len(label) > 20:
                label = label[:18] + "…"

            ax.text(
                lx, ly, label,
                ha="center", va="center",
                fontsize=4.5, color="#555555",
                fontstyle="italic",
                rotation=math.degrees(angle),
                zorder=5,
                bbox=dict(
                    boxstyle="round,pad=0.1",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.7,
                ),
            )

    # ── Draw unit nodes ──────────────────────────────────────────
    for u in ss.base_flowsheet.units:
        if u.id not in positions:
            continue
        x, y = positions[u.id]
        style = UNIT_STYLES.get(u.type, DEFAULT_STYLE)

        is_paired = u.id in paired_hx_ids
        is_utility = (u.type == "heat_exchanger" and not is_paired)

        _draw_unit_node(
            ax, x, y, u.id, u.type, style, u.optional,
            is_paired_hx=is_paired,
            is_utility_hx=is_utility,
        )

    # ── Draw feed/product markers ────────────────────────────────
    for node in G.nodes():
        if node not in positions:
            continue
        x, y = positions[node]

        if node.startswith("feed:"):
            stream_name = node.replace("feed:", "").replace("_", " ")
            ax.plot(x, y, "o", markersize=10, color="#27AE60",
                    markeredgecolor="#1A7A3E", markeredgewidth=1.5, zorder=3)
            ax.text(x, y + 0.35, stream_name, ha="center", va="bottom",
                    fontsize=6, fontweight="bold", color="#1A7A3E", zorder=4)

        elif node.startswith("prod:"):
            stream_name = node.replace("prod:", "").replace("_", " ")
            ax.plot(x, y, "s", markersize=10, color="#E74C3C",
                    markeredgecolor="#A93226", markeredgewidth=1.5, zorder=3)
            ax.text(x, y - 0.35, stream_name, ha="center", va="top",
                    fontsize=6, fontweight="bold", color="#A93226", zorder=4)

    # ── Title and legend ─────────────────────────────────────────
    plot_title = title or f"Superstructure: {ss.name.replace('_', ' ').title()}"
    ax.set_title(plot_title, fontsize=14, fontweight="bold", pad=15)

    present_types = {u.type for u in ss.base_flowsheet.units}
    legend_handles = []
    for utype in sorted(present_types):
        style = UNIT_STYLES.get(utype, DEFAULT_STYLE)
        patch = mpatches.Patch(
            facecolor=style["color"], edgecolor="#333333",
            label=f'{style["icon"]}  {utype.replace("_", " ").title()}',
            alpha=0.85,
        )
        legend_handles.append(patch)

    # Disjunction legend entries
    for i, dj in enumerate(ss.disjunctions):
        ci = i % len(DISJUNCTION_BORDER_COLORS)
        patch = mpatches.Patch(
            facecolor=DISJUNCTION_COLORS[ci],
            edgecolor=DISJUNCTION_BORDER_COLORS[ci],
            label=f'⊻ {dj.name.replace("_", " ").title()}',
            linestyle="--", linewidth=1.5,
        )
        legend_handles.append(patch)

    # Heat coupling legend
    if hx_pairs:
        legend_handles.append(mpatches.Patch(
            facecolor="#FFF3E0", edgecolor=HEAT_COUPLING_COLOR,
            label="Q~ Heat Coupling", linestyle="--", linewidth=1.5,
        ))

    # Feed/product markers
    legend_handles.append(mpatches.Patch(
        facecolor="#27AE60", edgecolor="#1A7A3E", label="● Feed Stream"
    ))
    legend_handles.append(mpatches.Patch(
        facecolor="#E74C3C", edgecolor="#A93226", label="■ Product Stream"
    ))

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
        edgecolor="#CCCCCC",
        ncol=2,
    )

    # Auto-scale
    all_x = [p[0] for p in positions.values()]
    all_y = [p[1] for p in positions.values()]
    margin = 1.5
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
    ax.axis("off")

    plt.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        _log.info("Saved PFD to %s", save)

    return fig


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    """CLI: python -m sep_agents.vis.flowsheet_vis <name> [-o output.png]"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Visualize a superstructure flowsheet"
    )
    parser.add_argument("name", help="Superstructure name or YAML path")
    parser.add_argument("-o", "--output", default=None,
                        help="Output file path (png, svg, pdf)")
    parser.add_argument("--no-labels", action="store_true",
                        help="Hide stream labels")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--width", type=float, default=20)
    parser.add_argument("--height", type=float, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    fig = visualize_superstructure(
        args.name,
        save=args.output,
        figsize=(args.width, args.height),
        dpi=args.dpi,
        show_stream_labels=not args.no_labels,
    )

    if not args.output:
        plt.show()


if __name__ == "__main__":
    main()
