"""
Auto-generate the DSL README.md from the YAML libraries.

Run directly to regenerate:
    python -m sep_agents.dsl.generate_readme

Or call programmatically:
    from sep_agents.dsl.generate_readme import generate_readme
    generate_readme()
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Dict, List

import yaml

_DSL_DIR = Path(__file__).parent
_COMPONENTS_DIR = _DSL_DIR / "components"
_SUPERSTRUCTURES_DIR = _DSL_DIR / "superstructures"
_RAW_MATERIALS_DIR = _DSL_DIR / "raw_materials"
_README_PATH = _DSL_DIR / "README.md"


def _load_yamls(directory: Path) -> List[Dict[str, Any]]:
    """Load all YAML files from a directory, sorted by name."""
    results = []
    if not directory.exists():
        return results
    for p in sorted(directory.glob("*.yaml")):
        with open(p) as f:
            data = yaml.safe_load(f)
        if data and isinstance(data, dict):
            data["_path"] = p
            results.append(data)
    return results


def _fmt_description(desc: str, width: int = 80) -> str:
    """Clean up a YAML multi-line description into a single paragraph."""
    return " ".join(desc.split())


def _render_components(components: List[Dict[str, Any]]) -> str:
    """Render the components section."""
    lines = [
        "## Components",
        "",
        f"**{len(components)} unit operation types** defined in "
        "`components/`:",
        "",
        "| Type | Description | Default Parameters |",
        "|------|-------------|--------------------|",
    ]
    for c in components:
        ctype = c.get("type", c["_path"].stem)
        desc = _fmt_description(c.get("description", ""))
        if len(desc) > 60:
            desc = desc[:57] + "..."
        defaults = c.get("defaults", {})
        params_str = ", ".join(f"`{k}`={v}" for k, v in list(defaults.items())[:3])
        if len(defaults) > 3:
            params_str += ", ..."
        lines.append(f"| `{ctype}` | {desc} | {params_str} |")
    lines.append("")
    return "\n".join(lines)


def _render_superstructures(superstructures: List[Dict[str, Any]]) -> str:
    """Render the superstructures section with flowsheet details."""
    lines = [
        "## Superstructures",
        "",
        f"**{len(superstructures)} GDP superstructures** defined in "
        "`superstructures/`:",
        "",
    ]

    for ss in superstructures:
        name = ss.get("name", ss["_path"].stem)
        desc = _fmt_description(ss.get("description", ""))
        objective = ss.get("objective", "minimize_opex")

        # Count units, streams, disjunctions, continuous bounds
        units_data = ss.get("units", {})
        if isinstance(units_data, dict):
            n_units = len(units_data)
            unit_ids = list(units_data.keys())
        elif isinstance(units_data, list):
            n_units = len(units_data)
            unit_ids = [u.get("id", f"unit_{i}") for i, u in enumerate(units_data)]
        else:
            n_units = 0
            unit_ids = []

        streams_data = ss.get("streams", {})
        n_streams = len(streams_data) if isinstance(streams_data, (dict, list)) else 0

        disj_data = ss.get("disjunctions", {})
        if isinstance(disj_data, dict):
            disjunctions = list(disj_data.items())
        elif isinstance(disj_data, list):
            disjunctions = [(d.get("name", f"disj_{i}"), d) for i, d in enumerate(disj_data)]
        else:
            disjunctions = []

        cb = ss.get("continuous_bounds", {})
        n_bounds = len(cb)

        fixed = ss.get("fixed_units", [])

        lines.append(f"### `{name}`")
        lines.append("")
        lines.append(f"> {desc}")
        lines.append("")
        lines.append(f"- **Objective**: `{objective}`")
        lines.append(f"- **Units**: {n_units} — **Streams**: {n_streams} — "
                      f"**Disjunctions**: {len(disjunctions)} — "
                      f"**Continuous bounds**: {n_bounds}")
        lines.append("")

        # Unit list
        lines.append("**Unit operations:**")
        lines.append("")

        # Categorise units
        optional_units = set()
        for uid, udata_pair in _iter_units(ss):
            if udata_pair.get("optional", False):
                optional_units.add(uid)

        for uid in unit_ids:
            marker = " *(optional)*" if uid in optional_units else ""
            lines.append(f"- `{uid}`{marker}")
        lines.append("")

        # Disjunctions table
        if disjunctions:
            lines.append("**GDP disjunctions:**")
            lines.append("")
            lines.append("| Disjunction | Choices | Description |")
            lines.append("|-------------|---------|-------------|")
            for dname, ddata in disjunctions:
                if isinstance(ddata, dict):
                    choices = ddata.get("choices", ddata.get("unit_ids", []))
                    ddesc = ddata.get("description", "")
                else:
                    choices = []
                    ddesc = ""
                choices_str = ", ".join(f"`{c}`" for c in choices)
                lines.append(f"| `{dname}` | {choices_str} | {ddesc} |")
            lines.append("")

        # Continuous bounds
        if cb:
            lines.append("**Continuous design variables (BoTorch bounds):**")
            lines.append("")
            lines.append("| Variable | Min | Max |")
            lines.append("|----------|----:|----:|")
            for var, bounds in cb.items():
                if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                    lines.append(f"| `{var}` | {bounds[0]} | {bounds[1]} |")
            lines.append("")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _iter_units(ss: Dict[str, Any]):
    """Yield (uid, udata) pairs regardless of dict or list format."""
    units_data = ss.get("units", {})
    if isinstance(units_data, dict):
        for uid, udata in units_data.items():
            yield uid, (udata or {})
    elif isinstance(units_data, list):
        for udata in units_data:
            yield udata.get("id", "unknown"), udata


def _render_raw_materials(materials: List[Dict[str, Any]]) -> str:
    """Render the raw materials section."""
    if not materials:
        return ""
    lines = [
        "## Raw Materials",
        "",
        f"**{len(materials)} raw materials** defined in "
        "`raw_materials/`:",
        "",
        "| Name | Physical Form | Value Streams | Throughput |",
        "|------|--------------|---------------|-----------|",
    ]
    for m in materials:
        name = m.get("name", m["_path"].stem)
        form = m.get("physical_form", "solid")
        streams = ", ".join(m.get("value_streams", []))
        tpd = m.get("throughput_tpd", 0)
        throughput = f"{tpd:.0f} t/d" if tpd else "—"
        lines.append(f"| `{name}` | {form} | {streams} | {throughput} |")
    lines.append("")
    return "\n".join(lines)


def generate_readme(output_path: Path | str | None = None) -> str:
    """Generate the DSL README.md content and write it to disk.

    Parameters
    ----------
    output_path : Path or str, optional
        Override output path. Defaults to ``dsl/README.md``.

    Returns
    -------
    str
        The generated Markdown content.
    """
    components = _load_yamls(_COMPONENTS_DIR)
    superstructures = _load_yamls(_SUPERSTRUCTURES_DIR)
    raw_materials = _load_yamls(_RAW_MATERIALS_DIR)

    sections = [
        "# Domain Specific Language (`dsl`)",
        "",
        "This module defines the YAML-based domain specific language for "
        "chemical process superstructure optimization.",
        "",
        "**Core files:**",
        "",
        "- **`schemas.py`** — Pydantic models (`Stream`, `UnitOp`, "
        "`Flowsheet`, `Superstructure`, `DisjunctionDef`)",
        "- **`yaml_loader.py`** — Load components, superstructures, "
        "and raw materials from YAML",
        "- **`generate_readme.py`** — Auto-regenerate this README "
        "from the YAML libraries",
        "",
        "> **Note**: This README is auto-generated. Run "
        "`python -m sep_agents.dsl.generate_readme` to update it.",
        "",
        "---",
        "",
        _render_components(components),
        "---",
        "",
        _render_superstructures(superstructures),
        _render_raw_materials(raw_materials),
    ]

    content = "\n".join(sections)

    out = Path(output_path) if output_path else _README_PATH
    with open(out, "w") as f:
        f.write(content)

    return content


if __name__ == "__main__":
    content = generate_readme()
    print(f"Generated {_README_PATH} ({len(content)} bytes)")
