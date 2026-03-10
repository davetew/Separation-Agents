"""
YAML Loader for Component, Superstructure & Raw Material Libraries
===================================================================

Loads YAML-defined components, superstructures, and raw materials.
Superstructures are converted into Pydantic :class:`Superstructure`
objects compatible with the GDP solver pipeline.

Usage
-----
>>> from sep_agents.dsl.yaml_loader import load_superstructure, load_raw_material
>>> ss = load_superstructure("eaf_steel_slag")
>>> mat = load_raw_material("olivine")

The loader automatically resolves ``component:`` references in unit
definitions against the built-in component library, merging defaults
with instance-specific overrides.

Raw materials are stored as flat YAML dicts describing composition,
conditions, value streams, and commodity prices.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .schemas import (
    DisjunctionDef,
    Flowsheet,
    Stream,
    Superstructure,
    UnitOp,
)

_log = logging.getLogger(__name__)

# Default search paths
_COMPONENTS_DIR = Path(__file__).parent / "components"
_SUPERSTRUCTURES_DIR = Path(__file__).parent / "superstructures"
_RAW_MATERIALS_DIR = Path(__file__).parent / "raw_materials"


# ---------------------------------------------------------------------------
# Component Loading
# ---------------------------------------------------------------------------

def load_components(
    directory: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Load all component YAML files from a directory.

    Parameters
    ----------
    directory : str, optional
        Path to component YAML files.  Defaults to the built-in
        ``dsl/components/`` directory.

    Returns
    -------
    dict
        Map of component type → parsed YAML dict (with keys:
        ``type``, ``description``, ``defaults``, ``bounds``,
        ``required_params``, ``optional_params``).
    """
    comp_dir = Path(directory) if directory else _COMPONENTS_DIR
    components: Dict[str, Dict[str, Any]] = {}

    if not comp_dir.exists():
        _log.warning("Components directory not found: %s", comp_dir)
        return components

    for yaml_path in sorted(comp_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if data and "type" in data:
            comp_type = data["type"]
            # Normalise missing keys
            data.setdefault("defaults", {})
            data.setdefault("bounds", {})
            data.setdefault("required_params", [])
            data.setdefault("optional_params", [])
            data.setdefault("description", "")
            components[comp_type] = data
            _log.debug("Loaded component: %s from %s", comp_type, yaml_path.name)
        else:
            _log.warning("Skipping invalid component file: %s", yaml_path)

    _log.info("Loaded %d component definitions from %s", len(components), comp_dir)
    return components


# ---------------------------------------------------------------------------
# Superstructure Loading
# ---------------------------------------------------------------------------

def _resolve_unit_params(
    unit_data: Dict[str, Any],
    components: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge component defaults with instance params.

    Instance params override component defaults.
    """
    comp_type = unit_data.get("component", "")
    instance_params = dict(unit_data.get("params", {}))

    if comp_type in components:
        comp = components[comp_type]
        # Start with component defaults, then overlay instance overrides
        merged = dict(comp.get("defaults", {}))
        merged.update(instance_params)
        return merged
    else:
        # No matching component — use raw params
        return instance_params


def _parse_stream(name: str, data: Dict[str, Any]) -> Stream:
    """Parse a stream from YAML data."""
    return Stream(
        name=name,
        phase=data.get("phase", "liquid"),
        temperature_K=float(data.get("temperature_K", 298.15)),
        pressure_Pa=float(data.get("pressure_Pa", 101325.0)),
        composition_wt=data.get("composition_wt", {}),
        pH=data.get("pH"),
        Eh_mV=data.get("Eh_mV"),
        solids_wtfrac=data.get("solids_wtfrac"),
    )


def _parse_unit(
    uid: str,
    data: Dict[str, Any],
    components: Dict[str, Dict[str, Any]],
) -> UnitOp:
    """Parse a unit operation from YAML data, resolving component refs."""
    comp_type = data.get("component", data.get("type", "mixer"))
    params = _resolve_unit_params(data, components)

    return UnitOp(
        id=uid,
        type=comp_type,
        params=params,
        inputs=data.get("inputs", []),
        outputs=data.get("outputs", []),
        optional=data.get("optional", False),
        alternatives=data.get("alternatives", []),
        stage_range=tuple(data["stage_range"]) if "stage_range" in data else None,
    )


def load_superstructure(
    name_or_path: str,
    components_dir: Optional[str] = None,
) -> Superstructure:
    """Load a YAML superstructure and return a Pydantic Superstructure.

    Parameters
    ----------
    name_or_path : str
        Either a built-in superstructure name (e.g. ``"eaf_steel_slag"``)
        or an absolute/relative path to a YAML file.
    components_dir : str, optional
        Path to component YAML directory.  Defaults to the built-in library.

    Returns
    -------
    Superstructure
        A fully resolved Pydantic Superstructure ready for the GDP solver.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If the YAML is malformed or references unknown components.
    """
    # Resolve path
    path = Path(name_or_path)
    if not path.suffix:
        # Try built-in superstructures directory
        path = _SUPERSTRUCTURES_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Superstructure YAML not found: {path}\n"
            f"Available: {[p.stem for p in _SUPERSTRUCTURES_DIR.glob('*.yaml')]}"
        )

    # Load component library
    components = load_components(components_dir)

    # Parse YAML
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(data).__name__}")

    ss_name = data.get("name", path.stem)
    _log.info("Loading superstructure '%s' from %s", ss_name, path)

    # ── Parse streams ────────────────────────────────────────────
    streams_data = data.get("streams", {})
    streams: List[Stream] = []
    if isinstance(streams_data, dict):
        # dict-of-dicts format: {stream_name: {phase: ..., ...}}
        for sname, sdata in streams_data.items():
            if sdata is None:
                sdata = {}
            streams.append(_parse_stream(sname, sdata))
    elif isinstance(streams_data, list):
        # list-of-dicts format: [{name: ..., phase: ..., ...}, ...]
        for sdata in streams_data:
            sname = sdata.get("name", f"stream_{len(streams)}")
            streams.append(_parse_stream(sname, sdata))
    else:
        _log.warning("Unexpected streams format: %s", type(streams_data).__name__)

    # ── Parse units ──────────────────────────────────────────────
    units_data = data.get("units", {})
    units: List[UnitOp] = []
    unknown_components = []

    if isinstance(units_data, dict):
        # dict-of-dicts format: {unit_id: {type: ..., ...}}
        for uid, udata in units_data.items():
            if udata is None:
                udata = {}
            comp_type = udata.get("component", udata.get("type", "mixer"))
            if comp_type not in components:
                unknown_components.append((uid, comp_type))
            units.append(_parse_unit(uid, udata, components))
    elif isinstance(units_data, list):
        # list-of-dicts format: [{id: ..., type: ..., ...}, ...]
        for udata in units_data:
            uid = udata.get("id", f"unit_{len(units)}")
            comp_type = udata.get("component", udata.get("type", "mixer"))
            if comp_type not in components:
                unknown_components.append((uid, comp_type))
            units.append(_parse_unit(uid, udata, components))

    if unknown_components:
        _log.warning(
            "Unknown component types (param validation skipped): %s",
            [(uid, ct) for uid, ct in unknown_components],
        )

    # ── Parse disjunctions ───────────────────────────────────────
    disj_data = data.get("disjunctions", {})
    disjunctions: List[DisjunctionDef] = []
    if isinstance(disj_data, dict):
        # dict-of-dicts format: {disj_name: {choices: [...], ...}}
        for dname, ddata in disj_data.items():
            disjunctions.append(DisjunctionDef(
                name=dname,
                unit_ids=ddata.get("choices", []),
                description=ddata.get("description", ""),
            ))
    elif isinstance(disj_data, list):
        # list-of-dicts format: [{name: ..., unit_ids: [...], ...}, ...]
        for ddata in disj_data:
            disjunctions.append(DisjunctionDef(
                name=ddata.get("name", f"disj_{len(disjunctions)}"),
                unit_ids=ddata.get("unit_ids", ddata.get("choices", [])),
                description=ddata.get("description", ""),
            ))

    # ── Parse continuous bounds ──────────────────────────────────
    cb_raw = data.get("continuous_bounds", {})
    continuous_bounds: Dict[str, Tuple[float, float]] = {}
    for key, val in cb_raw.items():
        if isinstance(val, (list, tuple)) and len(val) == 2:
            continuous_bounds[key] = (float(val[0]), float(val[1]))
        else:
            _log.warning("Invalid continuous_bounds entry: %s = %s", key, val)

    # ── Build Superstructure ─────────────────────────────────────
    flowsheet = Flowsheet(
        name=f"{ss_name}_base",
        units=units,
        streams=streams,
    )

    ss = Superstructure(
        name=ss_name,
        base_flowsheet=flowsheet,
        disjunctions=disjunctions,
        fixed_units=data.get("fixed_units", []),
        objective=data.get("objective", "minimize_opex"),
        continuous_bounds=continuous_bounds,
        description=data.get("description", ""),
    )

    _log.info(
        "Loaded superstructure '%s': %d units, %d streams, "
        "%d disjunctions, %d continuous vars",
        ss.name, len(units), len(streams),
        len(disjunctions), len(continuous_bounds),
    )
    return ss


def list_superstructures(
    directory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available YAML-defined superstructures.

    Returns
    -------
    list[dict]
        List of dicts with ``name``, ``path``, and ``description``.
    """
    ss_dir = Path(directory) if directory else _SUPERSTRUCTURES_DIR
    result = []

    if not ss_dir.exists():
        return result

    for yaml_path in sorted(ss_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if data and isinstance(data, dict):
            result.append({
                "name": data.get("name", yaml_path.stem),
                "path": str(yaml_path),
                "description": data.get("description", ""),
            })

    return result


# ---------------------------------------------------------------------------
# Raw Material Loading
# ---------------------------------------------------------------------------

_RAW_MATERIAL_REQUIRED_KEYS = {"name", "composition"}


def load_raw_material(
    name_or_path: str,
    directory: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a raw material definition from YAML.

    Parameters
    ----------
    name_or_path : str
        Built-in material name (e.g. ``"eaf_steel_slag"``) or path
        to a YAML file.
    directory : str, optional
        Override the default ``dsl/raw_materials/`` search directory.

    Returns
    -------
    dict
        Parsed material data with keys: ``name``, ``description``,
        ``source``, ``throughput_tpd``, ``physical_form``,
        ``composition``, ``conditions``, ``value_streams``,
        ``commodity_prices``.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    ValueError
        If required keys are missing.
    """
    mat_dir = Path(directory) if directory else _RAW_MATERIALS_DIR
    path = Path(name_or_path)
    if not path.suffix:
        path = mat_dir / f"{name_or_path}.yaml"
    if not path.exists():
        available = [p.stem for p in mat_dir.glob("*.yaml")] if mat_dir.exists() else []
        raise FileNotFoundError(
            f"Raw material YAML not found: {path}\n"
            f"Available: {available}"
        )

    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping, got {type(data).__name__}")

    # Validate required keys
    missing = _RAW_MATERIAL_REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(
            f"Raw material '{path.name}' missing required keys: {missing}"
        )

    # Normalise optional fields
    data.setdefault("name", path.stem)
    data.setdefault("description", "")
    data.setdefault("source", "")
    data.setdefault("throughput_tpd", 0.0)
    data.setdefault("physical_form", "solid")
    data.setdefault("conditions", {"temperature_K": 298.15, "pressure_Pa": 101325.0})
    data.setdefault("value_streams", [])
    data.setdefault("commodity_prices", {})

    _log.info("Loaded raw material '%s' from %s", data["name"], path)
    return data


def list_raw_materials(
    directory: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List available raw material definitions.

    Returns
    -------
    list[dict]
        List of dicts with ``name``, ``path``, ``description``,
        ``physical_form``, and ``value_streams``.
    """
    mat_dir = Path(directory) if directory else _RAW_MATERIALS_DIR
    result = []

    if not mat_dir.exists():
        return result

    for yaml_path in sorted(mat_dir.glob("*.yaml")):
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        if data and isinstance(data, dict):
            result.append({
                "name": data.get("name", yaml_path.stem),
                "path": str(yaml_path),
                "description": data.get("description", ""),
                "physical_form": data.get("physical_form", "solid"),
                "value_streams": data.get("value_streams", []),
            })

    return result


def save_raw_material(
    data: Dict[str, Any],
    name: Optional[str] = None,
    directory: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """Persist a raw material definition to the YAML library.

    Parameters
    ----------
    data : dict
        Material data. Must contain at least ``name`` and
        ``composition`` keys.
    name : str, optional
        Override filename (without extension). Defaults to
        ``data["name"]`` with spaces/slashes replaced by underscores.
    directory : str, optional
        Override the default ``dsl/raw_materials/`` directory.
    overwrite : bool
        If False (default), raise an error if the file already exists.

    Returns
    -------
    Path
        Absolute path to the saved YAML file.

    Raises
    ------
    ValueError
        If required keys are missing.
    FileExistsError
        If the file already exists and ``overwrite`` is False.
    """
    if not isinstance(data, dict):
        raise ValueError("data must be a dict")

    missing = _RAW_MATERIAL_REQUIRED_KEYS - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys: {missing}")

    mat_dir = Path(directory) if directory else _RAW_MATERIALS_DIR
    mat_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename
    fname = name or data.get("name", "unnamed")
    fname = fname.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    out_path = mat_dir / f"{fname}.yaml"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Raw material file already exists: {out_path}\n"
            f"Use overwrite=True to replace."
        )

    with open(out_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    _log.info("Saved raw material '%s' to %s", data.get("name", fname), out_path)
    _regenerate_readme()
    return out_path


def save_superstructure(
    yaml_content: str,
    name: str,
    overwrite: bool = False,
) -> Path:
    """Save a superstructure YAML string to the library.

    Parameters
    ----------
    yaml_content : str
        The full YAML content for the superstructure.
    name : str
        Filename (without extension).
    overwrite : bool
        If False (default), raise an error if the file exists.

    Returns
    -------
    Path
        Absolute path to the saved YAML file.
    """
    _SUPERSTRUCTURES_DIR.mkdir(parents=True, exist_ok=True)
    fname = name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
    out_path = _SUPERSTRUCTURES_DIR / f"{fname}.yaml"

    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Superstructure file already exists: {out_path}\n"
            f"Use overwrite=True to replace."
        )

    with open(out_path, "w") as f:
        f.write(yaml_content)

    _log.info("Saved superstructure '%s' to %s", name, out_path)
    _regenerate_readme()
    return out_path


# ---------------------------------------------------------------------------
# README Auto-regeneration
# ---------------------------------------------------------------------------

def _regenerate_readme() -> None:
    """Regenerate the DSL README.md after library changes."""
    try:
        from .generate_readme import generate_readme
        generate_readme()
        _log.debug("Auto-regenerated DSL README.md")
    except Exception as e:
        _log.warning("Could not auto-regenerate README: %s", e)
