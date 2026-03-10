"""
Provenance tracking for valorization analyses.

Captures the full chain: user requirements → assumptions → all topology
results → optimal selection → final economics, and saves it alongside
the report/presentation as ``provenance.json``.

Usage
-----
>>> from sep_agents.provenance import (
...     ValorizationProvenance, FeedSpec, Assumptions,
...     TopologyResult, ProductRevenue, CostItem, FinalEconomics,
... )
>>> prov = ValorizationProvenance(
...     raw_material="eaf_steel_slag",
...     feed=FeedSpec(throughput_tpy=100_000, ...),
...     assumptions=Assumptions(...),
...     topology_results=[TopologyResult(...), ...],
...     optimal_topology_id="T1",
...     final_economics=FinalEconomics(...),
... )
>>> prov.validate_consistency()   # raises if topology↔revenue mismatch
>>> prov.save("reports/eaf_steel_slag_2026-03-09_0535/")
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Feed specification
# ---------------------------------------------------------------------------

class FeedSpec(BaseModel):
    """Raw material feed characterization."""

    throughput_tpy: float = Field(..., description="Annual throughput (t/yr)")
    operating_days: int = Field(330, description="Operating days per year")
    composition_wt: Dict[str, float] = Field(
        ..., description="Composition as {species: wt_frac}"
    )
    speciation_database: str = Field(
        "supcrtbl", description="Thermodynamic database used for speciation"
    )
    speciation_notes: Optional[str] = Field(
        None, description="Key speciation findings"
    )


# ---------------------------------------------------------------------------
# Assumptions
# ---------------------------------------------------------------------------

class CommodityPrice(BaseModel):
    """A single commodity price assumption."""

    product: str
    price: float = Field(..., description="USD per unit")
    unit: str = Field("USD/t", description="Price unit (e.g., USD/t, USD/kg)")
    source: str = Field("", description="Source citation or 'assumed'")


class Assumptions(BaseModel):
    """All economic and technical assumptions."""

    discount_rate: float = Field(0.08, description="Discount rate (fraction)")
    project_life_yr: int = Field(20, description="Project life in years")
    capex_usd: float = Field(..., description="Total CAPEX in USD")
    commodity_prices: List[CommodityPrice] = Field(
        default_factory=list,
        description="Assumed commodity prices for all products"
    )
    labor_fte: int = Field(4, description="Full-time equivalent operators")
    labor_rate_usd_hr: float = Field(45.0, description="Labor cost per hour")
    electricity_usd_kwh: float = Field(0.08, description="Electricity cost")
    maintenance_pct_capex: float = Field(
        0.02, description="Annual maintenance as fraction of CAPEX"
    )
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional assumptions not covered above"
    )


# ---------------------------------------------------------------------------
# Topology results (ALL topologies, not just optimal)
# ---------------------------------------------------------------------------

class UnitKPI(BaseModel):
    """Key performance indicator for a single unit operation."""

    unit_id: str
    unit_type: str
    kpi_name: str = Field(..., description="e.g., 'recovery', 'extent_mol'")
    value: float
    unit: str = Field("", description="Unit of the KPI value")
    notes: str = Field("", description="e.g., 'zero extent at equilibrium'")


class ProductRevenue(BaseModel):
    """Revenue from a single product in a topology."""

    product: str = Field(..., description="Product name (e.g., Cr2O3)")
    volume_tpy: float = Field(..., description="Annual production volume (t/yr)")
    price_usd_t: float = Field(..., description="Assumed price (USD/t)")
    revenue_usd_yr: float = Field(..., description="Annual revenue (USD/yr)")
    from_unit: str = Field(
        ..., description="Unit ID that produces this product"
    )


class CostItem(BaseModel):
    """A single OPEX or CAPEX line item."""

    category: str = Field(..., description="e.g., 'HCl', 'Labor', 'LIMS'")
    annual_cost_usd: float = Field(..., description="Annual cost (USD/yr)")
    share_pct: float = Field(0.0, description="Share of total (%)")
    specific_cost: str = Field(
        "", description="Nominal specific cost (e.g., '$0.15/kg')"
    )
    cost_type: Literal["capex", "opex"] = "opex"


class TopologyResult(BaseModel):
    """Complete results for a single topology configuration."""

    topology_id: str = Field(..., description="e.g., 'T1', 'T2', ...")
    name: str = Field(..., description="Human-readable name")
    active_units: List[str] = Field(
        ..., description="List of active unit IDs in this topology"
    )
    excluded_units: List[str] = Field(
        default_factory=list,
        description="Excluded unit IDs with reasons"
    )
    exclusion_reasons: Dict[str, str] = Field(
        default_factory=dict,
        description="Unit ID → reason for exclusion"
    )
    unit_kpis: List[UnitKPI] = Field(
        default_factory=list,
        description="Per-unit performance KPIs"
    )
    products: List[ProductRevenue] = Field(
        default_factory=list,
        description="Revenue breakdown for this topology"
    )
    cost_items: List[CostItem] = Field(
        default_factory=list,
        description="OPEX and CAPEX line items"
    )
    # Aggregate economics
    revenue_usd_t: float = Field(0.0, description="Total revenue per tonne")
    opex_usd_t: float = Field(0.0, description="Total OPEX per tonne")
    margin_usd_t: float = Field(0.0, description="Revenue - OPEX per tonne")
    capex_usd: float = Field(0.0, description="Total CAPEX (USD)")
    npv_usd: float = Field(0.0, description="Net Present Value (USD)")
    irr_pct: float = Field(0.0, description="Internal Rate of Return (%)")
    payback_yr: float = Field(0.0, description="Simple payback (years)")
    levelized_net_usd_t: float = Field(
        0.0, description="Levelized net value of production (USD/t)"
    )
    rank: int = Field(0, description="Rank among all topologies (1 = best)")
    is_optimal: bool = Field(False, description="True if this is the selected topology")


# ---------------------------------------------------------------------------
# Final economics (derived from optimal topology)
# ---------------------------------------------------------------------------

class FinalEconomics(BaseModel):
    """Headline economics for the optimal topology."""

    revenue_usd_t: float
    opex_usd_t: float
    eac_usd_t: float = Field(..., description="Equivalent Annual Capital Cost per tonne")
    levelized_net_usd_t: float
    annual_revenue_usd: float
    annual_opex_usd: float
    capex_usd: float
    npv_usd: float
    irr_pct: float
    payback_yr: float


# ---------------------------------------------------------------------------
# Root provenance model
# ---------------------------------------------------------------------------

class ValorizationProvenance(BaseModel):
    """Full provenance record for a valorization analysis.

    Saved as ``provenance.json`` alongside report/presentation files.
    """

    # ── Metadata ──
    raw_material: str = Field(..., description="Raw material identifier")
    analysis_timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 timestamp of analysis"
    )
    analyst: str = Field("automated", description="Who ran the analysis")
    software_versions: Dict[str, str] = Field(
        default_factory=dict,
        description="e.g., {'sep_agents': '0.3', 'reaktoro': '2.x'}"
    )

    # ── Inputs ──
    feed: FeedSpec
    assumptions: Assumptions

    # ── Intermediates — ALL topologies ──
    superstructure_name: str = Field(
        "", description="Name of the GDP superstructure YAML"
    )
    n_topologies_evaluated: int = Field(
        0, description="Total number of configurations enumerated"
    )
    topology_results: List[TopologyResult] = Field(
        ..., description="Results for ALL topologies considered (ranked)"
    )

    # ── Selection ──
    optimal_topology_id: str = Field(
        ..., description="ID of the selected optimal topology"
    )
    selection_rationale: str = Field(
        "", description="Why this topology was selected"
    )

    # ── Final outputs ──
    final_economics: FinalEconomics

    # ── Output files ──
    report_path: str = Field("", description="Relative path to report PDF")
    presentation_path: str = Field(
        "", description="Relative path to presentation PDF"
    )

    # ------------------------------------------------------------------
    # Consistency validation
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def validate_consistency(self) -> "ValorizationProvenance":
        """Ensure topology↔revenue↔economics are internally consistent."""
        errors: List[str] = []

        # 1. Optimal topology must exist in topology_results
        optimal = None
        for t in self.topology_results:
            if t.topology_id == self.optimal_topology_id:
                optimal = t
                break
        if optimal is None:
            errors.append(
                f"optimal_topology_id '{self.optimal_topology_id}' not found "
                f"in topology_results"
            )
        else:
            # 2. Products in optimal topology must come from active units
            for p in optimal.products:
                if p.from_unit not in optimal.active_units:
                    errors.append(
                        f"Product '{p.product}' is attributed to unit "
                        f"'{p.from_unit}' which is not in active_units "
                        f"of topology '{optimal.topology_id}'"
                    )

            # 3. Final economics must match optimal topology
            if abs(optimal.revenue_usd_t - self.final_economics.revenue_usd_t) > 1.0:
                errors.append(
                    f"Revenue mismatch: optimal topology has "
                    f"{optimal.revenue_usd_t:.1f}/t but final_economics has "
                    f"{self.final_economics.revenue_usd_t:.1f}/t"
                )
            if abs(optimal.npv_usd - self.final_economics.npv_usd) > 1e6:
                errors.append(
                    f"NPV mismatch: optimal topology has "
                    f"{optimal.npv_usd/1e6:.1f}M but final_economics has "
                    f"{self.final_economics.npv_usd/1e6:.1f}M"
                )

        # 4. Topology ranks must be unique and sequential
        ranks = [t.rank for t in self.topology_results if t.rank > 0]
        if ranks and len(ranks) != len(set(ranks)):
            errors.append("Duplicate ranks found in topology_results")

        if errors:
            raise ValueError(
                "Provenance consistency check failed:\n  - "
                + "\n  - ".join(errors)
            )

        return self

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    def save(self, directory: str | Path) -> Path:
        """Save provenance as JSON to the given directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "provenance.json"
        path.write_text(
            self.model_dump_json(indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ValorizationProvenance":
        """Load provenance from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.model_validate(data)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def get_optimal(self) -> TopologyResult:
        """Return the optimal topology result."""
        for t in self.topology_results:
            if t.topology_id == self.optimal_topology_id:
                return t
        raise ValueError(f"Optimal topology '{self.optimal_topology_id}' not found")

    def summary(self) -> str:
        """Return a human-readable summary."""
        opt = self.get_optimal()
        lines = [
            f"═══ Provenance: {self.raw_material} ═══",
            f"  Timestamp:   {self.analysis_timestamp}",
            f"  Feed:        {self.feed.throughput_tpy:,.0f} t/yr",
            f"  Topologies:  {self.n_topologies_evaluated} evaluated, "
            f"{len(self.topology_results)} recorded",
            f"  Optimal:     {opt.topology_id} — {opt.name}",
            f"    Active:    {', '.join(opt.active_units)}",
            f"    Revenue:   ${opt.revenue_usd_t:.0f}/t "
            f"(${self.final_economics.annual_revenue_usd/1e6:.1f}M/yr)",
            f"    OPEX:      ${opt.opex_usd_t:.0f}/t",
            f"    Net:       ${opt.levelized_net_usd_t:.0f}/t",
            f"    NPV:       ${opt.npv_usd/1e6:.1f}M "
            f"(IRR {opt.irr_pct:.0f}%)",
        ]
        if opt.excluded_units:
            lines.append(f"    Excluded:  {', '.join(opt.excluded_units)}")
            for uid, reason in opt.exclusion_reasons.items():
                lines.append(f"      {uid}: {reason}")
        return "\n".join(lines)
