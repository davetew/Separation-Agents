---
description: Characterize a raw material feed, compute equilibrium speciation, and identify value streams for valorization
---

# Resource Characterization Workflow

// turbo-all

> **Environment**: All Python commands must use `conda run --no-capture-output -n rkt python3`. See `/valorize` for details.

When the user provides a raw material (mine tailings, ore, slag, brine, industrial waste, etc.) for valorization analysis, follow these steps to produce a standardized feed characterization.

---

## Step 1: Gather Composition Data

### 1a. Check the Raw Materials Library

Before gathering new data, check whether the material already exists in the library:

```python
from sep_agents.dsl.yaml_loader import list_raw_materials, load_raw_material

# List available materials
for m in list_raw_materials():
    print(f"  {m['name']}: {m['description'][:80]}")

# If the material exists, load and reuse it
mat = load_raw_material("eaf_steel_slag")  # returns dict with composition, conditions, etc.
```

If the material is found, present it to the user for confirmation and skip to Step 3 (value streams). If the user wants to update the composition, load the existing data as a starting point.

### 1b. Collect New Composition

If the material is not in the library, collect or infer the raw-material composition from the user. Required information:

- **Mineralogy** (mineral phases and approximate abundances, e.g., mol% or wt%)
- **Metal assay** (grade of valuable metals: Ni, Cu, Co, REE, PGE, etc.)
- **Bulk chemistry** (major oxides: MgO, SiO₂, Fe₂O₃, CaO, Al₂O₃)
- **Physical form**: solid (rock/tailings/slag), slurry, or aqueous brine
- **Throughput**: tonnes per day or per year

If the user does not provide all fields, use reasonable engineering defaults and state assumptions clearly.

Convert the mineralogy to an **aqueous-species basis** suitable for Reaktoro. Map each mineral to its dissolved species (e.g., Forsterite → Mg²⁺ + SiO₂(aq)). Use charge-balanced Cl⁻ or SO₄²⁻ to close the charge balance if needed.

---

## Step 2: Equilibrium Speciation (Conditional)

> **When to run**: Only if the value streams identified in Step 3 include **hydrometallurgical routes** (leaching, solvent extraction, ion exchange, precipitation) where aqueous chemistry (pH, Eh, complexation) determines process feasibility and reagent requirements.
>
> **When to skip**: If the process involves only **stoichiometric reactor paths** (serpentinization, carbonation, pyrometallurgy) or physical separation (magnetic, gravity, flotation), the mineralogical composition from Step 1 is sufficient — proceed directly to Step 3.

If speciation is needed, use the `speciate_ree_stream` or `run_speciation` MCP tool:
```
speciate_ree_stream(
    temperature_C=<feed_temp>,
    pressure_atm=1.0,
    water_kg=1.0,
    ree={...},       # if REE present
    other={...},     # non-REE species
    preset="light_ree" or appropriate database
)
```

Record:
- **pH** and **Eh** (redox potential)
- **Dominant aqueous species** and their concentrations
- **Mineral saturation indices** (identify minerals that could precipitate)

---

## Step 3: Identify Value Streams

Based on the composition and speciation results, categorize the raw material into value streams. Consider **all** of the following pathways:

| Value Stream | Trigger Condition | Product(s) |
|---|---|---|
| **Base metal recovery** | Ni, Cu, Co > 0.1 wt% | Metal cathodes, sulfate salts |
| **REE recovery** | Total REO > 0.05 wt% | Mixed REE oxide, individual REO |
| **PGE recovery** | Pt, Pd > 0.5 g/t | PGE concentrate |
| **Iron concentrate** | Magnetite or hematite present | Fe₃O₄ / Fe₂O₃ concentrate |
| **CO₂ mineralization** | Mg-silicates (forsterite, serpentine) > 20 mol% | MgCO₃ + carbon credits |
| **H₂ production** | Fe²⁺-bearing silicates (fayalite) | H₂ gas (serpentinization) |
| **H₂SO₄ production** | Sulfide minerals (pyrrhotite, pyrite) | Acid for self-supply |
| **Construction aggregate** | Residual silicates/aluminates | Controlled fill, aggregate |

---

## Step 4: Commodity Price Lookup

For each identified value stream, assign a current market price. Use the following defaults if the user does not provide specific prices:

| Product | Default Price | Source |
|---|---|---|
| Cu cathode | $8,500/t | LME |
| NiSO₄ equiv | $16,000/t | LME |
| Co metal | $33,000/t | LME |
| Nd₂O₃ | $150/kg | Asian Metal |
| Pr₆O₁₁ | $100/kg | Asian Metal |
| Dy₂O₃ | $350/kg | Asian Metal |
| Magnetite conc | $120/t | FOB |
| CO₂ credit | $60/t | Voluntary market |
| H₂ (green) | $4/kg | DOE target |
| H₂SO₄ | $80/t | Spot |

Update prices if the user provides more recent data or specifies a different market.

---

## Step 5: Produce Feed Characterization Output

Generate two outputs:

### a) `feed_characterization.yaml`
A YAML file in the Separation-Agents DSL format containing:
- Feed stream(s) with `composition_wt`, `temperature_K`, `pressure_Pa`, `pH`, `phase`
- Auxiliary streams (acid, CO₂, water) if relevant to the identified value streams

Use the same format as `examples/eagle_creek_valorization.yaml` for consistency.

### b) Characterization Summary (Markdown)
A concise Markdown summary including:
- Raw material description and source
- Composition table (mineralogy + assay)
- Speciation results (pH, Eh, key species)
- Identified value streams with estimated revenue potential ($/t feedstock)
- Recommended next step: invoke `/superstructure-selection`

---

## Handoff

After completing this workflow, recommend that the user proceed with:
```
/superstructure-selection
```
passing the `feed_characterization.yaml` and value stream summary as context.

---

## Step 6: Persist to Raw Materials Library

After the user confirms the characterization (or at the end of the workflow), save the raw material to the YAML library so it is available for future analyses:

```python
from sep_agents.dsl.yaml_loader import save_raw_material

material_data = {
    "name": "<material_name>",
    "description": "<description>",
    "source": "<source>",
    "throughput_tpd": <throughput>,
    "physical_form": "<solid|slurry|brine|liquid>",
    "composition": {
        "basis": "<mol|wt_pct|wt_kg>",
        "minerals": {"<Species>": <amount>, ...},
        "water_kg": <water>,
    },
    "conditions": {
        "temperature_K": <T>,
        "pressure_Pa": <P>,
    },
    "value_streams": ["<stream1>", "<stream2>", ...],
    "commodity_prices": {"<product>_usd_<unit>": <price>, ...},
}

save_raw_material(material_data, overwrite=True)
```

This ensures every raw material evaluated through the tool is automatically catalogued for reuse.
