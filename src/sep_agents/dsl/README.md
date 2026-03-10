# Domain Specific Language (`dsl`)

This module defines the YAML-based domain specific language for chemical process superstructure optimization.

**Core files:**

- **`schemas.py`** — Pydantic models (`Stream`, `UnitOp`, `Flowsheet`, `Superstructure`, `DisjunctionDef`)
- **`yaml_loader.py`** — Load components, superstructures, and raw materials from YAML
- **`generate_readme.py`** — Auto-regenerate this README from the YAML libraries

> **Note**: This README is auto-generated. Run `python -m sep_agents.dsl.generate_readme` to update it.

---

## Components

**16 unit operation types** defined in `components/`:

| Type | Description | Default Parameters |
|------|-------------|--------------------|
| `crystallizer` | Cooling or evaporative crystallizer for producing solid p... | `T_C`=25.0, `residence_time_s`=7200.0, `cooling_rate_K_per_s`=0.01, ... |
| `cyclone` | Hydrocyclone classifier. Separates particles by size usin... | `d50c_um`=75.0, `sharpness_alpha`=3.0, `pressure_kPa`=150.0 |
| `equilibrium_reactor` | Gibbs energy minimization reactor using Reaktoro. Compute... | `T_C`=200.0, `p_bar`=100.0, `residence_time_s`=7200.0, ... |
| `flotation_bank` | Froth flotation cell bank. First-order kinetic model with... | `k_s_1ps`=0.5, `R_inf`=0.9, `air_rate_m3m2s`=0.02, ... |
| `heat_exchanger` | Shell-and-tube or plate heat exchanger. Counter-current o... | `U_Wm2K`=500.0, `area_m2`=50.0, `dT_approach_K`=10.0, ... |
| `ion_exchange` | Ion exchange column for selective adsorption/desorption o... | `selectivity_coeff`=2.0, `bed_volume_m3`=5.0, `resin_type`=strong_acid_cation, ... |
| `leach_reactor` | Agitated leach reactor for acid/base dissolution of miner... | `residence_time_s`=7200.0, `T_C`=80.0, `tank_volume_m3`=10.0, ... |
| `lims` | Low-Intensity Magnetic Separator (LIMS). Recovers ferroma... | `magnetic_recovery`=0.85, `field_T`=0.3 |
| `mill` | Ball, rod, or SAG mill for comminution. Grinds feed to ta... | `fineness_factor`=0.5, `E_specific_kWhpt`=15.0, `media_type`=steel |
| `mixer` | Simple stream mixer — combines two or more inlet streams ... |  |
| `precipitator` | Chemical precipitation reactor. Adds a reagent to induce ... | `residence_time_s`=3600.0, `reagent_dosage_gpl`=10.0, `T_C`=50.0, ... |
| `pump` | Centrifugal or positive-displacement pump for pressurizin... | `head_m`=1000.0, `efficiency`=0.75 |
| `separator` | Generic stream splitter — splits an inlet into two outlet... | `recovery`=0.95, `split_fraction`=0.95 |
| `solvent_extraction` | Multi-stage counter-current solvent extraction (SX) casca... | `distribution_coeff`=5.0, `organic_to_aqueous_ratio`=1.0, `stages`=4, ... |
| `stoichiometric_reactor` | Fixed-conversion reactor using user-defined stoichiometri... | `T_C`=200.0, `p_bar`=50.0, `residence_time_s`=7200.0, ... |
| `thickener` | Gravity thickener for solid-liquid separation. Produces a... | `recovery`=0.9, `underflow_solids_frac`=0.6, `overflow_clarity`=0.99 |

---

## Superstructures

**5 GDP superstructures** defined in `superstructures/`:

### `eaf_steel_slag`

> EAF steel slag valorization for iron recovery (LIMS), H₂ production (serpentinization of fayalite), CO₂ sequestration (mineral carbonation of CaO/MgO), and optional Cr/Mn/V metal recovery via acid leaching. GDP choices: LIMS position, heat strategy, Cr/Mn recovery, V recovery. Throughput: 100,000 t/yr.

- **Objective**: `minimize_opex`
- **Units**: 27 — **Streams**: 34 — **Disjunctions**: 6 — **Continuous bounds**: 18

**Unit operations:**

- `mixer_water`
- `pump_slag`
- `pump_co2`
- `lims_pre_serp` *(optional)*
- `no_lims_pre` *(optional)*
- `hx_serp_preheat`
- `aux_heater` *(optional)*
- `no_aux_heater` *(optional)*
- `reactor_serpentinization` *(optional)*
- `bypass_serp` *(optional)*
- `hx_serp_recovery`
- `separator_h2`
- `mixer_co2`
- `hx_carb_preheat`
- `reactor_carbonation` *(optional)*
- `bypass_carb` *(optional)*
- `hx_carb_recovery`
- `separator_carbonate`
- `mixer_waste`
- `leach_cr_mn` *(optional)*
- `skip_cr_mn` *(optional)*
- `precip_cr` *(optional)*
- `precip_mn` *(optional)*
- `leach_v` *(optional)*
- `skip_v` *(optional)*
- `precip_v` *(optional)*
- `mixer_metal_waste`

**GDP disjunctions:**

| Disjunction | Choices | Description |
|-------------|---------|-------------|
| `lims_position` | `lims_pre_serp`, `no_lims_pre` | Magnetic separation before serpentinization vs. bypass (Fe₃O₄ formed in-situ by serpentinization)
 |
| `heat_strategy` | `aux_heater`, `no_aux_heater` | Supplemental heater vs. full heat recovery only |
| `h2_production` | `reactor_serpentinization`, `bypass_serp` | Serpentinization reactor (produces H₂ + Fe₃O₄) vs. bypass. GDP decides if H₂ revenue justifies reactor + utility cost.
 |
| `co2_mineralization` | `reactor_carbonation`, `bypass_carb` | Carbonation reactor (sequesters CO₂ as CaCO₃/MgCO₃) vs. bypass. GDP decides if CO₂ credit revenue justifies reactor + utility cost.
 |
| `cr_mn_recovery` | `leach_cr_mn`, `skip_cr_mn` | Acid leach for Cr₂O₃/MnO recovery vs. bypass. GDP decides if Cr/Mn revenue justifies leach + precip cost.
 |
| `v_recovery` | `leach_v`, `skip_v` | V₂O₃ acid leach + precipitation vs. bypass. GDP decides if vanadium revenue justifies the circuit.
 |

**Continuous design variables (BoTorch bounds):**

| Variable | Min | Max |
|----------|----:|----:|
| `mixer_water.water_kg` | 200.0 | 2000.0 |
| `reactor_serpentinization.T_C` | 200.0 | 300.0 |
| `reactor_serpentinization.p_bar` | 50.0 | 300.0 |
| `reactor_carbonation.T_C` | 100.0 | 200.0 |
| `reactor_carbonation.p_bar` | 10.0 | 150.0 |
| `hx_serp_preheat.area_m2` | 10.0 | 100.0 |
| `hx_carb_preheat.area_m2` | 5.0 | 60.0 |
| `hx_serp_recovery.area_m2` | 10.0 | 100.0 |
| `hx_carb_recovery.area_m2` | 5.0 | 60.0 |
| `pump_slag.head_m` | 500.0 | 3000.0 |
| `pump_co2.head_m` | 100.0 | 1500.0 |
| `leach_cr_mn.T_C` | 50.0 | 95.0 |
| `leach_cr_mn.residence_time_s` | 3600.0 | 14400.0 |
| `precip_cr.reagent_dosage_gpl` | 5.0 | 25.0 |
| `precip_mn.reagent_dosage_gpl` | 5.0 | 30.0 |
| `leach_v.T_C` | 60.0 | 95.0 |
| `leach_v.residence_time_s` | 3600.0 | 21600.0 |
| `precip_v.reagent_dosage_gpl` | 10.0 | 40.0 |

---

### `lree_acid_leach`

> LREE recovery from HCl acid-leach liquor. Choose between D2EHPA and PC88A extractants, optional scrubbing stage, and oxalate vs hydroxide product precipitation.

- **Objective**: `minimize_opex`
- **Units**: 5 — **Streams**: 13 — **Disjunctions**: 2 — **Continuous bounds**: 4

**Unit operations:**

- `sx_d2ehpa` *(optional)*
- `sx_pc88a` *(optional)*
- `scrubber` *(optional)*
- `oxalate_precip` *(optional)*
- `hydroxide_precip` *(optional)*

**GDP disjunctions:**

| Disjunction | Choices | Description |
|-------------|---------|-------------|
| `separation_method` | `sx_d2ehpa`, `sx_pc88a` | Choose SX extractant: D2EHPA vs PC88A |
| `product_form` | `oxalate_precip`, `hydroxide_precip` | Product precipitation: oxalate vs hydroxide |

**Continuous design variables (BoTorch bounds):**

| Variable | Min | Max |
|----------|----:|----:|
| `sx_d2ehpa.organic_to_aqueous_ratio` | 0.5 | 3.0 |
| `sx_pc88a.organic_to_aqueous_ratio` | 0.5 | 3.0 |
| `oxalate_precip.reagent_dosage_gpl` | 5.0 | 30.0 |
| `hydroxide_precip.reagent_dosage_gpl` | 3.0 | 20.0 |

---

### `olivine_carbonation_h2`

> Olivine valorization for CO₂ sequestration (direct carbonation of forsterite → MgCO₃) and H₂ production (serpentinization of fayalite → Fe₃O₄ + H₂). Includes heat recovery, pressurization, and product separation. GDP choices: auxiliary heating strategy + optional magnetic separation for Fe₃O₄ recovery.

- **Objective**: `minimize_opex`
- **Units**: 16 — **Streams**: 20 — **Disjunctions**: 2 — **Continuous bounds**: 11

**Unit operations:**

- `mixer_water`
- `pump_olivine`
- `pump_co2`
- `hx_serp_preheat`
- `aux_heater`
- `no_aux_heater`
- `reactor_serpentinization`
- `hx_serp_recovery`
- `separator_h2`
- `mixer_co2`
- `hx_carb_preheat`
- `reactor_carbonation`
- `hx_carb_recovery`
- `separator_carbonate`
- `lims_fe3o4`
- `no_mag_sep`

**GDP disjunctions:**

| Disjunction | Choices | Description |
|-------------|---------|-------------|
| `heat_strategy` | `aux_heater`, `no_aux_heater` | Supplemental heating vs full heat recovery only |
| `mag_separation` | `lims_fe3o4`, `no_mag_sep` | Magnetic separation of Fe₃O₄ vs no separation |

**Continuous design variables (BoTorch bounds):**

| Variable | Min | Max |
|----------|----:|----:|
| `mixer_water.water_kg` | 300.0 | 3000.0 |
| `reactor_serpentinization.T_C` | 200.0 | 300.0 |
| `reactor_serpentinization.p_bar` | 50.0 | 300.0 |
| `reactor_carbonation.T_C` | 150.0 | 200.0 |
| `reactor_carbonation.p_bar` | 50.0 | 200.0 |
| `hx_serp_preheat.area_m2` | 10.0 | 100.0 |
| `hx_carb_preheat.area_m2` | 5.0 | 60.0 |
| `hx_serp_recovery.area_m2` | 10.0 | 100.0 |
| `hx_carb_recovery.area_m2` | 10.0 | 120.0 |
| `pump_olivine.head_m` | 500.0 | 3000.0 |
| `pump_co2.head_m` | 500.0 | 2000.0 |

---

### `simple_sx_precip`

> Minimal SX → Precipitator with optional scrubber. 2 configurations for quick GDP validation.

- **Objective**: `minimize_opex`
- **Units**: 3 — **Streams**: 8 — **Disjunctions**: 0 — **Continuous bounds**: 2

**Unit operations:**

- `sx_1`
- `scrubber` *(optional)*
- `precipitator`

**Continuous design variables (BoTorch bounds):**

| Variable | Min | Max |
|----------|----:|----:|
| `sx_1.organic_to_aqueous_ratio` | 0.5 | 3.0 |
| `precipitator.reagent_dosage_gpl` | 5.0 | 25.0 |

---

### `steel_slag_h2_co2`

> Steel slag valorization for H₂ production (serpentinization of fayalite) and CO₂ sequestration (mineral carbonation of CaO/MgO). Includes heat recovery network, pressurization pumps, and product separators. GDP choices: supplemental heating vs full heat recovery; optional inter-reactor scrubbing.

- **Objective**: `minimize_opex`
- **Units**: 16 — **Streams**: 20 — **Disjunctions**: 1 — **Continuous bounds**: 13

**Unit operations:**

- `mixer_water`
- `pump_slag`
- `pump_co2`
- `hx_serp_preheat`
- `hx_carb_preheat`
- `reactor_serpentinization`
- `reactor_carbonation`
- `hx_serp_recovery`
- `hx_carb_recovery`
- `aux_heater_serp` *(optional)*
- `no_aux_heater` *(optional)*
- `separator_h2`
- `separator_carbonate`
- `scrubber` *(optional)*
- `mixer_co2`
- `mixer_waste`

**GDP disjunctions:**

| Disjunction | Choices | Description |
|-------------|---------|-------------|
| `heat_strategy` | `aux_heater_serp`, `no_aux_heater` | Supplemental heating: auxiliary heater vs full heat recovery only |

**Continuous design variables (BoTorch bounds):**

| Variable | Min | Max |
|----------|----:|----:|
| `mixer_water.water_kg` | 200.0 | 2000.0 |
| `reactor_serpentinization.T_C` | 200.0 | 300.0 |
| `reactor_serpentinization.p_bar` | 50.0 | 300.0 |
| `reactor_serpentinization.residence_time_s` | 3600.0 | 86400.0 |
| `reactor_carbonation.T_C` | 100.0 | 200.0 |
| `reactor_carbonation.p_bar` | 10.0 | 100.0 |
| `reactor_carbonation.residence_time_s` | 1800.0 | 28800.0 |
| `hx_serp_preheat.area_m2` | 10.0 | 100.0 |
| `hx_carb_preheat.area_m2` | 5.0 | 60.0 |
| `hx_serp_recovery.area_m2` | 10.0 | 100.0 |
| `hx_carb_recovery.area_m2` | 5.0 | 60.0 |
| `pump_slag.head_m` | 500.0 | 3000.0 |
| `pump_co2.head_m` | 100.0 | 1000.0 |

---

## Raw Materials

**2 raw materials** defined in `raw_materials/`:

| Name | Physical Form | Value Streams | Throughput |
|------|--------------|---------------|-----------|
| `eaf_steel_slag` | solid | iron_recovery, chromium_recovery, manganese_recovery, vanadium_recovery, h2_production, co2_mineralization | 303 t/d |
| `olivine` | solid | h2_production, co2_mineralization, iron_recovery | 303 t/d |
