import logging
from typing import Dict, Any

_log = logging.getLogger(__name__)

# Standard proxy prices for REE flowsheets in USD/kg
REAGENT_PRICES_USD_PER_KG = {
    # Acids
    "HCl(aq)": 0.20,
    "H2SO4(aq)": 0.15,
    "HNO3(aq)": 0.35,
    # Bases
    "NaOH(aq)": 0.50,
    "NH3(aq)": 0.45,
    "Na2CO3(s)": 0.25,
    # Organics / Ligands
    "C2O4-2": 1.20,       # Proxy for Oxalic Acid
    "H2C2O4(aq)": 1.20,
    "HC2O4-": 1.20,
    # Extractants
    "D2EHPA": 8.50,
    "PC88A": 12.00,
    "Cyanex272": 25.00,
    # Energy
    "electricity_kWh": 0.08,
}

# Proxy molar masses (g/mol) to convert StreamState moles -> kg
MOLAR_MASS_G_PER_MOL = {
    "HCl(aq)": 36.46,
    "H2SO4(aq)": 98.08,
    "HNO3(aq)": 63.01,
    "NaOH(aq)": 40.00,
    "NH3(aq)": 17.03,
    "Na2CO3(s)": 105.99,
    "C2O4-2": 88.02,
    "H2C2O4(aq)": 90.03,
    "HC2O4-": 89.03,
}

def estimate_opex_usd(flowsheet: Any, states: Dict[str, Any]) -> float:
    """Evaluate proxy OPEX by integrating incoming reagents and unit power requirements.
    
    Returns
    -------
    Total cost in USD (for the basis timeframe/batch size given by the flowsheet streams).
    """
    total_cost_usd = 0.0
    
    # 1. Total Reagent Consumption
    # Determine all feed streams (inputs not produced by any unit)
    produced = {s for u in flowsheet.units for s in u.outputs}
    feeds = [s for s in flowsheet.streams if s.name not in produced]
    
    for feed in feeds:
        if feed.name not in states:
            continue
        state = states[feed.name]
        
        # Scrape species amounts to cost reagents
        for sp, amt_mol in state.species_amounts.items():
            cost_per_kg = REAGENT_PRICES_USD_PER_KG.get(sp, 0.0)
            if cost_per_kg > 0:
                mm_g_mol = MOLAR_MASS_G_PER_MOL.get(sp, 100.0)
                mass_kg = (amt_mol * mm_g_mol) / 1000.0
                total_cost_usd += mass_kg * cost_per_kg

        # Provide a heuristic for bulk water transport/pumping if no specific species matched
        # Assumption: Aqueous solutions cost baseline pumping energy ($0.05 / 1000L).
        if "H2O(aq)" in state.species_amounts or "H2O" in state.species_amounts:
            amt_mol_water = state.species_amounts.get("H2O(aq)", state.species_amounts.get("H2O", 0))
            vol_liters = (amt_mol_water * 18.015) / 1000.0 
            total_cost_usd += (vol_liters / 1000.0) * 0.05 # USD per cubic meter pumping proxy

    # 2. Unit Operational Energy Consumptions
    for unit in flowsheet.units:
        # Example Proxy: Crystallizer Mixing and Cooling Energy
        if unit.type in ["crystallizer", "precipitator"]:
            res_time = unit.params.get("residence_time_s", 3600.0)
            reagent_dos = unit.params.get("reagent_dosage_gpl", 10.0)
            # Power scales with residence time and mixing requirement from dosage
            kwh = (res_time / 3600.0) * 5.0 + (reagent_dos * 0.1)
            total_cost_usd += kwh * REAGENT_PRICES_USD_PER_KG["electricity_kWh"]
        elif unit.type == "solvent_extraction":
            # Simplified proxy: Power scales with stages
            stages = unit.params.get("stages", 1)
            kwh = 2.0 * stages
            total_cost_usd += kwh * REAGENT_PRICES_USD_PER_KG["electricity_kWh"]
            
    return round(total_cost_usd, 2)
