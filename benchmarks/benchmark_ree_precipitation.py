import json
import yaml
from pathlib import Path

def run_hydroxide_precipitation():
    """
    Benchmark the pH of precipitation for Light REEs (La, Ce, Pr, Nd).
    Literature trend shows heavier REEs are less basic and precipitate at lower pH:
       Nd(OH)3 < Pr(OH)3 < Ce(OH)3 < La(OH)3
    """
    try:
        from sep_agents.properties.ree_databases import REEEquilibriumSolver
        import reaktoro as rkt
    except ImportError as e:
        print(f"Error importing modules: {e}")
        return

    print("Benchmarking REE Hydroxide Precipitation pH...")
    solver = REEEquilibriumSolver(preset="light_ree")
    
    # 0.01 mol of each REE in 1 kg water
    ree_feed = {
        "La+3": 0.01,
        "Ce+3": 0.01,
        "Pr+3": 0.01,
        "Nd+3": 0.01,
    }

    # Titrate with NaOH
    naoh_moles = [0.0, 0.05, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19]
    
    results = []
    
    for naoh in naoh_moles:
        other_mol = {"Na+": naoh, "OH-": naoh}
        
        # We need to compute equilibrium
        state = solver.speciate(
            temperature_C=25.0,
            ree_mol=ree_feed,
            other_mol=other_mol
        )
        
        if state["status"] != "ok":
            continue
            
        ph = state["pH"]
        solids = {}
        for sp, amt in state["species"].items():
            if "(s)" in sp and amt > 1e-6:
                solids[sp] = amt
                
        results.append({
            "NaOH_mol": naoh,
            "pH": round(ph, 3),
            "solids": solids
        })
        
        print(f"NaOH: {naoh:.3f} mol | pH: {ph:.2f} | Solids: {[(k, round(v,4)) for k,v in solids.items()]}")

    return results

if __name__ == "__main__":
    run_hydroxide_precipitation()
