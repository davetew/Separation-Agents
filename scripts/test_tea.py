import json
from sep_agents.dsl.schemas import Flowsheet, Stream, UnitOp
from sep_agents.sim.idaes_adapter import run_idaes

def main():
    print("Building a test flowsheet with REE and Oxalate inputs...")
    
    # 1. Define Feed Streams
    feed = Stream(
        name="feed_liquor",
        phase="liquid",
        temperature_K=298.15,
        composition_wt={
            "H2O(aq)": 1000.0,
            "La+3": 5.0,  # moles
            "Nd+3": 5.0, 
            "C2O4-2": 15.0, # Target 15 moles Oxalate to precipitate
            "Na+": 30.0,
            "Cl-": 30.0
        }
    )

    # 2. Define Unit Operations
    precip_unit = UnitOp(
        id="oxalate_precip",
        type="precipitator",
        inputs=["feed_liquor"],
        outputs=["oxalate_slurry"],
        params={
            "T_C": 25.0,
            "residence_time_s": 3600.0,
            "reagent_dosage_gpl": 10.0
        }
    )

    # 3. Assemble Flowsheet Structure
    flowsheet = Flowsheet(
        name="TEA_LCA_Validation",
        streams=[feed],
        units=[precip_unit]
    )

    print("Running IDAES Sequential Solver with light_ree database (incorporating the Ox pseudo-element)...")
    result = run_idaes(flowsheet, database="light_ree")

    if result["status"] == "ok":
        print("\n=== KPI Results (including TEA/LCA) ===")
        print(json.dumps(result["kpis"], indent=2))
    else:
        print("\nFlowsheet Failed:")
        print(result.get("error"))

if __name__ == "__main__":
    main()
