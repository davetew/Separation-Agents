import yaml
import json
from mcp_server.server import optimize_flowsheet
from sep_agents.dsl.schemas import Flowsheet, Stream, UnitOp

def main():
    print("Building a test flowsheet for optimization...")
    feed = Stream(
        name="feed_liquor",
        phase="liquid",
        composition_wt={
            "H2O(aq)": 1000.0,
            "La+3": 5.0,
            "C2O4-2": 15.0,
            "Na+": 30.0,
            "Cl-": 30.0
        }
    )

    precip_unit = UnitOp(
        id="oxalate_precip",
        type="precipitator",
        inputs=["feed_liquor"],
        outputs=["oxalate_slurry"],
        params={
            "T_C": 25.0,
            "residence_time_s": 3600.0,
            "reagent_dosage_gpl": 10.0 # <--- The variable we want to optimize (minimize OPEX)
        }
    )

    flowsheet = Flowsheet(name="Optimization_Test", streams=[feed], units=[precip_unit])
    fs_yaml = yaml.dump(flowsheet.dict(), sort_keys=False)

    print("Running Bayesian Optimization (Minimizing OPEX by varying reagent_dosage_gpl [0.1, 20.0])...")
    
    # Run server function
    result = optimize_flowsheet(
        flowsheet_yaml=fs_yaml,
        design_variables=[
            {
                "unit_id": "oxalate_precip",
                "param": "reagent_dosage_gpl",
                "bounds": [0.1, 20.0]
            }
        ],
        objective_kpi="overall.opex_USD",
        maximize=False, # Minimize Cost
        n_iters=5, # Short test
        database="light_ree"
    )

    if result.get("status") == "ok":
        print("\n=== Optimization Successful ===")
        print(f"Minimum Found OPEX (USD): {result['best_kpi']}")
        print(f"Optimal Parameters: {result['optimal_parameters']}")
        print("\n=== Best Flowsheet ===")
        print(result["flowsheet_yaml"])
    else:
        print("\n=== Optimization Failed ===")
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
