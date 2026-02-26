import math
import logging
logging.basicConfig(level=logging.INFO)

def test_oxalate_precipitation():
    from sep_agents.properties.ree_databases import REEEquilibriumSolver

    # Initialize the solver with light REE and Carbon for oxalates
    print("Building REE system with Oxalate...")
    solver = REEEquilibriumSolver(preset="light_ree", extra_elements=["Ox"])

    # Define a feed with REE + strong Oxalate concentration
    ree_feed = {
        "La+3": 0.01,
        "Ce+3": 0.01,
        "Nd+3": 0.01,
    }
    
    # We add 0.05 moles of Oxalate base
    other_mol = {"C2O4-2": 0.05, "Na+": 0.10, "Cl-": 0.09}

    print("Running initial speciation to observe precipitation...")
    state = solver.speciate(
        temperature_C=25.0,
        ree_mol=ree_feed,
        other_mol=other_mol
    )

    if state["status"] != "ok":
        print("Equilibrium fail:", state.get("error"))
        return
        
    ph = state["pH"]
    print(f"ok")
    print(f"pH = {ph:.2f}")

    # Print out dominant species
    print("Dominant species:")
    for sp, amt in state["species"].items():
        if amt > 1e-4:
            print(f"{sp}: {amt:.4e} mol")

if __name__ == "__main__":
    test_oxalate_precipitation()
