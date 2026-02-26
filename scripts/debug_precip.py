import math
import logging
logging.basicConfig(level=logging.DEBUG)

def test_precipitation():
    import reaktoro as rkt
    from sep_agents.properties.ree_databases import REEEquilibriumSolver

    # Initialize the solver with light REE
    print("Building REE system...")
    solver = REEEquilibriumSolver(preset="light_ree")
    
    ree_feed = {"Nd+3": 0.01}
    other = {"Na+": 0.15, "OH-": 0.15}
    
    print("Running speciation...")
    state = solver.speciate(temperature_C=25.0, ree_mol=ree_feed, other_mol=other)
    
    print(state["status"])
    if state["status"] == "ok":
        print(f"pH = {state['pH']:.2f}")
        for sp, amt in state["species"].items():
            if amt > 1e-6:
                print(f"{sp}: {amt:.4e} mol")
    else:
        print("Equilibrium failed.")

if __name__ == "__main__":
    test_precipitation()
