import reaktoro as rkt
import math

def build_custom_ree_db():
    print("Loading base SUPRCRT database...")
    db = rkt.SupcrtDatabase("supcrtbl")
    
    # Solubility Products (pKsp) for REE hydroxides:
    # REE+3 + 3OH- = REE(OH)3(s)
    # Ksp = 10^(-pKsp)
    # The formation reaction defined with reactants: REE+3 + 3OH- -> REE(OH)3(s)
    # K_f = 1/Ksp => log10(K_f) = pKsp
    hydroxides = [
        ("La(OH)3(s)", "La+3", 20.7),
        ("Ce(OH)3(s)", "Ce+3", 19.7),
        ("Pr(OH)3(s)", "Pr+3", 23.47),
        ("Nd(OH)3(s)", "Nd+3", 21.49),
        ("Sm(OH)3(s)", "Sm+3", 22.08)
    ]
    
    R = 8.31446261815324
    T = 298.15
    P = 100000.0
    
    # Pre-fetch OH- properties
    vol_params = rkt.StandardVolumeModelParamsConstant()
    vol_params.V0 = 5.0e-5 # approx 50 cm3/mol
        
    for sp_name, cation_name, pKsp in hydroxides:
        # Calculate Delta G Reaction
        # dG_rxn = -RT ln(Kf) = -RT * pKsp * ln(10)
        dG_rxn = -R * T * pKsp * math.log(10)
        
        cation = db.species().get(cation_name)
        OH = db.species().get("OH-")
        
        G0_cation = cation.standardThermoProps(T, P).G0
        G0_OH = OH.standardThermoProps(T, P).G0
        
        # dG_rxn = G0_product - (G0_cation + 3*G0_OH)
        G0_product = dG_rxn + G0_cation + 3.0 * G0_OH
        
        # 1. Element composition
        elem_comp = [(cation.elements().symbols()[0], 1.0), ("O", 3.0), ("H", 3.0)]
        
        # 2. Define standard thermo model based on calculated G0
        model = rkt.StandardThermoModelParamsConstant()
        model.G0 = float(G0_product)
        # We must supply a minimal constant volume model param as well
        model.V0 = vol_params.V0
        thermo_model = rkt.StandardThermoModelConstant(model)
        
        # 3. Define species
        sp = rkt.Species().withName(sp_name) \
            .withElements(rkt.ElementalComposition(elem_comp)) \
            .withAggregateState(rkt.AggregateState.CrystallineSolid) \
            .withStandardThermoModel(thermo_model)
            
        try:
            db.addSpecies(sp)
            print(f"Added {sp_name} (pKsp={pKsp}) with G0 = {G0_product:.2f} J/mol")
        except Exception as e:
            print(f"Failed to add {sp_name}: {e}")
            
    return db

if __name__ == "__main__":
    db = build_custom_ree_db()
    system = rkt.ChemicalSystem(db, rkt.AqueousPhase("H O Na Cl La Ce Pr Nd Sm"), rkt.MineralPhase("La(OH)3(s)"))
    print("System built successfully with custom species!")
