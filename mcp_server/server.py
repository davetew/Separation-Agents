# mcp_server/server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("MineralSeparationMCP")

# Tools
@mcp.tool()
def simulate_flowsheet(flowsheet_yaml: str) -> dict:
    from sep_agents.orchestrator.planner import load_flowsheet, Orchestrator
    import tempfile, pathlib
    p = pathlib.Path(tempfile.mkstemp(suffix=".yaml")[1])
    p.write_text(flowsheet_yaml)
    fs = load_flowsheet(str(p))
    return Orchestrator().run_once(fs)

@mcp.tool()
def run_speciation(stream: dict) -> dict:
    try:
        from sep_agents.sim.reaktoro_adapter import run_reaktoro
        from sep_agents.dsl.schemas import Stream
    except ImportError:
         return {"status": "error", "error": "Internal module import failed"}

    try:
        s = Stream(**stream)
        s_out = run_reaktoro(s)
        return {"status": "ok", "stream_out": s_out.dict()}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def perform_sweep(initial_conditions: dict, param_name: str, values: list) -> dict:
    try:
        from sep_agents.sim.equilibrium_agent import EquilibriumAgent
        agent = EquilibriumAgent()
        df = agent.sweep(initial_conditions, param_name, values)
        return {"status": "ok", "results": df.to_dict(orient="records")}
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def estimate_cost(kpis: dict) -> dict:
    from sep_agents.cost.tea import estimate_opex_kwh_reagents
    from sep_agents.cost.lca import estimate_co2e
    return {"status": "ok", "OPEX": estimate_opex_kwh_reagents(kpis), "CO2e": estimate_co2e(kpis)}

@mcp.tool()
def optimize_flowsheet(
    flowsheet_yaml: str,
    design_variables: list[dict],
    objective_kpi: str = "overall.opex_USD",
    maximize: bool = False,
    n_iters: int = 15,
    database: str = "light_ree"
) -> dict:
    """Optimize flowsheet parameters using BoTorch Bayesian Optimization.

    Iteratively adjusts unit parameters to maximize or minimize a specific KPI.
    
    Args:
        flowsheet_yaml: Base YAML string of the flowsheet.
        design_variables: List of dicts specifying variables to change:
            [{"unit_id": "sx_1", "param": "organic_to_aqueous_ratio", "bounds": [0.5, 3.0]}]
        objective_kpi: The KPI to optimize (e.g. 'overall.opex_USD' or 'overall.recovery').
        maximize: True to maximize the KPI, False to minimize (e.g. for OPEX).
        n_iters: Budget for optimization iterations.
        database: Reaktoro database preset to use.
    """
    import yaml
    import torch
    from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
    from sep_agents.sim.idaes_adapter import run_idaes
    from sep_agents.opt.bo import BotorchOptimizer
    import copy

    # Ensure bounds are well formatted for torch
    try:
        bounds_list = [v["bounds"] for v in design_variables]
        bounds_tensor = torch.tensor(bounds_list, dtype=torch.double).T # 2 x d
    except Exception as e:
        return {"status": "error", "error": f"Invalid bounds format: {e}"}

    data = yaml.safe_load(flowsheet_yaml)

    # Dynamic objective evaluator
    def flowsheet_objective(candidate_x: torch.Tensor) -> float:
        # Clone the base dictionaries to prevent cross-contamination
        current_data = copy.deepcopy(data)
        
        units = [UnitOp(**u) for u in current_data.get("units", [])]
        streams = [Stream(**s) for s in current_data.get("streams", [])]
        
        # Overlay the candidate parameters
        for i, var in enumerate(design_variables):
            target_unit_id = var["unit_id"]
            param_key = var["param"]
            val = candidate_x[i].item()
            
            # Find the unit and update
            for u in units:
                if u.id == target_unit_id:
                    if u.params is None:
                        u.params = {}
                    u.params[param_key] = val
                    break
                    
        # Reconstruct Flowsheet object
        fs = Flowsheet(name=current_data.get("name", "optimized_flowsheet"), units=units, streams=streams)
        
        # Run Evaluation
        result = run_idaes(fs, database=database)
        if result["status"] == "ok" and objective_kpi in result["kpis"]:
            return float(result["kpis"][objective_kpi])
            
        # Error penalty mapping (Worst case scenarios)
        return -1e9 if maximize else 1e9

    try:
        opt = BotorchOptimizer(maximize=maximize)
        best_x, best_y, history = opt.optimize(
            objective_fn=flowsheet_objective,
            bounds=bounds_tensor,
            n_initial=5,
            n_iters=n_iters
        )
        
        # Apply the final optimal parameters to return the optimized dict
        best_x_list = best_x.tolist()
        final_data = copy.deepcopy(data)
        units = [UnitOp(**u) for u in final_data.get("units", [])]
        for i, var in enumerate(design_variables):
            for u in units:
                if u.id == var["unit_id"]:
                    if u.params is None: u.params = {}
                    u.params[var["param"]] = best_x_list[i]
                    break
        final_fs = Flowsheet(name="Optimized_Flowsheet", units=units, streams=[Stream(**s) for s in final_data.get("streams", [])])
        
        return {
            "status": "ok",
            "best_kpi": best_y,
            "optimal_parameters": best_x_list,
            "flowsheet_yaml": yaml.dump(final_fs.dict(), sort_keys=False),
            "history": history
        }
    except Exception as e:
        return {"status": "error", "error": f"Optimization failed: {str(e)}"}

@mcp.tool()
def run_idaes_flowsheet(flowsheet_yaml: str, database: str = "SUPRCRT - BL") -> dict:
    """Build and solve an IDAES separation flowsheet with Reaktoro thermodynamics.

    Accepts a YAML string defining a separation process flowsheet with units
    (leach_reactor, precipitator, cyclone, lims, flotation_bank, mixer,
    solvent_extraction, thickener, mill) and streams.

    Returns per-stream thermodynamic states (T, P, composition, pH) and
    per-unit KPIs (recovery fractions).

    Args:
        flowsheet_yaml: YAML string defining the flowsheet (name, streams, units).
        database: Reaktoro thermodynamic database (default: "SUPRCRT - BL").
    """
    import tempfile, pathlib, yaml
    from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
    from sep_agents.sim.idaes_adapter import run_idaes

    try:
        data = yaml.safe_load(flowsheet_yaml)
        units = [UnitOp(**u) for u in data.get("units", [])]
        streams = [Stream(**s) for s in data.get("streams", [])]
        fs = Flowsheet(name=data.get("name", "flowsheet"), units=units, streams=streams)
        return run_idaes(fs, database=database)
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def get_stream_analysis(stream_state: dict) -> dict:
    """Analyse a solved stream state: list species by abundance, report pH/Eh.

    Args:
        stream_state: A stream dict returned by run_idaes_flowsheet
                      (with keys: temperature_K, pressure_Pa, species_amounts, …).
    """
    species = stream_state.get("species_amounts", {})
    total = sum(species.values())
    ranked = sorted(species.items(), key=lambda kv: kv[1], reverse=True)
    return {
        "status": "ok",
        "temperature_K": stream_state.get("temperature_K"),
        "pressure_Pa": stream_state.get("pressure_Pa"),
        "pH": stream_state.get("pH"),
        "Eh_mV": stream_state.get("Eh_mV"),
        "total_mol": round(total, 6),
        "species_ranked": [
            {"species": sp, "mol": round(amt, 8), "mol_frac": round(amt / total, 6) if total > 0 else 0}
            for sp, amt in ranked[:20]  # Top 20 species
        ],
    }

@mcp.tool()
def speciate_ree_stream(
    temperature_C: float = 25.0,
    pressure_atm: float = 1.0,
    water_kg: float = 1.0,
    acid: dict = None,
    ree: dict = None,
    other: dict = None,
    preset: str = "light_ree",
) -> dict:
    """Compute REE speciation at equilibrium using Reaktoro.

    Calculates the equilibrium distribution of rare earth element species
    in an aqueous solution. Returns pH, Eh, and the full REE speciation.

    Args:
        temperature_C: Temperature in °C (default 25).
        pressure_atm: Pressure in atm (default 1).
        water_kg: Mass of water in kg (default 1).
        acid: Acid species and molar amounts, e.g. {"HCl(aq)": 0.5}.
        ree: REE species and molar amounts, e.g. {"Ce+3": 0.01, "Nd+3": 0.005}.
        other: Other dissolved species, e.g. {"NaCl(aq)": 0.1, "FeCl3(aq)": 0.02}.
        preset: REE element set — "light_ree", "heavy_ree", or "full_ree".
    """
    try:
        from sep_agents.properties.ree_databases import REEEquilibriumSolver
        solver = REEEquilibriumSolver(preset=preset)
        result = solver.speciate(
            temperature_C=temperature_C,
            pressure_atm=pressure_atm,
            water_kg=water_kg,
            acid_mol=acid,
            ree_mol=ree,
            other_mol=other,
        )
        return result
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def evaluate_separation_factor(
    element_a: str,
    element_b: str,
    temperature_C: float = 25.0,
    acid: dict = None,
    ree: dict = None,
    other: dict = None,
    preset: str = "light_ree",
) -> dict:
    """Calculate the separation factor β(A/B) between two REE elements.

    Runs Reaktoro equilibrium speciation, then computes the ratio of
    total dissolved amounts of element A vs element B.  β > 1 means
    element A partitions more into solution.

    Args:
        element_a: First REE element symbol (e.g. "Ce").
        element_b: Second REE element symbol (e.g. "Nd").
        temperature_C: Temperature in °C.
        acid: Acid species and molar amounts.
        ree: REE species and molar amounts.
        other: Other dissolved species.
        preset: REE element set.
    """
    try:
        from sep_agents.properties.ree_databases import REEEquilibriumSolver
        solver = REEEquilibriumSolver(preset=preset)
        result = solver.speciate(
            temperature_C=temperature_C,
            acid_mol=acid,
            ree_mol=ree,
            other_mol=other,
        )
        if result.get("status") != "ok":
            return result

        beta = solver.separation_factors(result, element_a, element_b)
        return {
            "status": "ok",
            "element_a": element_a,
            "element_b": element_b,
            "beta": round(beta, 6),
            "pH": result["pH"],
            "temperature_C": temperature_C,
            "ree_distribution": result["ree_distribution"],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

@mcp.tool()
def build_ree_flowsheet(
    feed_composition: dict,
    sx_distribution: dict,
    precipitation_temp_C: float = 25.0,
    organic_to_aqueous_ratio: float = 1.0,
    database: str = "light_ree"
) -> str:
    """Generate a YAML string for a basic REE separation flowsheet.
    
    Creates a flowsheet with a feed stream, a solvent extraction (SX) stage, 
    and a final crystallizer (precipitation) unit. This YAML can then be run 
    via `run_idaes_flowsheet`.
    
    Args:
        feed_composition: Dict of species to mass/moles, e.g. {"H2O(aq)": 1000, "Ce+3": 10}.
        sx_distribution: Dict predicting D for each species, e.g. {"Ce+3": 2.0, "Nd+3": 5.0}.
        precipitation_temp_C: Temperature for the final precipitation step.
        organic_to_aqueous_ratio: O/A volumetric ratio for the SX stage.
        database: Reaktoro database preset to use, e.g., "light_ree".
        
    Returns:
        YAML string representation of the flowsheet.
    """
    import yaml
    
    flowsheet = {
        "name": "ree_separation_flowsheet",
        "streams": [
            {
                "name": "feed",
                "phase": "liquid",
                "composition_wt": feed_composition
            }
        ],
        "units": [
            {
                "id": "sx_1",
                "type": "solvent_extraction",
                "inputs": ["feed"],
                "outputs": ["org_extract", "aq_raffinate"],
                "params": {
                    "distribution_coeff": sx_distribution,
                    "organic_to_aqueous_ratio": organic_to_aqueous_ratio
                }
            },
            {
                "id": "precipitator",
                "type": "crystallizer",
                "inputs": ["aq_raffinate"],
                "outputs": ["crystals", "liquor"],
                "params": {
                    "T_C": precipitation_temp_C
                }
            }
        ]
    }
    
    return yaml.dump(flowsheet, sort_keys=False)

@mcp.tool()
def simulate_sx_cascade(
    feed_composition: dict,
    stages: int,
    distribution_coeffs: dict,
    organic_to_aqueous_ratio: float = 1.0,
    database: str = "light_ree"
) -> dict:
    """Simulate a multi-stage solvent extraction (SX) cascade.
    
    Builds a flowsheet with sequential SX stages and runs it using IDAES.
    
    Args:
        feed_composition: Dict of aqueous feed species.
        stages: Number of SX stages to connect in cascade.
        distribution_coeffs: Dict predicting D for each species.
        organic_to_aqueous_ratio: O/A volumetric ratio.
        database: Reaktoro database preset.
        
    Returns:
        Results of `run_idaes_flowsheet` including the states of all streams.
    """
    import yaml
    
    streams = [{"name": "feed", "phase": "liquid", "composition_wt": feed_composition}]
    units = []
    
    last_aq = "feed"
    for i in range(1, stages + 1):
        aq_out = f"aq_{i}"
        org_out = f"org_{i}"
        units.append({
            "id": f"sx_{i}",
            "type": "solvent_extraction",
            "inputs": [last_aq],
            "outputs": [org_out, aq_out],
            "params": {
                "distribution_coeff": distribution_coeffs,
                "organic_to_aqueous_ratio": organic_to_aqueous_ratio
            }
        })
        last_aq = aq_out
        
    flowsheet = {
        "name": f"sx_cascade_{stages}_stages",
        "streams": streams,
        "units": units
    }
    
    fs_yaml = yaml.dump(flowsheet, sort_keys=False)
    
    try:
        from sep_agents.dsl.schemas import Flowsheet, UnitOp, Stream
        from sep_agents.sim.idaes_adapter import run_idaes
        
        data = yaml.safe_load(fs_yaml)
        fs_units = [UnitOp(**u) for u in data.get("units", [])]
        fs_streams = [Stream(**s) for s in data.get("streams", [])]
        fs = Flowsheet(name=data.get("name", "flowsheet"), units=fs_units, streams=fs_streams)
        
        return run_idaes(fs, database=database)
    except Exception as e:
        return {"status": "error", "error": str(e)}

# ---------------------------------------------------------------------------
# GDP Superstructure Optimization Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_superstructures_tool() -> dict:
    """List available predefined REE separation superstructures.

    Returns a list of superstructure templates that can be optimized
    using `optimize_superstructure_tool`.  Each entry includes the name,
    description, number of units, disjunctions, and optional units.
    """
    try:
        from sep_agents.dsl.ree_superstructures import list_superstructures
        return {"status": "ok", "superstructures": list_superstructures()}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def evaluate_topology(
    superstructure_name: str,
    active_unit_ids: list[str],
    param_overrides: dict = None,
    database: str = "light_ree",
) -> dict:
    """Evaluate a specific topology configuration from a superstructure.

    Builds a sub-flowsheet with only the specified active units, solves it,
    and returns KPIs.  Useful for comparing specific configurations without
    running full optimization.

    Args:
        superstructure_name: Name of a predefined superstructure (e.g. "simple_sx_precip").
        active_unit_ids: List of unit IDs to activate (others are bypassed).
        param_overrides: Optional dict of "unit_id.param_name" → value overrides.
        database: Reaktoro database preset.
    """
    try:
        from sep_agents.dsl.ree_superstructures import SUPERSTRUCTURE_REGISTRY
        from sep_agents.opt.gdp_builder import Configuration, build_sub_flowsheet
        from sep_agents.opt.gdp_solver import evaluate_configuration

        if superstructure_name not in SUPERSTRUCTURE_REGISTRY:
            return {"status": "error", "error": f"Unknown superstructure: {superstructure_name}. "
                    f"Available: {list(SUPERSTRUCTURE_REGISTRY.keys())}"}

        ss = SUPERSTRUCTURE_REGISTRY[superstructure_name]()
        all_ids = {u.id for u in ss.base_flowsheet.units}
        active = frozenset(active_unit_ids)
        bypassed = frozenset(all_ids - active)

        config = Configuration(id=0, active_unit_ids=active, bypassed_unit_ids=bypassed)
        ev = evaluate_configuration(ss, config, database=database, param_overrides=param_overrides)

        return {
            "status": ev.status,
            "active_units": sorted(ev.config.active_unit_ids),
            "bypassed_units": sorted(ev.config.bypassed_unit_ids),
            "kpis": ev.kpis,
            "objective_value": ev.objective_value,
            "elapsed_s": round(ev.elapsed_s, 2),
            "error": ev.error if ev.status != "ok" else None,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@mcp.tool()
def optimize_superstructure_tool(
    superstructure_name: str,
    objective: str = "minimize_opex",
    optimize_continuous: bool = False,
    n_bo_iters: int = 5,
    database: str = "light_ree",
) -> dict:
    """Optimize a superstructure to find the best process topology.

    Enumerates all valid topology configurations (which units to include,
    how many stages), evaluates each via sequential-modular simulation,
    and ranks results by the specified objective.  Optionally runs BoTorch
    Bayesian Optimization on continuous parameters within each topology.

    Args:
        superstructure_name: Name of a predefined superstructure
            (e.g. "simple_sx_precip", "lree_acid_leach").
        objective: Optimization objective. One of:
            "minimize_opex", "maximize_recovery", "minimize_lca",
            "maximize_value_per_kg_ore".
        optimize_continuous: If True, also optimize continuous parameters
            (e.g. O/A ratio) within each topology using BoTorch.
        n_bo_iters: Number of BoTorch iterations per topology (if enabled).
        database: Reaktoro database preset.
    """
    try:
        from sep_agents.dsl.ree_superstructures import SUPERSTRUCTURE_REGISTRY
        from sep_agents.opt.gdp_solver import optimize_superstructure

        if superstructure_name not in SUPERSTRUCTURE_REGISTRY:
            return {"status": "error", "error": f"Unknown superstructure: {superstructure_name}. "
                    f"Available: {list(SUPERSTRUCTURE_REGISTRY.keys())}"}

        ss = SUPERSTRUCTURE_REGISTRY[superstructure_name]()
        ss.objective = objective

        result = optimize_superstructure(
            ss,
            database=database,
            optimize_continuous=optimize_continuous,
            n_bo_iters=n_bo_iters,
        )

        # Format results for MCP response
        ranked = []
        for ev in sorted(result.all_results, key=lambda r: r.objective_value):
            ranked.append({
                "rank": len(ranked) + 1,
                "active_units": sorted(ev.config.active_unit_ids),
                "bypassed_units": sorted(ev.config.bypassed_unit_ids),
                "stage_choices": ev.config.stage_choices,
                "objective_value": round(ev.objective_value, 4),
                "kpis": {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in ev.kpis.items()},
                "optimized_params": ev.optimized_params,
                "status": ev.status,
                "elapsed_s": round(ev.elapsed_s, 2),
            })

        return {
            "status": "ok",
            "objective": objective,
            "num_configs_evaluated": result.num_configs_evaluated,
            "total_elapsed_s": round(result.total_elapsed_s, 1),
            "best": ranked[0] if ranked else None,
            "all_ranked": ranked,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


if __name__ == "__main__":
    mcp.run()  # stdio by default; see docs for other transports
