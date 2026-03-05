# Gaps and Limitations

The Separation Agents framework provides rigorous surrogate-assisted thermodynamic sequence modeling with three backends: sequential-modular (SM), equation-oriented (EO), and JAX differentiable. Several core limitations remain.

## 1. Deep Learning Kinetics
The IDAES simulator bridges the Reaktoro `ChemicalSystem` primarily by asserting *state equilibrium* for every sequential stream. Rare Earth Element precipitation and competitive solvent extraction (like PC88A multi-equilibrium cascades) can often be rate-limited by slow kinetic mixing that diverges heavily from theoretical equilibrium.

**Planned Resolution:** Replace pure sequence equilibrium solving with Deep Learning surrogate integrations predicting empirical deviations.

## 2. Deep BoTorch Multi-Objective Optimization
Currently the `BotorchOptimizer` evaluates exactly *one* scalar objective via `SingleTaskGP` (e.g., minimizing OPEX or maximizing $\beta$ separation). Practical flowsheets represent complex Pareto frontiers between competing economic ($$$) and engineering (% purity) criteria.

**Planned Resolution:** Upgrade the Bayesian surrogate to `qExpectedHypervolumeImprovement` over multiple outputs concurrently, resolving constrained constraint penalties directly.

## 3. TEA/LCA Proxy Generalization
The OPEX and CO₂e models rely on generalized proxy parameters (e.g., bulk water transport at $0.05/1000L). The JAX TEA module (`jax_tea.py`) provides differentiable itemized cost functions, but absolute dollar estimates remain at screening-level fidelity.

**Planned Resolution:** Connect the internal Pydantic Unit blocks dynamically to an expansive database referencing contemporary commercial vendor scaling rules and up-to-date reagent grid pricing variables.

## 4. Recycle Convergence
Neither the SM nor EO solver currently supports recycle (tear) streams. All flowsheets must be acyclic DAGs. This prevents modeling counter-current extraction cascades with true recycle topology.

**Planned Resolution:** Implement tear-stream convergence via Wegstein or Broyden methods in the EO solver.

## 5. EO Model Coverage
The EO backend supports three unit types (SX, Precipitator, IX). The SM backend supports additional types (reactor, crystallizer, separator, mixer, mill) that have not yet been ported to EO Pyomo blocks.

**Planned Resolution:** Port remaining SM unit types to EO blocks to enable full GDP optimization across all unit types.

## 6. IPOPT AMPL-ASL Compatibility
IPOPT 3.14.19 exhibits a segfault (SIGSEGV, RC -11) when successive solve calls are made in the same Python process. This is worked around by setting `TMPDIR=/tmp` and/or running benchmarks in subprocess isolation.

**Status:** Workaround in place. Upstream fix pending in IPOPT/Pyomo.

## 7. JAX EOS Species Coverage
The JAX equilibrium solver implements HKF (aqueous), Holland-Powell (mineral), and Peng-Robinson (gas) equations of state, but the built-in species database is limited to the `light_ree` preset (~30 species). Loading the full SUPCRTBL database (1100+ species) requires the JSON thermodynamic data file.

**Planned Resolution:** Bundle the full SUPCRTBL JSON and expand the preset system to cover `heavy_ree`, `full_ree`, and custom geo-chemical systems.

## 8. No SX Thermodynamic Model
Distribution coefficients ($D$) are empirical user inputs. Real SX systems exhibit pH-dependent, non-ideal $D$ values governed by extractant complexation equilibria (D2EHPA, PC88A, Cyanex 272).

**Planned Resolution:** Integrate COSMO-RS or D-pH correlation models to predict $D$ from first principles.

## Resolved Limitations

| Item | Resolution |
|------|-----------|
| ~~Single-stage SX only~~ | EO backend supports McCabe-Thiele cascades via `build_sx_cascade` |
| ~~No topology optimization~~ | GDP via `solve_gdp_eo()` enables simultaneous topology + parameter optimization |
| ~~Sequential solving only~~ | EO backend solves entire flowsheet simultaneously via IPOPT |
| ~~Reaktoro-only equilibrium~~ | JAX equilibrium solver (`jax_equilibrium.py`) provides differentiable GEM with HKF/HP/PR EOS |
| ~~No differentiable cost model~~ | JAX TEA (`jax_tea.py`) computes EAC with `jax.grad` autodiff sensitivity |
| ~~No topology screening~~ | GDP builder/solver (`gdp_builder.py` + `gdp_solver.py`) enumerates and ranks all feasible topologies |
| ~~Manual workflow only~~ | Seven agent workflows automate the full valorization pipeline from feed to report |
