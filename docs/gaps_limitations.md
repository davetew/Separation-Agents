# Gaps and Limitations

The Separation Agents framework provides rigorous surrogate-assisted thermodynamic sequence modeling with both sequential-modular (SM) and equation-oriented (EO) backends. Several core limitations remain.

## 1. Deep Learning Kinetics
The IDAES simulator bridges the Reaktoro `ChemicalSystem` primarily by asserting *state equilibrium* for every sequential stream. Rare Earth Element precipitation and competitive solvent extraction (like PC88A multi-equilibrium cascades) can often be rate-limited by slow kinetic mixing that diverges heavily from theoretical equilibrium.

**Planned Resolution:** Replace pure sequence equilibrium solving with Deep Learning surrogate integrations predicting empirical deviations.

## 2. Deep BoTorch Multi-Objective Optimization
Currently the `BotorchOptimizer` evaluates exactly *one* scalar objective via `SingleTaskGP` (e.g., minimizing OPEX or maximizing $\beta$ separation). Practical flowsheets represent complex Pareto frontiers between competing economic ($$$) and engineering (% purity) criteria.

**Planned Resolution:** Upgrade the Bayesian surrogate to `qExpectedHypervolumeImprovement` over multiple outputs concurrently, resolving constrained constraint penalties directly.

## 3. TEA/LCA Proxy Generalization
The exact calculation of Operating Expenses (OPEX USD) and Cycle Emissions (CO$_2$e) presently relies on hard-coded generalized lookup parameters (e.g., predicting `$0.05 / 1000L` bulk water transport). While mathematically functional for relative gradient optimization, absolute dollar quotes should not be utilized in feasibility engineering.

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

## Resolved Limitations

| Item | Resolution |
|------|-----------|
| ~~Single-stage SX only~~ | EO backend supports McCabe-Thiele cascades via `build_sx_cascade` |
| ~~No topology optimization~~ | GDP via `solve_gdp_eo()` enables simultaneous topology + parameter optimization |
| ~~Sequential solving only~~ | EO backend solves entire flowsheet simultaneously via IPOPT |
