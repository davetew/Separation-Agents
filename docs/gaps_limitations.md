# Gaps and Limitations

The Separation Agents framework provides highly rigorous surrogate-assisted thermodynamic sequence modeling, however Several core limitations remain.

## 1. Deep Learning Kinetics
The IDAES simulator bridges the Reaktoro `ChemicalSystem` primarily by asserting *state equilibrium* for every sequential stream. Rare Earth Element precipitation and competitive solvent extraction (like PC88A multi-equilibrium cascades) can often be rate-limited by slow kinetic mixing that diverges heavily from theoretical equilibrium.

**Planned Resolution:** Replace pure sequence equilibrium solving with Deep Learning surrogate integrations predicting empirical deviations.

## 2. Deep BoTorch Multi-Objective Optimization
Currently the `BotorchOptimizer` evaluates exactly *one* scalar objective via `SingleTaskGP` (e.g., minimizing OPEX or maximizing $\beta$ separation). Practical flowsheets represent complex Pareto frontiers between competing economic ($$$) and engineering (\% purity) criteria.

**Planned Resolution:** Upgrade the Bayesian surrogate to `qExpectedHypervolumeImprovement` over multiple outputs concurrently, resolving constrained constraint penalties directly.

## 3. TEA/LCA Proxy Generalization
The exact calculation of Operating Expenses (OPEX USD) and Cycle Emissions (CO$_2$e) presently relies on hard-coded generalized lookup parameters (e.g., predicting `$0.05 / 1000L` bulk water transport). While mathematically functional for relative gradient optimization, absolute dollar quotes should not be utilized in feasibility engineering.

**Planned Resolution:** Connect the internal Pydantic Unit blocks dynamically to an expansive database referencing contemporary commercial vendor scaling rules and up-to-date reagent grid pricing variables.
