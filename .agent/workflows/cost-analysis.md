---
description: How to present cost analysis results to the user
---

# Cost Analysis Reporting Workflow

When the user requests a cost analysis or techno-economic assessment, follow these steps:

// turbo-all

1. Run the cost estimation using the `estimate_cost` MCP tool or the `jax_tea` module.

2. **Always produce a Markdown report** that includes:
   - **Request summary**: What was analyzed and why
   - **Major assumptions**: Throughput, commodity prices, discount rate, project life, reagent consumption, etc.
   - **CAPEX**: Total and itemized capital expenditure
   - **OPEX**: Total and itemized annual operating expenditure
   - **Levelized Cost of Production (LCOP)**: In output-specific metrics (e.g., $/kg or $/tonne)
   - **Revenue** (if applicable): From each product stream
   - **Net value**: Revenue minus annualized cost

3. **Scale mass units** for human readability:
   - Use millions of dollars ($M) for CAPEX/OPEX/revenue
   - Use thousands of tonnes (kt) or tonnes (t) for production volumes
   - Use $/kg or $/tonne for unit costs (choose whichever gives values in the 1–1000 range)

4. **Include sensitivity analysis** if JAX differentiation is available:
   - Report the top-5 cost drivers by |∂EAC/∂parameter|
   - Format as a table with parameter name and marginal cost impact

5. Present all tables using standard Markdown table syntax with aligned columns.
