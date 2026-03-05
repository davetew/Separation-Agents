import pytest
import jax.numpy as jnp
from sep_agents.cost.jax_tea import (
    mining_cost,
    comminution_cost,
    leaching_cost,
    processing_cost,
    itemized_cost,
    total_annualized_cost,
    cost_sensitivity
)

@pytest.fixture
def default_params():
    return {
        "ore_throughput_tpd": jnp.array(1000.0),
        "strip_ratio": jnp.array(2.0),
        "mine_depth_m": jnp.array(50.0),
        "bond_work_index": jnp.array(15.0),
        "residence_time_h": jnp.array(24.0),
        "acid_consumption_kg_t": jnp.array(50.0),
        "operating_temp_c": jnp.array(80.0),
        "sx_stages": jnp.array(10.0),
        "precipitation_reagent_tpy": jnp.array(500.0),
        "aq_flow_m3_h": jnp.array(100.0)
    }

def test_mining_cost(default_params):
    capex, opex = mining_cost(default_params)
    assert capex.shape == ()
    assert opex.shape == ()
    assert capex > 0
    assert opex > 0

def test_comminution_cost(default_params):
    capex, opex = comminution_cost(default_params)
    assert capex > 0
    assert opex > 0
    
def test_leaching_cost(default_params):
    capex, opex = leaching_cost(default_params)
    assert capex > 0
    assert opex > 0

def test_processing_cost(default_params):
    capex, opex = processing_cost(default_params)
    assert capex > 0
    assert opex > 0

def test_itemized_cost(default_params):
    costs = itemized_cost(default_params)
    assert isinstance(costs, dict)
    assert "total_capex" in costs
    assert "total_opex" in costs
    assert costs["total_capex"] > 0
    assert costs["total_opex"] > 0

def test_total_annualized_cost(default_params):
    eac1 = total_annualized_cost(default_params, discount_rate=0.08)
    eac2 = total_annualized_cost(default_params, discount_rate=0.10)
    assert eac1 > 0
    assert eac2 > eac1 # Higher discount rate = higher annualized CAPEX recovery
    
def test_cost_sensitivity(default_params):
    # Compute auto-diff gradients w.r.t parameters
    sensitivities = cost_sensitivity(default_params)
    
    # Assert gradient structure matches input parameter structure
    assert set(sensitivities.keys()) == set(default_params.keys())
    
    # Sensitivities should be scalars
    assert sensitivities["ore_throughput_tpd"].shape == ()
    
    # Higher throughput should increase cost (gradient is positive)
    assert sensitivities["ore_throughput_tpd"] > 0
    
    # Higher strip ratio should increase cost (gradient is positive)
    assert sensitivities["strip_ratio"] > 0
