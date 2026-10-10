import jax.numpy as jnp
import config

def evaluate_basket_call(S1, S2):
    """Calculates the intrinsic contract payoff profile for an average basket call."""
    average_asset_value = 0.5 * (S1 + S2)
    payoff = jnp.maximum(average_asset_value - config.STRIKE_PRICE, 0.0)
    return payoff

def evaluate_worst_of_call(S1, S2):
    """Calculates intrinsic value targeting the minimum performing asset node."""
    min_asset_value = jnp.minimum(S1, S2)
    return jnp.maximum(min_asset_value - config.STRIKE_PRICE, 0.0)
