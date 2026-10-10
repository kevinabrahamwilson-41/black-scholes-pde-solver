import jax.numpy as jnp
import jax.scipy as jsp
import config

def black_scholes_call(S, K, r, sigma, tau):
    """Analytical European call price under the Black-Scholes model."""
    tau_safe = jnp.maximum(tau, 1e-12)
    sqrt_tau = jnp.sqrt(tau_safe)
    d1 = (
        jnp.log(jnp.maximum(S, 1e-12) / K)
        + (r + 0.5 * sigma**2) * tau_safe
    ) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    price = (
        S * jsp.special.ndtr(d1)
        - K * jnp.exp(-r * tau_safe) * jsp.special.ndtr(d2)
    )
    # At expiry, return the intrinsic call payoff.
    intrinsic = jnp.maximum(S - K, 0.0)
    return jnp.where(tau <= 0.0, intrinsic, price)

def apply_boundary_conditions(V, S1, S2, tau):
    """Apply lower analytical and upper extrapolated boundary conditions."""
    # S1 = 0: half-weighted call on asset 2, strike 2K.
    lower_x = 0.5 * black_scholes_call(
        S2[0, :],
        2.0 * config.STRIKE_PRICE,
        config.RISK_FREE_RATE,
        config.VOLATILITY_Y,
        tau,
    )
    V = V.at[0, :].set(lower_x)
    # S2 = 0: half-weighted call on asset 1, strike 2K.
    lower_y = 0.5 * black_scholes_call(
        S1[:, 0],
        2.0 * config.STRIKE_PRICE,
        config.RISK_FREE_RATE,
        config.VOLATILITY_X,
        tau,
    )
    V = V.at[:, 0].set(lower_y)
    # Upper boundaries: zero-second-derivative extrapolation.
    V = V.at[-1, :].set(2.0 * V[-2, :] - V[-3, :])
    V = V.at[:, -1].set(2.0 * V[:, -2] - V[:, -3])
    return V