import jax.numpy as jnp
import config
import operators

def evaluate_black_scholes_rhs(V, S1, S2, dx, dy):
    """Assembles the complete explicit differential field step for the 2D operator."""
    # Compute derivative maps
    dV_dx = operators.compute_first_derivative_x(V, dx)
    dV_dy = operators.compute_first_derivative_y(V, dy)
    d2V_dx2 = operators.compute_second_derivative_x(V, dx)
    d2V_dy2 = operators.compute_second_derivative_y(V, dy)
    d2V_dxdy = operators.compute_cross_derivative_xy(V, dx, dy)

    # Differential Component Allocations
    theta_term = -config.RISK_FREE_RATE * V
    convective_x = config.RISK_FREE_RATE * S1 * dV_dx
    convective_y = config.RISK_FREE_RATE * S2 * dV_dy
    diffusive_x = 0.5 * (config.VOLATILITY_X ** 2) * (S1 ** 2) * d2V_dx2
    diffusive_y = 0.5 * (config.VOLATILITY_Y ** 2) * (S2 ** 2) * d2V_dy2
    cross_diffusion = config.ASSET_CORRELATION * config.VOLATILITY_X * config.VOLATILITY_Y * S1 * S2 * d2V_dxdy

    # Accumulate continuous spatial RHS derivative mapping matrix
    dV_dt = theta_term + convective_x + convective_y + diffusive_x + diffusive_y + cross_diffusion
    return dV_dt
