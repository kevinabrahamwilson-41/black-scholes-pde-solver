import jax.numpy as jnp
import config

def compute_first_derivative_x(V, dx):
    """Applies a central second-order finite difference stencil along Dimension X."""
    dV_dx = jnp.zeros_like(V)
    # Interior nodes
    dV_dx = dV_dx.at[1:-1, :].set((V[2:, :] - V[:-2, :]) / (2.0 * dx))
    return dV_dx

def compute_first_derivative_y(V, dy):
    """Applies a central second-order finite difference stencil along Dimension Y."""
    dV_dy = jnp.zeros_like(V)
    # Interior nodes
    dV_dy = dV_dy.at[:, 1:-1].set((V[:, 2:] - V[:, :-2]) / (2.0 * dy))
    return dV_dy

def compute_second_derivative_x(V, dx):
    """Applies central second-order second-derivative stencil along X."""
    d2V_dx2 = jnp.zeros_like(V)
    d2V_dx2 = d2V_dx2.at[1:-1, :].set((V[2:, :] - 2.0 * V[1:-1, :] + V[:-2, :]) / (dx ** 2))
    return d2V_dx2

def compute_second_derivative_y(V, dy):
    """Applies central second-order second-derivative stencil along Y."""
    d2V_dy2 = jnp.zeros_like(V)
    d2V_dy2 = d2V_dy2.at[:, 1:-1].set((V[:, 2:] - 2.0 * V[:, 1:-1] + V[:, :-2]) / (dy ** 2))
    return d2V_dy2

def compute_cross_derivative_xy(V, dx, dy):
    """Extracts second-order mixed partial derivatives across spatial topologies."""
    d2V_dxdy = jnp.zeros_like(V)
    d2V_dxdy = d2V_dxdy.at[1:-1, 1:-1].set(
        (V[2:, 2:] - V[2:, :-2] - V[:-2, 2:] + V[:-2, :-2]) / (4.0 * dx * dy)
    )
    return d2V_dxdy
