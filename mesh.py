import jax.numpy as jnp
import config

def generate_spatial_domains():
    """Constructs continuous spatial meshes across the asset dimensions."""
    s_x = jnp.linspace(0.0, config.S_MAX_X, config.GRID_SIZE_X)
    s_y = jnp.linspace(0.0, config.S_MAX_Y, config.GRID_SIZE_Y)
    return s_x, s_y

def compute_tensor_mesh():
    """Generates a 2D coordinate lattice grid representing asset pairs."""
    s_x, s_y = generate_spatial_domains()
    S1, S2 = jnp.meshgrid(s_x, s_y, indexing='ij')
    return S1, S2

def generate_temporal_mesh():
    """Calculates temporal spatial vectors and reverse integration increments."""
    dt = config.MATURITY_TIME / config.TIME_STEPS
    time_vector = jnp.linspace(0.0, config.MATURITY_TIME, config.TIME_STEPS + 1)
    return dt, time_vector
