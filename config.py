import os
# Force JAX to use 64-bit precision for high-fidelity numerical derivatives
os.environ["JAX_ENABLE_X64"] = "True"
# Ensure JAX locks onto the RTX 4060 GPU allocation platform
os.environ["JAX_PLATFORM_NAME"] = "gpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# Market Economics Constraints
STRIKE_PRICE = 100.0       # K
RISK_FREE_RATE = 0.05      # r
VOLATILITY_X = 0.25        # sigma_1 (Asset 1 Volatility)
VOLATILITY_Y = 0.30        # sigma_2 (Asset 2 Volatility)
ASSET_CORRELATION = 0.40   # rho (Cross-asset dependency)
MATURITY_TIME = 1.0        # T (Expiration time in years)

# Spatiotemporal Discretization Resolution
GRID_SIZE_X = 256
GRID_SIZE_Y = 256
TIME_STEPS = 160000
# Domain Bounds
S_MAX_X = 300.0            # Terminal valuation horizon for Asset 1
S_MAX_Y = 300.0            # Terminal valuation horizon for Asset 2
