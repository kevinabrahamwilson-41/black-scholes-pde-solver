import time
import jax.numpy as jnp
import boundaries
import config
import mesh
import payoff
import engine
import os
os.environ["JAX_ENABLE_X64"] = "True"
import jax
jax.config.update("jax_enable_x64", True)
def bilinear_interpolate(V, s_x, s_y, target_x, target_y):
    """Interpolate V at a fixed point on a uniform 2D grid."""
    i = int((target_x - float(s_x[0])) /
            float(s_x[1] - s_x[0]))
    j = int((target_y - float(s_y[0])) /
            float(s_y[1] - s_y[0]))
    i = max(0, min(i, len(s_x) - 2))
    j = max(0, min(j, len(s_y) - 2))
    x0, x1 = float(s_x[i]), float(s_x[i + 1])
    y0, y1 = float(s_y[j]), float(s_y[j + 1])
    wx = (target_x - x0) / (x1 - x0)
    wy = (target_y - y0) / (y1 - y0)
    v00 = float(V[i, j])
    v10 = float(V[i + 1, j])
    v01 = float(V[i, j + 1])
    v11 = float(V[i + 1, j + 1])
    return (
        (1 - wx) * (1 - wy) * v00
        + wx * (1 - wy) * v10
        + (1 - wx) * wy * v01
        + wx * wy * v11
    )

def main():
    print("====================================================")
    print("INITIALIZING GPU ACCELERATED 2D BLACK-SCHOLES SOLVER")
    print("====================================================")
    # 1. Initialize Domains and Grids
    s_x, s_y = mesh.generate_spatial_domains()
    dx = s_x[1] - s_x[0]
    dy = s_y[1] - s_y[0]
    dt, _ = mesh.generate_temporal_mesh()
    S1, S2 = mesh.compute_tensor_mesh()
    # 2. Extract Terminal Payoff Condition
    print("Computing contract payoff profiles...")
    initial_payoff = payoff.evaluate_basket_call(S1, S2)
    # 3. Trigger GPU Warmup Run (To capture XLA compilation overhead upfront)
    print("Compiling CUDA Kernels on RTX 4060 device pipeline...")
    _ = engine.step_explicit_euler(
            initial_payoff, S1, S2, dx, dy, dt, dt
        )
    # 4. Run full backward time-stepping simulation
    print(f"Executing {config.TIME_STEPS} backwards integration steps...")
    start_time = time.perf_counter()
    final_option_prices = engine.run_solver(S1, S2, initial_payoff, dx, dy, dt)
    print("\n=== JAX Precision Check ===")
    print("JAX x64 enabled:", jax.config.read("jax_enable_x64"))
    print("Grid dtype:", S1.dtype)
    print("Solution dtype:", final_option_prices.dtype)
    print("Lower x boundary error:",
      float(jnp.max(jnp.abs(
          final_option_prices[0, :]
          - 0.5 * boundaries.black_scholes_call(
              S2[0, :],
              2.0 * config.STRIKE_PRICE,
              config.RISK_FREE_RATE,
              config.VOLATILITY_Y,
              config.MATURITY_TIME
          )
      ))))

    print("Lower y boundary error:",
        float(jnp.max(jnp.abs(
            final_option_prices[:, 0]
            - 0.5 * boundaries.black_scholes_call(
                S1[:, 0],
                2.0 * config.STRIKE_PRICE,
                config.RISK_FREE_RATE,
                config.VOLATILITY_X,
                config.MATURITY_TIME
            )
        ))))
    # Debug interpolation using the final PDE solution
    target_spot = 151.1811
    i = int((target_spot - float(s_x[0])) / float(s_x[1] - s_x[0]))
    j = int((target_spot - float(s_y[0])) / float(s_y[1] - s_y[0]))
    print("\n=== Interpolation Debug ===")
    print(f"Grid indices: i={i}, j={j}")
    print(f"S1 coordinates: {float(s_x[i]):.6f}, {float(s_x[i+1]):.6f}")
    print(f"S2 coordinates: {float(s_y[j]):.6f}, {float(s_y[j+1]):.6f}")
    print(f"V[i, j]       : {float(final_option_prices[i, j]):.6f}")
    print(f"V[i+1, j]     : {float(final_option_prices[i+1, j]):.6f}")
    print(f"V[i, j+1]     : {float(final_option_prices[i, j+1]):.6f}")
    print(f"V[i+1, j+1]   : {float(final_option_prices[i+1, j+1]):.6f}")
    print("JAX x64 enabled:", jax.config.read("jax_enable_x64"))
    print("Grid dtype:", S1.dtype)
    print("Solution dtype:", final_option_prices.dtype)
    interpolated_price = bilinear_interpolate(
        final_option_prices,
        s_x,
        s_y,
        target_spot,
        target_spot,
    )
    print(
        f"Interpolated price at "
        f"({target_spot:.4f}, {target_spot:.4f}): "
        f"{interpolated_price:.6f}"
    )
    print("\n=== Local Grid Diagnostic ===")
    print("dx:", float(dx), "dy:", float(dy))

    print("S1 target offset:",
          (target_spot - float(s_x[i])) / float(dx))
    print("S2 target offset:",
          (target_spot - float(s_y[j])) / float(dy))

    print("dV/dS1 along lower edge:",
          float((final_option_prices[i + 1, j]
                 - final_option_prices[i, j]) / dx))

    print("dV/dS2 along left edge:",
          float((final_option_prices[i, j + 1]
                 - final_option_prices[i, j]) / dy))

    print("dV/dS1 along upper edge:",
          float((final_option_prices[i + 1, j + 1]
                 - final_option_prices[i, j + 1]) / dx))

    print("dV/dS2 along right edge:",
          float((final_option_prices[i + 1, j + 1]
                 - final_option_prices[i + 1, j]) / dy))
    # 5. Validate against the Monte Carlo reference
    reference_price = 56.319219
    mc_standard_error = 0.024771
    absolute_error = abs(interpolated_price - reference_price)
    relative_error = (absolute_error / abs(reference_price)) * 100.0
    print("\n=== Reference Validation ===")
    print(f"Monte Carlo reference : {reference_price:.6f}")
    print(f"PDE price             : {interpolated_price:.6f}")
    print(f"Absolute error        : {absolute_error:.6f}")
    print(f"Relative error        : {relative_error:.4f}%")
    print(f"MC standard error     : {mc_standard_error:.6f}")
    print(f"MC 95% confidence CI  : [56.270669, 56.367770]")
    # Block until execution complete to get precise hardware performance timings
    final_option_prices.block_until_ready()
    if not bool(jnp.all(jnp.isfinite(final_option_prices))):
        raise RuntimeError("Solver produced NaN or Inf values.")
    print(f"All values finite: {bool(jnp.all(jnp.isfinite(final_option_prices)))}")
    print(f"Minimum option value: {float(jnp.min(final_option_prices)):.6f}")
    print(f"Maximum option value: {float(jnp.max(final_option_prices)):.6f}")
    print(f"Negative values: {int(jnp.sum(final_option_prices < 0.0))}")
    # Locate negative option prices.
    negative_mask = final_option_prices < 0.0
    negative_indices = jnp.argwhere(negative_mask)
    print("\nFirst 10 negative grid points:")
    for i, j in negative_indices[:10].tolist():
        print(
            f"S1={float(S1[i, j]):.4f}, "
            f"S2={float(S2[i, j]):.4f}, "
            f"V={float(final_option_prices[i, j]):.8f}"
        )
    end_time = time.perf_counter()
    print(f"Solver execution time: {end_time - start_time:.4f} seconds")
    # 5. Extract a focal point price metrics for verification
    mid_idx_x = config.GRID_SIZE_X // 2
    mid_idx_y = config.GRID_SIZE_Y // 2
    print(f"Sample Option Price at Node ({S1[mid_idx_x, mid_idx_y]:.2f}, {S2[mid_idx_x, mid_idx_y]:.2f}): {final_option_prices[mid_idx_x, mid_idx_y]:.4f}")
    print("====================================================")

if __name__ == "__main__":
    main()
