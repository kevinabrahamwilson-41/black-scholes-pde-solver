import jax
import jax.numpy as jnp
import config
import system
import boundaries

@jax.jit
def step_explicit_euler(V, S1, S2, dx, dy, dt, tau):
    """Execute one explicit Euler step with time-dependent boundaries."""
    dV_dt = system.evaluate_black_scholes_rhs(V, S1, S2, dx, dy)
    V_next = V + dV_dt * dt
    V_next = boundaries.apply_boundary_conditions(
        V_next, S1, S2, tau
    )
    return V_next

def run_solver(S1, S2, initial_payoff, dx, dy, dt):
    """Advance the solution and report intermediate diagnostics."""
    V = initial_payoff
    report_steps = {
        1,
        config.TIME_STEPS // 4,
        config.TIME_STEPS // 2,
        3 * config.TIME_STEPS // 4,
        config.TIME_STEPS,
    }
    for step in range(1, config.TIME_STEPS + 1):
        tau = step * dt
        V = step_explicit_euler(
            V, S1, S2, dx, dy, dt, tau
        )
        if step in report_steps:
            V.block_until_ready()
            print(
                f"Step {step:>7}/{config.TIME_STEPS}: "
                f"tau={tau:.6f}, "
                f"min={float(jnp.min(V)):.6f}, "
                f"max={float(jnp.max(V)):.6f}, "
                f"center={float(V[V.shape[0] // 2, V.shape[1] // 2]):.6f}, "
                f"finite={bool(jnp.all(jnp.isfinite(V)))}"
            )
            if not bool(jnp.all(jnp.isfinite(V))):
                raise RuntimeError(
                    f"Non-finite values detected at step {step}"
                )
    return V