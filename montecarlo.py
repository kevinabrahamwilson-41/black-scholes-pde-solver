import numpy as np
import config

def monte_carlo_basket_call(
    S1,
    S2,
    n_paths=2_000_000,
    seed=42,
    batch_size=100_000,
):
    """Estimate a two-asset basket-call price using Monte Carlo."""
    rng = np.random.default_rng(seed)
    r = config.RISK_FREE_RATE
    T = config.MATURITY_TIME
    sigma1 = config.VOLATILITY_X
    sigma2 = config.VOLATILITY_Y
    rho = config.ASSET_CORRELATION
    K = config.STRIKE_PRICE
    drift1 = (r - 0.5 * sigma1**2) * T
    drift2 = (r - 0.5 * sigma2**2) * T
    vol1 = sigma1 * np.sqrt(T)
    vol2 = sigma2 * np.sqrt(T)
    sum_payoffs = 0.0
    sum_squared_payoffs = 0.0
    completed = 0
    while completed < n_paths:
        n = min(batch_size, n_paths - completed)
        z1 = rng.standard_normal(n)
        z2_independent = rng.standard_normal(n)
        # Correlated standard-normal shocks.
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * z2_independent
        ST1 = S1 * np.exp(drift1 + vol1 * z1)
        ST2 = S2 * np.exp(drift2 + vol2 * z2)
        payoff = np.maximum(0.5 * (ST1 + ST2) - K, 0.0)
        sum_payoffs += np.sum(payoff)
        sum_squared_payoffs += np.sum(payoff**2)
        completed += n
    mean_payoff = sum_payoffs / n_paths
    variance = (
        sum_squared_payoffs / n_paths - mean_payoff**2
    ) * n_paths / (n_paths - 1)
    discount = np.exp(-r * T)
    price = discount * mean_payoff
    standard_error = discount * np.sqrt(variance / n_paths)
    return price, standard_error

if __name__ == "__main__":
    S1 = 151.1811
    S2 = 151.1811
    price, standard_error = monte_carlo_basket_call(S1, S2)
    print("Monte Carlo basket-call validation")
    print(f"Spot prices: S1={S1:.4f}, S2={S2:.4f}")
    print(f"Estimated price: {price:.6f}")
    print(f"Standard error:  {standard_error:.6f}")
    print(
        f"Approximate 95% CI: "
        f"[{price - 1.96 * standard_error:.6f}, "
        f"{price + 1.96 * standard_error:.6f}]"
    )