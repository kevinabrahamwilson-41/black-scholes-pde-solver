import matplotlib.pyplot as plt

grid_sizes = [64, 128, 256]
pde_prices = [56.285920, 56.284135, 56.283609]

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    grid_sizes,
    pde_prices,
    marker="o",
    linewidth=2,
    markersize=7,
)

for grid, price in zip(grid_sizes, pde_prices):
    ax.annotate(
        f"{price:.6f}",
        (grid, price),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
    )

ax.set_title("Spatial Grid Convergence — 2D Black–Scholes PDE")
ax.set_xlabel("Spatial Grid Resolution (N × N)")
ax.set_ylabel("Interpolated Option Price")
ax.set_xticks(grid_sizes)
ax.set_ylim(56.2825, 56.2870)
ax.grid(True, linestyle="--", alpha=0.5)

fig.text(
    0.5,
    0.01,
    "Fixed time steps: 40,000 | Evaluation point: (151.1811, 151.1811)",
    ha="center",
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig("spatial_convergence.png", dpi=300, bbox_inches="tight")
plt.show()