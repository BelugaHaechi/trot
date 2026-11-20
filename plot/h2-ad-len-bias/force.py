import pandas as pd
import matplotlib.pyplot as plt
import sys

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(sys.argv[1])

x = df.iloc[:, 1].to_numpy()
force = df.iloc[:, 4].to_numpy()
force_se = df.iloc[:, 5].to_numpy()

# ============================================================
# Settings
# ============================================================

# Reference
force_ref = -0.11075
force_ref_se = 10e-5

# 7-point finite difference
fd_ref = -0.11012
fd_ref_se = 75e-5

# Equilibration point
eq_sr = 20

error_multiplier = 1.96

# Total y-axis range
y_range = 0.005

# ============================================================
# Axis conversion
# ============================================================

def sr_to_time(x):
    return 4.0 * x

def time_to_sr(t):
    return t / 4.0


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(figsize=(5, 3))

ax.errorbar(
    x,
    force,
    yerr=error_multiplier * force_se,
    fmt="x",            # x-shaped data points
    markersize=4,       # smaller marker
    markeredgewidth=1,  # thickness of the x
    linestyle="none",
    capsize=2,
    elinewidth=1,
    alpha=0.8,
    label="Tests, n = O(1000), 95% CI"
)


# ============================================================
# Reference
# ============================================================

ax.axhline(
    force_ref,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label="Reference, n = 31.2k, l = 20k"
)

ax.axhspan(
    force_ref - force_ref_se,
    force_ref + force_ref_se,
    color="tab:red",
    alpha=0.15,
    linewidth=0
)


# ============================================================
# 7-point finite difference
# ============================================================

ax.axhline(
    fd_ref,
    color="tab:purple",
    linestyle="--",
    linewidth=1.5,
    label="7-point finite difference, n = 8.8k, l = 20k"
)

ax.axhspan(
    fd_ref - fd_ref_se,
    fd_ref + fd_ref_se,
    color="tab:purple",
    alpha=0.12,
    linewidth=0
)


# # ============================================================
# # Equilibration line
# # ============================================================

# ax.axvline(
#     eq_sr,
#     color="tab:green",
#     linestyle="--",
#     linewidth=1.5
# )

# ax.annotate(
#     "eq phase = 80",
#     xy=(eq_sr, 0.97),
#     xycoords=("data", "axes fraction"),
#     xytext=(5, 0),
#     textcoords="offset points",
#     rotation=90,
#     va="top",
#     ha="left",
#     color="tab:green"
# )


# ============================================================
# Axes
# ============================================================

ax.set_xlabel("Propagation length / 1000")
ax.set_ylabel("Energy derivative")

# 0.05 total y range
ax.set_ylim(
    force_ref - y_range / 2,
    force_ref + y_range / 2.5
)

ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig("force_1z.png", dpi=300, bbox_inches="tight")
