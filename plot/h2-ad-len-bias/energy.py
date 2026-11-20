import pandas as pd
import matplotlib.pyplot as plt
import sys

# ============================================================
# Load data
# ============================================================

df = pd.read_csv(sys.argv[1])

x = df.iloc[:, 1].to_numpy()
energy = df.iloc[:, 2].to_numpy()
energy_se = df.iloc[:, 3].to_numpy()


# ============================================================
# Settings
# ============================================================

energy_ref = -1.11925
energy_ref_se = 2e-5

eq_sr = 20

error_multiplier = 1.96 # 95% CI

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

fig, ax = plt.subplots(figsize=(7, 5))

ax.errorbar(
    x,
    energy,
    yerr=error_multiplier * energy_se,
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
    energy_ref,
    color="tab:red",
    linestyle="--",
    linewidth=1.5,
    label="Reference, n = 75k"
)

ax.axhspan(
    energy_ref - energy_ref_se,
    energy_ref + energy_ref_se,
    color="tab:red",
    alpha=0.15,
    linewidth=0
)


# ============================================================
# Equilibration line
# ============================================================

ax.axvline(
    eq_sr,
    color="tab:green",
    linestyle="--",
    linewidth=1.5
)

ax.annotate(
    "eq phase = 80",
    xy=(eq_sr, 0.97),
    xycoords=("data", "axes fraction"),
    xytext=(5, 0),
    textcoords="offset points",
    rotation=90,
    va="top",
    ha="left",
    color="tab:green"
)


# ============================================================
# Axes
# ============================================================

ax.set_xlabel("SR intervals per AD")
ax.set_ylabel("Molecular energy")

ax.set_ylim(
    energy_ref - y_range / 2,
    energy_ref + y_range / 2
)

ax.ticklabel_format(
    axis="y",
    style="plain",
    useOffset=False
)

# Top x-axis
secax = ax.secondary_xaxis(
    "top",
    functions=(sr_to_time, time_to_sr)
)
secax.set_xlabel("Imaginary time")

ax.grid(alpha=0.2)
ax.legend(frameon=False, fontsize=8)

fig.tight_layout()
fig.savefig("energy.png", dpi=300, bbox_inches="tight")
