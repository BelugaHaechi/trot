
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

population = np.array([50, 100, 200, 400, 600, 800])

# Fill in your measured PSS values (GiB).
# Use np.nan for missing measurements.
pss = {
    # Checkpoint propagation only (CS)
    "1":   [np.nan] * 6,
    "2":   [np.nan] * 6,
    "5":   [np.nan] * 6,
    "10":  [np.nan] * 6,
    "25":  [np.nan] * 6,
    "50b": [np.nan] * 6,

    # Checkpoint entire energy block (CE)
    "50":  [np.nan] * 6,
    "100": [np.nan] * 6,
    "200": [np.nan] * 6,

    # Original checkpointing implementation
    "400": [np.nan] * 6,

    # No checkpointing
    "None": [np.nan] * 6,
}

# Colors: strongest emphasis on CE50.
# Alpha decreases for intervals farther from 50.
styles = {
    "1":    ("steelblue", 0.35),
    "2":    ("steelblue", 0.45),
    "5":    ("steelblue", 0.55),
    "10":   ("steelblue", 0.70),
    "25":   ("steelblue", 0.85),
    "50b":  ("steelblue", 1.00),

    "50":   ("darkorange", 1.00),
    "100":  ("darkorange", 0.70),
    "200":  ("darkorange", 0.45),

    "400":  ("black", 0.85),
    "None": ("darkred", 0.85),
}

fig, ax = plt.subplots(figsize=(5.6, 4.3))

# Leave space for endpoint labels.
fig.subplots_adjust(right=0.82)

lines = {}

for name, values in pss.items():
    y = np.asarray(values, dtype=float)
    color, alpha = styles[name]

    line, = ax.plot(
        population, y,
        "o-",
        color=color,
        alpha=alpha,
        markersize=3,
        linewidth=1.5 if name != "50" else 2.2,
        zorder=4 if name == "50" else 2,
    )
    lines[name] = line

    # Label the rightmost available point.
    # Exclude the no-checkpoint series from the main panel.
    if name == "None":
        continue

    valid = np.flatnonzero(np.isfinite(y))
    if len(valid) == 0:
        continue

    i = valid[-1]
    ax.annotate(
        name,
        xy=(population[i], y[i]),
        xytext=(5, 0),
        textcoords="offset points",
        color=color,
        alpha=alpha,
        fontsize=8,
        va="center",
        annotation_clip=False,
    )

ax.set_xlabel("Sampler population")
ax.set_ylabel("Memory usage, PSS (GiB)")
ax.set_xticks(population)
ax.set_xlim(35, 875)
ax.set_ylim(0, 8)

ax.grid(alpha=0.2)
ax.set_axisbelow(True)

# Legend: only the four checkpointing categories.
legend_handles = [
    Line2D([0], [0], color="steelblue", lw=2,
           label="Checkpoint propagation only"),
    Line2D([0], [0], color="darkorange", lw=2,
           label="Checkpoint entire energy block"),
    Line2D([0], [0], color="black", lw=2,
           label="Original checkpointing implementation"),
    Line2D([0], [0], color="darkred", lw=2,
           label="No checkpointing"),
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
    fontsize=7,
)

# Inset: show all 11 series, including no checkpointing.
axins = inset_axes(
    ax,
    width="43%",
    height="36%",
    loc="upper right",
    borderpad=1.0,
)

for name, values in pss.items():
    color, alpha = styles[name]
    axins.plot(
        population,
        values,
        "o-",
        color=color,
        alpha=alpha,
        markersize=1.5,
        linewidth=1,
    )

axins.set_xlim(35, 825)
axins.set_ylim(0, 45)
axins.set_xticks([50, 400, 800])
axins.set_yticks([0, 20, 40])
axins.tick_params(labelsize=6)
axins.grid(alpha=0.15)
axins.set_title("Full range", fontsize=7)

plt.savefig(
    "checkpoint_scaling.png",
    dpi=300,
    bbox_inches="tight",
)
