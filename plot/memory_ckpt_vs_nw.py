import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.family"] = "Noto Sans"

population = np.array([50, 100, 200, 400, 600, 800])
pss = {
    '1': [2.28,2.45,2.73,3.24,3.76,4.21],
    '2': [2.20,2.29,2.54,2.81,3.17,3.56],
    '5': [2.16,2.26,2.44,2.68,2.94,3.15],
    '10': [2.18,2.19,2.32,2.67,2.82,3.01],
    '25': [2.20,2.31,2.43,2.74,2.83,3.02],
    '50b': [2.19,2.25,2.42,2.73,2.89,3.00], # dt50
    'None': [5.18,7.73,12.02,21.32,28.93,40.32],
    '50': [2.10,2.12,2.16,2.27,2.41,2.51],
    '100': [2.19,2.21,2.32,2.51,2.74,2.93],
    '200': [2.18,2.26,2.39,2.73,3.05,3.30],
    '400': [2.22,2.40,2.65,3.19,3.72,4.22],
}

# Colors: strongest emphasis on CE50.
# Alpha decreases for intervals farther from 50.
styles = {
    "1":    ("darkorange", 0.5),
    "2":    ("darkorange", 0.6),
    "5":    ("darkorange", 0.7),
    "10":   ("darkorange", 0.8),
    "25":   ("darkorange", 0.8),
    "50b":  ("darkorange", 1.0),

    "50":   ("steelblue", 1.0),
    "100":  ("steelblue", 0.8),
    "200":  ("steelblue", 0.6),
    "400":  ("steelblue", 0.5),

    "None": ("darkred", 0.85),
}

fig, ax = plt.subplots(figsize=(8, 4.3))
fig.subplots_adjust(right=0.82)

for name, values in pss.items():
    y = np.asarray(values, dtype=float)
    color, alpha = styles[name]

    # No-checkpoint data appears only in the inset.
    if name == "None":
        continue
    # Hide these two as well, too close to 25
    if name in ("10", "50b"):
        continue

    ax.plot(
        population, y, "o-",
        color=color,
        alpha=alpha,
        markersize=3,
        linewidth=2.2 if name == "50" else 1.5,
        zorder=4 if name == "50" else 2,
    )


    offsets = {
        "1":   (10, 5),
        "400": (10, -5),
    }

    dx, dy = offsets.get(name, (5, 0))

    ax.annotate(
        name,
        xy=(population[-1], y[-1]),
        xytext=(dx, dy),
        textcoords="offset points",
        color=color,
        alpha=alpha,
        fontsize=9,
        va="center",
        annotation_clip=False,
        arrowprops=(
            dict(
                arrowstyle="-",
                color=color,
                alpha=alpha,
                lw=0.8,
                shrinkA=0,
                shrinkB=3,
            )
            if name in offsets else None
        ),
    )

ax.set_xlabel("Sampler population")
ax.set_ylabel("Memory usage, PSS (GiB)")
ax.set_xticks(population)
ax.set_xlim(35, 875)
ax.set_ylim(2, 4.5)
ax.grid(alpha=0.2)
ax.set_axisbelow(True)

# Smaller inset in upper-left corner.
axins = inset_axes(
    ax,
    width="34%",
    height="28%",
    loc="upper left",
    borderpad=(2.8, 1.0),
)

for name, values in pss.items():
    color, alpha = styles[name]
    axins.plot(
        population, values, "o-",
        color=color,
        alpha=alpha,
        markersize=1.8,
        linewidth=1,
    )

axins.set_xlim(35, 825)
axins.set_ylim(0, 45)
axins.set_xticks([50, 400, 800])
axins.set_yticks([0, 20, 40])
axins.tick_params(labelsize=9)
axins.grid(alpha=0.15)
axins.set_axisbelow(True)

# Legend below inset, also in upper-left region.
legend_handles = [
    Line2D([0], [0], color="darkorange", lw=2,
           label="Checkpoint before sampling"),
    Line2D([0], [0], color="steelblue", lw=2,
           label="Checkpoint after sampling"),
    Line2D([0], [0], color="darkred", lw=2,
           label="No checkpointing"),
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.42, 0.99),
    frameon=False,
    fontsize=9,
    labelspacing=0.1,
)

ax.annotate(
    "Steps per\ncheckpoint",
    xy=(820, pss["50"][-1] -0.05),
    xytext=(790, 2.1),
    color="black",
    fontsize=9,
    ha="center",
    arrowprops=dict(
        arrowstyle="-",
        color="black",
        lw=0.7,
        shrinkA=0,
        shrinkB=3,
    ),
)

plt.savefig(
    "memory-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)
