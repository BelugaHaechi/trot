import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.family"] = "Noto Sans"

import pandas as pd
from pathlib import Path
from scipy.stats import t

population = np.array([50, 100, 200, 400, 600, 800])

df = pd.read_csv("walltime.csv", keep_default_na=False, na_values=["nan", ""])

df["ckpt"] = df["relpath"].apply(
    lambda p: Path(p).parts[-3]
)
df["nw"] = df["relpath"].apply(
    lambda p: int(Path(p).parts[-2])
)

df = df.set_index(["ckpt", "nw"])

names = ["1", "5", "50", "200", "8000"]

eq, sp = {}, {}
ci_eq, ci_sp = {}, {}

for name in names:
    rows = df.reindex(
        pd.MultiIndex.from_product([[name], population])
    )

    eq[name] = rows["avg/eq"].to_numpy()
    sp[name] = rows["avg/sp"].to_numpy()

    ci_eq[name] = (
        t.ppf(0.975, df=9)
        * rows["std/eq"].to_numpy()
        / np.sqrt(10)
    )

    ci_sp[name] = (
        t.ppf(0.975, df=3)
        * rows["std/sp"].to_numpy()
        / np.sqrt(4)
    )

# Colors: strongest emphasis on CE50.
# Alpha decreases for intervals farther from 50.
# styles = {
#     "1":    ("darkorange", 0.5),
#     "2":    ("darkorange", 0.6),
#     "5":    ("darkorange", 0.7),
#     "10":   ("darkorange", 0.8),
#     "25":   ("darkorange", 0.8),
#     "50b":  ("darkorange", 1.0),

#     "50":   ("steelblue", 1.0),
#     "100":  ("steelblue", 0.8),
#     "200":  ("steelblue", 0.6),
#     "400":  ("steelblue", 0.5),

#     "None": ("darkred", 0.85),
# }

styles = {
    "1":    ("darkorange", 0.6),
    "5":    ("darkorange", 1.0),
    "50":   ("steelblue", 1.0),
    "200":  ("steelblue", 0.6),
    "8000": ("darkred", 0.85),
}

x_offsets = {
    "1":     -4,
    "5":      4,
    "50":     4,
    "200":   -4,
    "8000":   0,
}

fig, ax = plt.subplots(figsize=(8, 4.3))
fig.subplots_adjust(right=0.82)

for name, values in sp.items():
    y = np.asarray(values, dtype=float)
    color, alpha = styles[name]

    ax.errorbar(
        population + x_offsets[name], y,
        yerr=ci_sp[name],
        fmt="o-",
        color=color,
        alpha=alpha,
        markersize=3,
        linewidth=2.2 if name == "50" else 1.5,
        elinewidth=0.8,
        capsize=1.5,
        capthick=0.8,
        zorder=4 if name == "50" else 2,
    )

    if name == "8000": # no label for No checkpointing case
        continue

    ax.annotate(
        name,
        xy=(population[-1], y[-1]),
        xytext=(5, 0),
        textcoords="offset points",
        color=color,
        alpha=alpha,
        fontsize=9,
        va="center",
        annotation_clip=False,
    )

ax.set_xlabel("Sampler population")
ax.set_ylabel("Walltime per 8000 steps (s)")
ax.set_xticks(population)
ax.set_xlim(25, 875)
ax.set_ylim(0, 140)
ax.grid(alpha=0.2)
ax.set_axisbelow(True)


# Equilibration time
for name, values in eq.items():
    color, alpha = styles[name]
    ax.plot(
        population,
        10 * np.asarray(values),
        "o--",
        color=color,
        alpha=alpha,
        markersize=3,
        linewidth=1.5,
    )


# Legend: unchanged.
legend_handles = [
    Line2D([0], [0], color="darkorange", lw=2,
           label="Checkpoint before sampling"),
    Line2D([0], [0], color="steelblue", lw=2,
           label="Checkpoint after sampling"),
    Line2D([0], [0], color="darkred", lw=2,
           label="No checkpointing"),
    Line2D([0], [0], color="black", lw=1.5,
           linestyle="-", label="Reverse-mode AD"),
    Line2D([0], [0], color="black", lw=1.5,
           linestyle="--", label="Forward propagation"),
    Line2D(
        [], [],
        color="steelblue",
        marker="$50$",
        markersize=9,
        markeredgewidth=0,
        linestyle="none",
        label="Checkpoint interval (steps)",
    )
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    frameon=False,
    fontsize=9,
)

plt.savefig(
    "walltime-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)
