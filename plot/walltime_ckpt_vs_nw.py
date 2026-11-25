import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.family"] = "Noto Sans"

import pandas as pd
from pathlib import Path
from scipy.stats import t

population = np.array([50, 100, 200, 400, 600, 800])

df = pd.read_csv(sys.argv[1], keep_default_na=False, na_values=["nan", ""])

df["ckpt"] = df["relpath"].apply(
    lambda p: Path(p).parts[-3]
)
df["nw"] = df["relpath"].apply(
    lambda p: int(Path(p).parts[-2])
)

df = df.set_index(["ckpt", "nw"])

names = [
    "CS1",
    "CS2",
    # "CS5",
    "CS10",
    # "CS25",
    "CS50",
    "CE50",
    # "CE100",
    "CE200",
    # "CE400",
    "CSR400",
    "8000",
]

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

print("\nForward propagation (8000 steps):")
for name, values in eq.items():
    print(f"{name:>6}: {np.round(10 * values, 3)}")

print("\nReverse-mode AD (8000 steps):")
for name, values in sp.items():
    print(f"{name:>6}: {np.round(values, 3)}")

print("\nAD 95% CI half-width (s):")
for name, values in ci_sp.items():
    print(f"{name:>6}: {np.round(values, 3)}")

styles = {
    "CS1":    ("#df7500", 1.0),
    "CS2":    ("#e67e00", 1.0),
    "CS5":    ("#f59016", 1.0),
    "CS10":   ("#ffa638", 1.0),
    "CS25":   ("#fa9b28", 1.0),
    "CS50":   ("#ed8700", 1.0),

    "CE50":   ("#91bde0", 1.0),
    "CE100":  ("#6096c4", 1.0),
    "CE200":  ("#326da3", 1.0),
    "CE400":  ("#174578", 1.0),

    "CSR400": ("#174578", 1.0),
    "8000":   ("darkred", 0.85),
}

fig, ax = plt.subplots(figsize=(8, 4.3))
fig.subplots_adjust(right=0.82)

for name, values in sp.items():
    y = np.asarray(values, dtype=float)
    color, alpha = styles[name]

    ax.errorbar(
        population, y,
        yerr=ci_sp[name],
        fmt="-",
        color=color,
        alpha=alpha,
        markersize=3,
        elinewidth=0.8,
        capsize=2,
        capthick=1.0,
        linewidth=1.8 if name == "CS10" else 1.2,
        zorder=4 if name == "CS10" else 2,
    )

    if name == "8000": # Hide label for no-ckpt
        continue

    offsets = {
        "CS10": (8,  3),
        "CS1":  (8, -3),
        "CE50": (8,  4),
        "CS50": (8, -4),
    }

    ax.annotate(
        name.lstrip("CESR"),
        xy=(population[-1], y[-1]),
        xytext=offsets.get(name, (5, 0)),
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
    "plot/walltime-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)

# Relative graph

baseline = 10 * np.asarray(eq["8000"])

fig, ax = plt.subplots(figsize=(8, 4.3))
fig.subplots_adjust(right=0.82)

# Reverse-mode AD
for name, values in sp.items():
    y = np.asarray(values) / baseline
    color, alpha = styles[name]

    ax.errorbar(
        population, y,
        yerr=ci_sp[name] / baseline,
        fmt="-",
        color=color,
        alpha=alpha,
        linewidth=2.2 if name == "CS10" else 1.5,
        elinewidth=0.8,
        capsize=2,
        capthick=0.8,
        zorder=4 if name == "CS10" else 2,
    )

    if name == "8000":
        continue

    ax.annotate(
        name.lstrip("CESR"),
        xy=(population[-1], y[-1]),
        xytext=offsets.get(name, (5, 0)),
        textcoords="offset points",
        color=color,
        alpha=alpha,
        fontsize=9,
        va="center",
        annotation_clip=False,
    )

# Forward propagation
for name, values in eq.items():
    color, alpha = styles[name]

    if name == "8000":
        continue

    ax.plot(
        population,
        10 * np.asarray(values) / baseline,
        "o--",
        color=color,
        alpha=alpha,
        markersize=3,
        linewidth=1.5,
    )

ax.set_xlabel("Sampler population")
ax.set_ylabel("Relative walltime")
ax.set_xticks(population)
ax.set_xlim(25, 875)
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.set_axisbelow(True)

# ax.legend(
#     handles=legend_handles,
#     loc="upper left",
#     frameon=False,
#     fontsize=9,
# )

plt.savefig(
    "plot/walltime-rel-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)