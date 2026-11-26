import json
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams["font.family"] = "Noto Sans"

population = np.array([50, 100, 200, 400, 600, 800])

df = pd.read_csv("jobs/ckpt-vs-nw-v2.csv")
pss = {}
for _, row in df.iterrows():
    # Determine checkpoint strategy
    if row["n_steps_per_ckpt"]:
        name = f"CS{int(row['n_steps_per_ckpt'])}"
    elif row["n_ene_per_ckpt"]:
        name = f"CE{int(row['n_ene_per_ckpt'] * row['n_prop_steps'])}"
    elif row["do_sr_ckpt"]:
        name = f"CSR{int(row['n_ene_blocks'] * row['n_prop_steps'])}"
    else:
        name = "None"

    if name not in pss:
        pss[name] = np.full(len(population), np.nan)

    try:
        with open(Path(row["workdir"]) / "memtot.json") as f:
            mem = json.load(f)

        i = np.where(population == row["n_walkers"])[0][0]
        pss[name][i] = mem["peak_pss_gib"]

    except (OSError, ValueError, KeyError) as e:
        print(f"Failed: {row['workdir']}: {e}")

for name, values in pss.items():
    print(f"{name:>6}: {np.round(values, 3)}")

styles = {
    "CS1":  ("#df7500", 1.0),
    "CS2":  ("#e67e00", 1.0),
    "CS5":  ("#f59016", 1.0),
    "CS10": ("#ffa638", 1.0),
    "CS25": ("#fa9b28", 1.0),
    "CS50": ("#ed8700", 1.0),

    "CE50":   ("#91bde0", 1.0),
    "CE100":  ("#6096c4", 1.0),
    "CE200":  ("#326da3", 1.0),
    "CE400":  ("#174578", 0.7),

    "CSR400": ("#174578", 1.0),
    "None":   ("darkred", 0.85),
}

hidden = (
    "None",
    "CE400",
    "CS5",
    "CS25",
)

fig, ax = plt.subplots(figsize=(8, 4.3))
fig.subplots_adjust(right=0.82)

for name, values in pss.items():
    y = np.asarray(values, dtype=float)
    color, alpha = styles[name]

    if name in hidden:
        continue

    ax.plot(
        population, y, "o-",
        color=color,
        alpha=alpha,
        markersize=3,
        linewidth=2.2 if name == "CS10" else 1.2,
    )

    offsets = {
        # "CS10": (8, -10),
        # "CS25": (8, -6),
        # "CS5":  (8, -1),
        # "CE50": (8,  7),
        # "CS50": (8, 10),
    }
    dx, dy = offsets.get(name, (5, 0))

    ax.annotate(
        name.lstrip("CESR"),
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
axins = ax.inset_axes((0.07, 0.69, 0.34, 0.28))

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
    Line2D(
        [], [],
        color="steelblue",
        marker="$50$",
        markersize=9,
        markeredgewidth=0,
        linestyle="none",
        label="Checkpoint interval",
    )
]

ax.legend(
    handles=legend_handles,
    loc="upper left",
    bbox_to_anchor=(0.42, 0.99),
    frameon=False,
    fontsize=9,
)

# plt.savefig(
#     "plot/memory-ckpt-nw.png",
#     dpi=300,
#     bbox_inches="tight",
# )

# Memory growth reduction
slopes = {}

for name, values in pss.items():
    y = np.asarray(values)
    valid = np.isfinite(y)

    if valid.sum() < 2:
        continue

    slope, intercept = np.polyfit(
        population[valid], y[valid], 1
    )
    slopes[name] = slope

reduction = {
    name: slopes["None"] / slope
    for name, slope in slopes.items()
    if name != "None" and slope > 0
}

names = sorted(reduction, key=reduction.get)

fig, ax = plt.subplots(figsize=(6, 4.3))

for name in names:
    color, alpha = styles[name]
    value = reduction[name]

    if name == "CE400":
        continue

    ax.barh(
        name, value,
        color=color,
        alpha=alpha,
        height=0.65,
    )

    ax.text(
        value + 3, name,
        f"{value:.3g}×",
        va="center",
        fontsize=9,
    )

ax.set_xlabel("Memory growth reduction factor, linear fit")
ax.set_xlim(0, max(reduction.values()) * 1.15)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

plt.savefig(
    "plot/memory-growth-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)


# Memory-runtime trade-off

plt.rcParams["font.family"] = "Noto Sans"

from scipy.stats import t

population = np.array([50, 100, 200, 400, 600, 800])

df = pd.read_csv('plot/walltime-ckpt-nw.csv', keep_default_na=False, na_values=["nan", ""])

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
    "CS5",
    "CS10",
    "CS25",
    "CS50",
    "CE50",
    "CE100",
    "CE200",
    "CE400",
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

fig, ax = plt.subplots(figsize=(6, 4.3))

baseline = sp["8000"][-1]

for name, value in reduction.items():
    if name not in sp:
        continue

    x = value
    y = 100 * (sp[name][-1] / baseline - 1)
    color, alpha = styles[name]

    ax.scatter(
        x, y,
        color=color,
        alpha=alpha,
        s=35,
        zorder=3,
    )

    ax.annotate(
        name,
        xy=(x, y),
        xytext=(0, -8),
        ha="center",
        va="top",
        textcoords="offset points",
        color=color,
        fontsize=8,
    )

ax.set_xlabel("Memory growth reduction factor, linear fit")
ax.set_ylabel("AD runtime overhead (%)")
ax.set_xlim(0, 165)
ax.set_ylim(0, 65)
ax.grid(alpha=0.2)
ax.set_axisbelow(True)

groups = {
    "CS": ["CS1", "CS2", "CS5", "CS10", "CS25", "CS50"],
    "CE": ["CE50", "CE100", "CE200", "CE400"],
}

for group, names in groups.items():
    names = [n for n in names if n in reduction and n in sp]

    ax.plot(
        [reduction[n] for n in names],
        [100 * (sp[n][-1] / baseline - 1) for n in names],
        "-",
        color="darkorange" if group == "CS" else "steelblue",
        alpha=0.3,
        linewidth=1.8,
        zorder=1,
    )

plt.savefig(
    "plot/tradeoff-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)