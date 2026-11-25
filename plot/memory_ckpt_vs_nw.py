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

# Colors: strongest emphasis on CE50.
# Alpha decreases for intervals farther from 50.
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
    "CE400":  ("#174578", 1.0),

    "CSR400": ("#174578", 0.7),
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

plt.savefig(
    "plot/memory-ckpt-nw.png",
    dpi=300,
    bbox_inches="tight",
)
