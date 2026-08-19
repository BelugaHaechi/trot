import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import sys

root_pid = int(sys.argv[1])

df = pd.read_csv(f"{root_pid}.csv")

total = df[df["name"] == "TOTAL"]
processes = df[df["name"] != "TOTAL"]

fig, ax = plt.subplots(figsize=(10, 6))

# Total PSS
ax.plot(
    total["elapsed_s"],
    total["pss_gib"],
    linewidth=2.5,
    label="Total",
)

# Individual subprocesses
for pid, proc in processes.groupby("pid"):
    name = proc["name"].iloc[0]

    ax.plot(
        proc["elapsed_s"],
        proc["pss_gib"],
        linewidth=1,
        label=f"{name} ({pid})",
    )

# Axis labels
ax.set_xlabel("Time (s)")
ax.set_ylabel("PSS (GiB)")

# Y-axis:
# numbered ticks every 2 GiB
ax.yaxis.set_major_locator(MultipleLocator(2))

# gridlines every 0.5 GiB
ax.yaxis.set_minor_locator(MultipleLocator(0.5))

ax.grid(which="major", linewidth=0.8)
ax.grid(which="minor", linewidth=0.3)

ax.legend()

fig.tight_layout()

plt.savefig(
    f"{root_pid}.png",
    dpi=300,
    bbox_inches="tight",
)

print("Plot generated.")