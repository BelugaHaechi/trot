import matplotlib.pyplot as plt

intervals = [1, 50, 100, 200, 400, "None"]
pss = [3.16, 2.29, 2.54, 2.41, 2.52, 9.62]

fig, ax = plt.subplots(figsize=(2, 3))

ax.bar(range(len(intervals)), pss, color="steelblue")

ax.set_xticks(range(len(intervals)))
ax.set_xticklabels(intervals, rotation=-60)
ax.set_xlabel("Checkpoint interval")
ax.set_ylabel("Memory usage, PSS (GiB)")

ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()
plt.savefig("checkpoint_memory.png", dpi=300, bbox_inches="tight")
