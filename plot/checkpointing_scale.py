import matplotlib.pyplot as plt

population = [50, 200, 400, 600, 800]

pss_50 = [2.24, 2.36, 2.47, 2.60, 2.77]
pss_400 = [2.39, 2.76, 3.38, 3.90, 4.50]

fig, ax = plt.subplots(figsize=(4, 3.3))

ax.plot(population, pss_50, "o-",
        label="Checkpoint per 50 steps", color="steelblue")
ax.plot(population, pss_400, "s-",
        label="Checkpoint per 400 steps", color="orange")

ax.set_xlabel("Sampler population")
ax.set_ylabel("Memory usage, PSS (GiB)")
ax.set_xticks(population)
ax.set_ylim(2, 5)

ax.legend(frameon=False)
ax.grid(alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()
plt.savefig("checkpoint_scaling.png", dpi=300, bbox_inches="tight")
