import json
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("runs/verify-mpi-effect")
rows = []

for nproc in [1, 2, 4, 8]:
    path = root / str(nproc)

    with open(path / "options.json") as f:
        params = json.load(f)

    with open(path / "memtot.json") as f:
        mem = json.load(f)

    eq, sp = [], []
    section = None

    with open(path / ".out") as f:
        for line in f:
            line = line.strip()

            if line.startswith("# Equilibration sweeps:"):
                section = eq
                continue

            if line.startswith("# Sampling sweeps:"):
                section = sp
                continue

            if line == "#":
                section = None
                continue

            fields = line.lstrip("#").split()

            if section is not None and fields and fields[0].isdigit():
                section.append(float(fields[-1]))

    rows.append({
        "n_proc": nproc,
        "nw/proc": params["n_walkers"],
        "PSS": mem["peak_pss_gib"],
        "RSS": mem["peak_rss_gib"],
        "t/init": eq[0],
        "avg/eq": np.mean(np.diff(eq)),
        "avg/sp": np.mean(np.diff(sp)),
        "t/total": mem["elapsed_s"],
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))