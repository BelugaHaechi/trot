import sys
import os
import csv
import re
import statistics
from pathlib import Path

N_EQ = 10
N_SP = 4

NAN = float("nan")


def read_walltimes(path):
    eq = {}
    sp = {}
    section = None

    with path.open() as f:
        for line in f:
            line = line.strip()

            if line.startswith("# Equilibration sweeps:"):
                section = eq
                continue

            if line.startswith("# Sampling sweeps:"):
                section = sp
                continue

            if section is None:
                continue

            # End of a table
            if line == "#":
                section = None
                continue

            # Remove optional leading comment marker
            line = line.lstrip("#").strip()

            # Only read rows beginning with an integer
            if not re.match(r"^\d+\s+", line):
                continue

            fields = line.split()
            iteration = int(fields[0])
            walltime = float(fields[-1])

            section[iteration] = walltime

    # Missing iterations raise KeyError
    eq = [eq[i] for i in range(N_EQ + 1)]
    sp = [sp[i] for i in range(N_SP + 1)]

    return eq, sp


def analyze(path):
    try:
        eq, sp = read_walltimes(path)

        dt_eq = [b - a for a, b in zip(eq, eq[1:])]
        dt_sp = [b - a for a, b in zip(sp, sp[1:])]

        return [
            eq[0],
            statistics.mean(dt_eq),
            statistics.stdev(dt_eq),
            statistics.mean(dt_sp),
            statistics.stdev(dt_sp),
            sp[-1],
        ]

    except Exception:
        return [NAN] * 6


def main():
    root = Path(sys.argv[1])

    paths = sorted(root.rglob(".out"))

    prev_parent = None

    with open("walltime.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "id",
            "relpath",
            "t/init",
            "avg/eq",
            "std/eq",
            "avg/sp",
            "std/sp",
            "t/total",
        ])

        for i, path in enumerate(paths, start=1):
            folder = path.parent
            parent = folder.parent

            # Compact progress printing
            if parent != prev_parent:
                if prev_parent is not None:
                    print()
                print(f"{folder}/", end="", flush=True)
            else:
                print(f" {folder.name}/", end="", flush=True)

            prev_parent = parent

            # Read and analyze
            results = analyze(path)

            writer.writerow([
                i,
                os.path.relpath(path),
                *results,
            ])

            f.flush()

    print()
    print(f"Processed {len(paths)} files.")


if __name__ == "__main__":
    main()
