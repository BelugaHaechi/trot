import argparse
import csv
import subprocess
import sys
import shlex
from pathlib import Path

"""
Launch Python script with arguments loaded from a given row in a CSV file.
- Arguments: script, csv, id
- CSV file headers: id, workdir, *args (need to match exactly with script)
- The CSV row with the matching id will be selected
- The workdir is auto created if didn't exist previously
- Name of args need to match exactly with what the script accepts
- Parses True/False entries: adds --no- prefix for False

Intended for workflows with SLURM arrays. Supply SLURM array task ID as id.
- Currently has memory monitoring code attached
"""

parser = argparse.ArgumentParser()
parser.add_argument("script")
parser.add_argument("csv")
parser.add_argument("id", type=int)
args = parser.parse_args()

with open(args.csv) as f:
    row = next(
        r for r in csv.DictReader(f)
        if int(r["id"]) == args.id
    )
row.pop("id")

workdir = Path(row.pop("workdir")).resolve()
workdir.mkdir(parents=True, exist_ok=True)
print(f"Working directory: {workdir}", flush=True)

script = Path(args.script).resolve()
cmd = [sys.executable, "-u", str(script)]
for key, value in row.items():
    value = value.strip()
    if not value:
        continue
    elif value.lower() == "true":
        cmd.append(f"--{key}")
    elif value.lower() == "false":
        cmd.append(f"--no-{key}")
    else:
        cmd.extend([f"--{key}", value])

print(shlex.join(cmd), flush=True)
monitor_script = Path(__file__).resolve().parent / "memwatch.py"

with open(workdir / ".out", "w") as out:
    proc = subprocess.Popen(
        cmd,
        cwd=workdir,
        stdout=out,
        stderr=subprocess.STDOUT, # combine out and err
    )

    print(f"Process PID: {proc.pid}", flush=True)

    monitor = subprocess.Popen(
        [sys.executable, "-u", str(monitor_script), str(proc.pid)],
        cwd=workdir,
    )

    returncode = proc.wait()
    monitor_returncode = monitor.wait()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)
