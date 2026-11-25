import argparse
import csv
import subprocess
import sys
import shlex
from pathlib import Path

"""
Submit a SLURM job array using experiment specifications from a CSV file as defined in launch.py
- Arguments: script, csv, partition, max_simul_tasks
- Launches jobs for every data row in the CSV
- Default job name: CSV filename without extension
- By default, allocates all CPU and memory in the node, read from partitions.csv
- Default time limit: 1 day
- SLURM logs are saved to .slurm_logs/job_name/JOBID-ARRAYID.log

Calls batch.sh to configure the execution environment and invoke launch.py.

Run from the repository root:
python bench/submit.py bench/hchain.py jobs/JOBNAME.csv PARTITION 4 --dry-run
"""

with open("bench/partitions.csv") as f:
    partitions = {row["name"]: row for row in csv.DictReader(f)}

parser = argparse.ArgumentParser()
parser.add_argument("script")
parser.add_argument("csv")
parser.add_argument("partition", choices=partitions)
parser.add_argument("max_simul_tasks", type=int)

parser.add_argument("--job-name", default=None)
parser.add_argument("--cpus-per-task", type=int, default=None)
parser.add_argument("--mem", default=None)
parser.add_argument("--time", default="1-00:00:00")
parser.add_argument("--array", default=None)

parser.add_argument("--dry-run", action="store_true") # only assemble sbatch cmd

args = parser.parse_args()

job_name = args.job_name or Path(args.csv).stem
logdir = Path("jobs/.slurm_logs") / job_name

partition = partitions[args.partition]
n_cpus = args.cpus_per_task or int(partition["n_cpus"])
mem = args.mem or f'{partition["mem_slurm_mb"]}M'

with open(args.csv) as f:
    n_tasks = sum(1 for _ in csv.DictReader(f))
array = args.array or f"1-{n_tasks}%{args.max_simul_tasks}"

cmd = [
    "sbatch",
    f"--job-name={job_name}",
    f"--partition={args.partition}",
    f"--array={array}",
    f"--cpus-per-task={n_cpus}",
    f"--mem={mem}",
    f"--time={args.time}",
    f"--output={logdir}/%A-%a.log",
    "bench/batch.sh",
    args.script,
    args.csv,
]

print(shlex.join(cmd), flush=True)

if not args.dry_run:
    logdir.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
