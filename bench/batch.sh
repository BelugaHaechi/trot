#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

# Linear algebra environment
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Launch MPI processes locally rather than delegating process creation to SLURM
# This allows MPI ranks to remain descendants of hchain.py, so that memwatch works
export HYDRA_LAUNCHER=fork

# Python environment
source /home/dsl44/anaconda3/etc/profile.d/conda.sh
conda activate trot
export PYTHONPATH="/home/dsl44/trot"

# PySCF environment
export PYSCF_MAX_MEMORY=$SLURM_MEM_PER_NODE
export PYSCF_TMPDIR=/scr/$USER/$SLURM_JOB_ID/$SLURM_ARRAY_TASK_ID

mkdir -p "$PYSCF_TMPDIR"
cleanup() {
    if [[ -n "$PYSCF_TMPDIR" && -d "$PYSCF_TMPDIR" ]]; then
        rm -rf "$PYSCF_TMPDIR"
    fi
}
trap cleanup EXIT

# Execution
echo "Node: $(hostname)"
echo "Task begins: $(date -Is)"

python -u bench/launch.py "$1" "$2" "$SLURM_ARRAY_TASK_ID"

echo "Task finished: $(date -Is)"
