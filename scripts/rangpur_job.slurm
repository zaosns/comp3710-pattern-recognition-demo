#!/bin/bash -l
#SBATCH --job-name=comp3710-gpu
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --qos=comp3710
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

if (( $# == 0 )); then
    echo "Usage: sbatch scripts/rangpur_job.slurm <python-file> [arguments...]" >&2
    exit 2
fi

cd "$SLURM_SUBMIT_DIR"

PYTHON="/home/Student/$(id -un)/miniconda3/envs/torch/bin/python"
export PYTHONNOUSERSITE=1

echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PYTHON" "$@"
