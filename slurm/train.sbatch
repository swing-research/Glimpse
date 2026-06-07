#!/bin/bash
#SBATCH --job-name=glimpse
#SBATCH --output=experiments/%x/output.log
#SBATCH --error=experiments/%x/error.log
#SBATCH --cluster=gmerlin6
#SBATCH --partition=gwendolen
#SBATCH --account=gwendolen
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=disable-binding
#SBATCH --mem=25G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#
# Train GLIMPSE on the PSI Merlin GPU cluster (gmerlin6 / gwendolen, A100).
#
# Usage (submit from the repository root):
#   mkdir -p experiments/<exp>            # <exp> = <image_size>_<n_angles>_<exp_desc>
#   sbatch --job-name=<exp> slurm/train.sbatch [configs/lodopab.yaml]
#
# Set --job-name to the experiment directory name so this job's stdout/stderr
# (output.log / error.log) land inside experiments/<exp>/ next to the checkpoint
# (glimpse.pt), Loss.jpg, results.txt and the Results/ images written by train.py.

set -e
CONFIG="${1:-configs/lodopab.yaml}"

source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
conda activate /data/user/khoras_a/conda_envs/odl
cd "${SLURM_SUBMIT_DIR:-.}"

echo "host=$(hostname) | job=$SLURM_JOB_ID | config=$CONFIG | CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
python3 train.py --config "$CONFIG" --num-workers 6
