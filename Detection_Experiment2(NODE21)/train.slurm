#!/bin/bash
#SBATCH --job-name=LIP2FAST
#SBATCH --partition=biggpu

#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

echo "======================================"
echo "Starting RetinaNet Training"
echo "======================================"
echo "Date                  : $(date)"
echo "Username              : $USER"
echo "Job ID                : $SLURM_JOB_ID"
echo "Job Name              : $SLURM_JOB_NAME"
echo "Compute Node          : $HOSTNAME"
echo "Working Directory     : $SLURM_SUBMIT_DIR"
echo "======================================"

# --------------------------------------------------
# Go to project directory
# --------------------------------------------------
cd "$SLURM_SUBMIT_DIR"


# --------------------------------------------------
# Load Anaconda and use mltest environment
# --------------------------------------------------

echo "Loading Anaconda..."
source /etc/profile.d/lmod.sh
module load anaconda/anaconda-2025-12-2

echo "Anaconda Loaded"

# --------------------------------------------------
# Start training
# --------------------------------------------------
echo "Starting python"

#export CUDA_LAUNCH_BLOCKING=1

PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH" \
/home-mscluster/mmabeloane/.conda/envs/mltest/bin/python -u training/train.py

echo "======================================"
echo "Training Finished"
echo "Date: $(date)"
echo "======================================"
