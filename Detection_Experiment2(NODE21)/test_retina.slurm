#!/bin/bash

#SBATCH --job-name=RetinLip2
#SBATCH --partition=bigbatch
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/test_%j.out
#SBATCH --error=logs/test_%j.err

# --------------------------------------------------
# Move to project directory
# --------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

echo "======================================"
echo "Starting RetinaNet Testing"
echo "======================================"

echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Compute Node: $SLURMD_NODENAME"
echo "Working Directory: $PWD"

# --------------------------------------------------
# Load Anaconda
# --------------------------------------------------

echo "Loading Anaconda..."

source /etc/profile.d/lmod.sh

module load anaconda/anaconda-2025-12-2

echo "Anaconda Loaded"

# --------------------------------------------------
# Run testing
# --------------------------------------------------

echo "Starting testing..."

PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH" \
/home-mscluster/mmabeloane/.conda/envs/mltest/bin/python -u testing/test.py

# --------------------------------------------------
# Finished
# --------------------------------------------------

echo "======================================"
echo "Testing Finished"
echo "Date: $(date)"
echo "======================================"

