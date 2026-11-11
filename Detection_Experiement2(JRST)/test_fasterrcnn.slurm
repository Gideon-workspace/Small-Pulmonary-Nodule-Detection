#!/bin/bash

#SBATCH --job-name=FLipEval
#SBATCH --partition=bigbatch
#SBATCH --nodelist=mscluster67
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/faster_rcnn_test_%j.out
#SBATCH --error=logs/faster_rcnn_test_%j.err

# --------------------------------------------------
# Move to project directory
# --------------------------------------------------

cd "$SLURM_SUBMIT_DIR"

echo "======================================"
echo "Starting Faster R-CNN Testing"
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

# Update testing/test.py below if your python script has a different filename
PYTHONPATH="$SLURM_SUBMIT_DIR:$PYTHONPATH" \
/home-mscluster/mmabeloane/.conda/envs/mltest/bin/python -u testing/test.py

# --------------------------------------------------
# Finished
# --------------------------------------------------

echo "======================================"
echo "Testing Finished"
echo "Date: $(date)"
echo "======================================"
