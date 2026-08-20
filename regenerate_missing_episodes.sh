#!/bin/bash
#SBATCH --job-name=mcts_regen_episodes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --array=1-4
#SBATCH --time=2:00:00
#SBATCH --output=logs/mcts_regen_%A_%a.log

echo "Job Start"
date;hostname;pwd
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "---"

module purge
module load python/3.11

cd /home/khek.do/MCTS-Episode-Gen/

source .venv/bin/activate

# Runs the specific episode mapped to this SLURM_ARRAY_TASK_ID (1-4)
python -u regenerate_missing_episodes.py

echo "Job Finished."
