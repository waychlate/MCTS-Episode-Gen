#!/bin/bash
#SBATCH --job-name=mcts_regen_episodes
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=2:00:00
#SBATCH --output=logs/mcts_regen_%j.log

echo "Job Start"
date;hostname;pwd
echo "---"

module purge
module load python/3.11

cd /home/khek.do/MCTS-Episode-Gen/tools

source .venv/bin/activate

python -u regenerate_missing_episodes.py

echo "Job Finished."
