#!/bin/bash
#SBATCH --job-name=mcts_episode_gen
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=khek.do@ufl.edu
#SBATCH --nodes=1
#SBATCH --ntasks=10          # 1 instance of the execution script
#SBATCH --cpus-per-task=1   # ...allocated with 10 physical cores!
#SBATCH --mem=20gb
#SBATCH --time=12:00:00
#SBATCH --output=multiprocess_%j.log # Standard output and error log
echo "Job Start"
date;hostname;pwd
echo "---"

module purge
module load python/3.11

cd /home/khek.do/MCTS-Episode-Gen/

python -m venv .venv

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python -u episode_gen.py     # Python uses its own internal pooling to fill the 10 cores

echo "Job End"
date
echo "---"