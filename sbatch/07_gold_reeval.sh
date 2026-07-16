#!/bin/bash
#SBATCH --job-name=anlp_gold_reeval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

python steps/07_gold_eval/code/embed_gold.py
python steps/07_gold_eval/code/gold_reeval.py
