#!/bin/bash
#SBATCH --job-name=anlp_canon_eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=01:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# dataset is already built (CPU); embed raw+canonical (GPU) then evaluate (CPU).
python steps/11_canonicalization_linking/code/embed_variants.py
python steps/11_canonicalization_linking/code/eval_canonical.py
