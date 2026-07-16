#!/bin/bash
#SBATCH --job-name=anlp_verifier_fewshot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=05:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# Experiment: 4 worked examples instead of 2 (2 new: a fresh positive
# continuation and a hard negative isolating an even closer confound --
# same presenter + same funding template, different specific matter).

python steps/08_retrieve_verify/code/run_verifier.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2_fewshot \
    --batch_size 2 --fewshot extended \
    || echo "=== MODEL FAILED: dictalm2_fewshot (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b_fewshot \
    --batch_size 2 --fewshot extended \
    || echo "=== MODEL FAILED: qwen7b_fewshot (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b_fewshot \
    --batch_size 4 --fewshot extended \
    || echo "=== MODEL FAILED: qwen3b_fewshot (exit $?) ==="

python steps/08_retrieve_verify/code/eval_verifier.py
