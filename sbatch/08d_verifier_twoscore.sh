#!/bin/bash
#SBATCH --job-name=anlp_verifier_twoscore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=05:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# Experiment: separate topical_similarity (domain/ministry overlap) from
# specific_match_confidence (same exact matter) instead of one conflated
# same_confidence number. eval_verifier.py sweeps specific_match_confidence
# as theta (falls back to it automatically when same_confidence is absent).

python steps/08_retrieve_verify/code/run_verifier.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2_twoscore \
    --batch_size 2 --two_score \
    || echo "=== MODEL FAILED: dictalm2_twoscore (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b_twoscore \
    --batch_size 2 --two_score \
    || echo "=== MODEL FAILED: qwen7b_twoscore (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b_twoscore \
    --batch_size 4 --two_score \
    || echo "=== MODEL FAILED: qwen3b_twoscore (exit $?) ==="

python steps/08_retrieve_verify/code/eval_verifier.py
