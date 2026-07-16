#!/bin/bash
#SBATCH --job-name=anlp_verifier_longsnippet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=09:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# Experiment: does the 100-word candidate snippet starve the model of the
# specific-matter signal? Re-derive snippets from full gold span text at 400
# words instead. Prompts get ~4x longer (11 segments/query x 400 words), so
# batch_size is reduced to avoid OOM and time budget is raised accordingly.

python steps/08_retrieve_verify/code/run_verifier.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2_long400 \
    --batch_size 1 --snippet_words 400 \
    || echo "=== MODEL FAILED: dictalm2_long400 (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b_long400 \
    --batch_size 1 --snippet_words 400 \
    || echo "=== MODEL FAILED: qwen7b_long400 (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b_long400 \
    --batch_size 2 --snippet_words 400 \
    || echo "=== MODEL FAILED: qwen3b_long400 (exit $?) ==="

python steps/08_retrieve_verify/code/eval_verifier.py
