#!/bin/bash
#SBATCH --job-name=anlp_log_writing_v2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=03:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# v2 prompt: adds a worked BAD-vs-GOOD example targeting the recap-then-append
# failure mode found in the original run (mean novelty 18.1/10.1).
# journal_chains.json already built by the original run, no need to rebuild.

python steps/09_log_writing/code/generate_logs.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2_v2 \
    --batch_size 1 --prompt_version v2 \
    || echo "=== GENERATION FAILED: dictalm2_v2 (exit $?) ==="

python steps/09_log_writing/code/generate_logs.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b_v2 \
    --batch_size 1 --prompt_version v2 \
    || echo "=== GENERATION FAILED: qwen7b_v2 (exit $?) ==="

python steps/09_log_writing/code/eval_faithfulness.py \
    --judge_model Qwen/Qwen2.5-3B-Instruct --judge_short_name qwen3b \
    --targets dictalm2_v2 qwen7b_v2 \
    || echo "=== JUDGING FAILED (exit $?) ==="
