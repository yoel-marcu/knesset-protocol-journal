#!/bin/bash
#SBATCH --job-name=anlp_log_writing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=03:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

python steps/09_log_writing/code/build_journal_chains.py

# Generate: 2 models, deliberately not including qwen3b so it can judge both
# without self-preference bias.
python steps/09_log_writing/code/generate_logs.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 1 \
    || echo "=== GENERATION FAILED: dictalm2 (exit $?) ==="

python steps/09_log_writing/code/generate_logs.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b --batch_size 1 \
    || echo "=== GENERATION FAILED: qwen7b (exit $?) ==="

# Judge with qwen3b (independent of both generators above).
python steps/09_log_writing/code/eval_faithfulness.py \
    --judge_model Qwen/Qwen2.5-3B-Instruct --judge_short_name qwen3b \
    --targets dictalm2 qwen7b \
    || echo "=== JUDGING FAILED (exit $?) ==="
