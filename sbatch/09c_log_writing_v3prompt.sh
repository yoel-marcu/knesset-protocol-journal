#!/bin/bash
#SBATCH --job-name=anlp_log_writing_v3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=03:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# v3 prompt: same anti-recap goal as v2, but the worked example is now fully
# fictional (invented names/numbers) plus an explicit anti-copying instruction --
# v2's real-corpus example caused 36% verbatim copying on dictalm2.

python steps/09_log_writing/code/generate_logs.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2_v3 \
    --batch_size 1 --prompt_version v3 \
    || echo "=== GENERATION FAILED: dictalm2_v3 (exit $?) ==="

python steps/09_log_writing/code/generate_logs.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b_v3 \
    --batch_size 1 --prompt_version v3 \
    || echo "=== GENERATION FAILED: qwen7b_v3 (exit $?) ==="

python steps/09_log_writing/code/eval_faithfulness.py \
    --judge_model Qwen/Qwen2.5-3B-Instruct --judge_short_name qwen3b \
    --targets dictalm2_v3 qwen7b_v3 \
    || echo "=== JUDGING FAILED (exit $?) ==="
