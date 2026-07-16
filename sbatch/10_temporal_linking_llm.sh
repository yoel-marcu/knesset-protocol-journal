#!/bin/bash
#SBATCH --job-name=anlp_temporal_llm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=05:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

# Family D: extract structured matter-fingerprints per span (two models to compare).
python steps/10_temporal_linking/code/extract_fingerprints.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 4 \
    || echo "=== FAILED: extract dictalm2 (exit $?) ==="

python steps/10_temporal_linking/code/extract_fingerprints.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b --batch_size 4 \
    || echo "=== FAILED: extract qwen7b (exit $?) ==="

# Family D matching (CPU-fast, run here after extraction).
python steps/10_temporal_linking/code/fingerprint_linking.py --fp dictalm2 \
    || echo "=== FAILED: fp_linking dictalm2 (exit $?) ==="
python steps/10_temporal_linking/code/fingerprint_linking.py --fp qwen7b \
    || echo "=== FAILED: fp_linking qwen7b (exit $?) ==="

# Family E: verifier reading each candidate's FULL accumulated timeline.
python steps/10_temporal_linking/code/timeline_verify.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b --batch_size 2 \
    || echo "=== FAILED: timeline_verify qwen7b (exit $?) ==="
python steps/10_temporal_linking/code/timeline_verify.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 2 \
    || echo "=== FAILED: timeline_verify dictalm2 (exit $?) ==="
