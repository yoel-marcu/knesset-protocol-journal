#!/bin/bash
#SBATCH --job-name=anlp_verifier_compare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=04:30:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

python steps/08_retrieve_verify/code/build_verifier_inputs.py

python steps/08_retrieve_verify/code/run_verifier.py \
    --model dicta-il/dictalm2.0-instruct --short_name dictalm2 --batch_size 2 \
    || echo "=== MODEL FAILED: dictalm2 (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-7B-Instruct --short_name qwen7b --batch_size 2 \
    || echo "=== MODEL FAILED: qwen7b (exit $?) ==="

python steps/08_retrieve_verify/code/run_verifier.py \
    --model Qwen/Qwen2.5-3B-Instruct --short_name qwen3b --batch_size 4 \
    || echo "=== MODEL FAILED: qwen3b (exit $?) ==="

python steps/08_retrieve_verify/code/eval_verifier.py
