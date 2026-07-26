#!/bin/bash
#SBATCH --job-name=anlp_joint_journal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:1
#SBATCH --time=04:00:00
#SBATCH --output=/cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT/logs/slurm/%j.log

source /cs/labs/daphna/yoel.marcu2003/miniconda/etc/profile.d/conda.sh
conda activate anlp
cd /cs/labs/daphna/yoel.marcu2003/ANLP-PROJECT

python steps/12_joint_pipeline/code/build_segments.py

# RAW run (always available). Produces the full journal end-to-end.
python steps/12_joint_pipeline/code/run_journal.py \
    --segments outputs/segments_raw.json --tag raw --theta 0.34 \
    || echo "=== FAILED: journal raw ==="

# CANONICAL run (only if the Colab output has been dropped in place).
if [ -f steps/12_joint_pipeline/outputs/segments_canonical.json ]; then
  python steps/12_joint_pipeline/code/run_journal.py \
      --segments outputs/segments_canonical.json --tag canonical --theta 0.34 \
      || echo "=== FAILED: journal canonical ==="
fi
