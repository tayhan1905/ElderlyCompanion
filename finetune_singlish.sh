#!/bin/bash
#SBATCH --job-name=singlish_finetune       # Job name
#SBATCH --output=logs/%x_%j.out            # Log file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err             # Error log
#SBATCH --time=24:00:00                    # Time limit (hh:mm:ss)
#SBATCH --partition=gpu                    # Partition/queue name
#SBATCH --gres=gpu:1                       # Number of GPUs
#SBATCH --cpus-per-task=8                  # CPU cores
#SBATCH --mem=32G                          # Memory
#SBATCH --mail-type=END,FAIL               # Email notifications (optional)
#SBATCH --mail-user=your_email@domain.com  # Your email (optional)

# -----------------------------
# Environment setup
# -----------------------------
echo "[$(date)] Job started on $HOSTNAME"

# Load modules (if your cluster uses modules)
module purge
module load cuda/11.8
module load python/3.10

# Activate virtual environment
source ~/envs/singlish/bin/activate

# -----------------------------
# Run training
# -----------------------------
echo "[$(date)] Starting fine-tuning job..."
python finetuning.py

echo "[$(date)] Job finished."
