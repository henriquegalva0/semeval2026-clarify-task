!/bin/bash

SBATCH --job-name=treino_modelo_SemEval2026task_CLARITY
SBATCH --output=saida_%j.log
SBATCH --error=erro_%j.log
SBATCH --time=01:00:00
SBATCH --partition=h100n3
SBATCH --gres=gpu:1
SBATCH --cpus-per-task=4
SBATCH --mem=16G

python3 encoder_train.py
#python3 encoder_inference.py

#sbatch training_job.sh