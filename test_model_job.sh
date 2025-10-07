#!/bin/bash

#SBATCH --job-name=test_AlBERT_train

#SBATCH --output=saida_%j.log

#SBATCH --error=erro_%j.log

#SBATCH --time=01:00:00

#SBATCH --partition=h100n3

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16G


python3 encoder_inference.py