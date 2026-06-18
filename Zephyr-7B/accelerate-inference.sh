#!/bin/bash
#SBATCH --job-name=llm_eval
#SBATCH --account=def-erangauk-ab
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=inference_separate.out

module purge
module load python/3.11 gcc arrow/23.0.1
source #environment-path

python inference.py