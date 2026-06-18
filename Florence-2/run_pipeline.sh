#!/bin/bash
#SBATCH --job-name=vlm_pipeline
#SBATCH --gres=gpu:1        # We need the GPU for the first half (Inference)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G           # 32G to safely cover both scripts
#SBATCH --time=01:00:00     # Combined time (30m + 30m)
#SBATCH --mail-user= #email-address 
#SBATCH --mail-type=BEGIN,END,FAIL            
#SBATCH --output=pipeline_output_%j.log

echo "============================================="
echo "STEP 1: STARTING VLM INFERENCE"
echo "============================================="

# 1. Load Inference Modules and Environment
module purge
module load gcc arrow/21.0.0 python/3.11
source vlm_env/bin/activate

# 2. Run Inference Script
python test_vlm.py

# 3. Safely exit the inference environment
deactivate
echo "Inference complete. CSV generated."

echo "============================================="
echo "STEP 2: STARTING EVALUATION"
echo "============================================="

# 4. Load Evaluation Modules (matching your original eval script)
module purge
module load StdEnv/2023 gcc python/3.11 arrow/23.0.1

# 5. Activate Evaluation Environment
source # environment-path

# 6. Define variables and run Evaluation Script
CSV_INPUT="# saved-result"  # This should match the output CSV from the training script
RESULTS_DIR="# Result Analysis Path"

python evaluation_script.py "$CSV_INPUT" "$RESULTS_DIR"

# 7. Cleanup
deactivate
echo "============================================="
echo "PIPELINE COMPLETE. Check the '$RESULTS_DIR' folder."
echo "============================================="