# FZ-VLM: Florence-Zephyr Vision Language Model for Pulmonary Nodule Characterization

This repository contains the code for my M.A.Sc. thesis project:

**A Two Stage Florence-Zephyr Vision Language Model for Pulmonary Nodule Characterization and Clinical Decision Making from Lung CT in a Post Detection Framework**

The project develops a two-stage medical vision-language framework for lung CT interpretation.

The framework has two main stages:

1. **Stage 1: Florence-2 Vision-Language Model**

   * Extracts structured radiological attributes from lung CT slices.
   * The extracted attributes include:

     * anatomical location
     * longest diameter
     * margin characteristics
     * attenuation type

2. **Stage 2: Zephyr-7B Language Model**

   * Uses the extracted attributes from Stage 1.
   * Generates structured clinical outputs, such as:

     * nodule description
     * follow-up recommendation
     * longitudinal analysis

This project is designed as a post-detection framework. It assumes that the nodule has already been detected and the relevant CT slice has been selected.

---

## Important Data and Model Availability Notice

The dataset curation process used in this project is specific to my thesis work.

The original CT data and related information were used under an agreement with the National Cancer Institute (NCI). Because of this data use agreement, the curated dataset cannot be shared publicly in this GitHub repository.

Also, the trained model weights cannot be shared publicly because the models were trained using this restricted dataset.

Therefore, this repository only provides:

* source code
* training scripts
* inference scripts
* evaluation scripts
* environment setup instructions
* placeholder paths for users to set up their own local data and model files

This repository does **not** include:

* original CT images
* curated VQA dataset
* patient-level metadata
* trained Florence-2 weights
* generated patient-level outputs from restricted data

Users who want to reproduce the full experiment must obtain proper data access and prepare their own dataset following their approved data use agreement.

---

## Repository Structure

```text
.
├── Data_Curation/
│   └── converting_dicom.ipynb
│
├── Florence2/
│   ├── accelerate_config.yaml
│   ├── accelerate-train.sh
│   ├── evaluation_script.py
│   ├── run_pipeline.sh
│   ├── test_vlm.py
│   └── train_vlm.py
│
├── HuggingFace_Compatibility/
│   ├── dataset_format.ipynb
│   └── vqa_format.ipynb
│
├── Integration/
│   ├── combined_dataset.ipynb
│   └── data_prompt_formation.ipynb
│
├── Zephyr7B/
│   ├── accelerate-inference.sh
│   └── inference.py
│
└── README.md
```

The repository is organized into five main folders. The `Data_Curation` folder contains the notebook used to convert and prepare DICOM images. The `Florence2` folder contains the training, testing, pipeline, and evaluation scripts for the Florence-2 vision-language model. The `HuggingFace_Compatibility` folder contains notebooks for preparing the dataset in a Hugging Face compatible format and VQA format. The `Integration` folder contains notebooks used to combine the extracted attributes and form prompts for the second stage. The `Zephyr7B` folder contains the inference scripts for the Zephyr-7B language model.

---

## Project Files

### Florence-2 Stage

The Florence-2 stage is used for visual attribute extraction from CT images.

Main files:

```text
train_vlm.py
test_vlm.py
accelerate-train.sh
run_pipeline.sh
accelerate_config.yaml
evaluation_script.py
```

### Zephyr-7B Stage

The Zephyr-7B stage is used for language based clinical interpretation.

Main files:

```text
inference.py
accelerate-inference.sh
```

### Dataset Preparation Notebooks

The notebooks are used for project specific data preparation.

```text
converting_dicom.ipynb
dataset_format.ipynb
vqa_format.ipynb
combined_dataset.ipynb
data_prompt_formation.ipynb
```

These notebooks are included to show the data preparation workflow. However, the actual data files are not included because of the NCI data use agreement.

---

## Stage 1: Florence-2 VLM

The Florence-2 model is fine-tuned using a structured Medical VQA format.

Each training example has:

```text
image
question
answer
```

Example question:

```text
<MedVQA>What is the diameter of the nodule?
```

Example answer:

```text
[5.0]
```

The model learns to answer focused clinical questions from the CT image.

---

## Stage 2: Zephyr-7B LLM

The Zephyr-7B model receives structured attributes extracted from Florence-2.

Example input:

```text
Nodule characteristics:
location: Left Upper Lobe
longest diameter: 5 mm
margin: Smooth
attenuation: Soft Tissue

Based on the nodule characteristics provided, describe the nodule.
```

The model then generates a clinical-style response based only on the structured attributes.

---

## Environment Setup Overview

This repository has two different setup needs:

1. **Florence-2 setup**

   * used for VLM training, testing, and evaluation
   * uses CT image-question-answer triplets
   * uses `AutoProcessor` and `AutoModelForCausalLM`

2. **Zephyr-7B setup**

   * used for LLM-based clinical interpretation
   * uses structured text prompts generated from extracted nodule attributes
   * uses `AutoTokenizer` and `AutoModelForCausalLM`

The exact environment may depend on the computing system. The instructions below are written for an HPC/cluster environment using Python scripts and modules. JupyterLab is not required for the main workflow.

---

# Florence-2 Setup

The Florence-2 setup is used for Stage 1 of the framework. This stage trains and tests the vision-language model for radiological attribute extraction from CT images.

---

## 1. Load required modules

First, remove currently loaded modules:

```bash
module purge
```

Load the required modules:

```bash
module load python/3.11 gcc arrow/23.0.1
```

Check if `pyarrow` is available:

```bash
pip list | grep pyarrow
```

To check available Arrow versions:

```bash
module spider arrow
```

If there is any problem with the module environment, load the standard environment:

```bash
module load StdEnv/2023
```

---

## 2. Create the Florence-2 virtual environment

Create a virtual environment with system site packages:

```bash
virtualenv --system-site-packages vlm_env
```

Activate the environment:

```bash
source /path/to/your/vlm_env/bin/activate
```

Example:

```bash
source /home/pdutta/links/projects/def-erangauk-ab/pdutta/vlm_training/vlm_env/bin/activate
```

Use your own path when running this project.

---

## 3. Install required packages

Install the required packages:

```bash
pip install --no-index datasets==2.17.0 transformers==4.36.0 accelerate==0.25.0 trl==0.7.11
```

You may also need:

```bash
pip install torch torchvision torchaudio tqdm pillow pandas numpy matplotlib seaborn scikit-learn
```

On an HPC system, some packages may already be available through the loaded modules.

---

## 4. Download the Florence-2 model

The first step is to download the model. On a login node, run the following commands to download the model to your project directory using the `git-lfs` module:

```bash
cd projects/account-name/user-name/

module load git-lfs

git clone https://huggingface.co/microsoft/Florence-2-large
```

You can also use another Florence-2 version if needed, such as:

```bash
git clone https://huggingface.co/microsoft/Florence-2-base
```

---

## 5. Set the Florence-2 paths

Before running training, update the paths in `Florence2/train_vlm.py`.

```python
model_path = "# pretrained-models-path"
dataset_path = "# dataset-path"
output_dir = "# saved-model-path"
```

Example:

```python
model_path = "/path/to/Florence-2-large"
dataset_path = "/path/to/processed_medvqa_dataset"
output_dir = "/path/to/save/fine_tuned_florence"
```

Before running testing, update the paths in `Florence2/test_vlm.py`.

```python
model_path = "# saved-model-path"
dataset_path = "# dataset-path"
csv_filename = "# saved-result"
```

Example:

```python
model_path = "/path/to/save/fine_tuned_florence"
dataset_path = "/path/to/processed_medvqa_dataset"
csv_filename = "/path/to/results/florence_test_results.csv"
```

---

## 6. Run Florence-2 training

Go to the Florence-2 folder:

```bash
cd Florence2
```

To run training directly:

```bash
python train_vlm.py
```

To run training with SLURM:

```bash
sbatch accelerate-train.sh
```

The training script uses:

```text
accelerate_config.yaml
```

The current Accelerate setup uses:

```text
MULTI_GPU
fp16 mixed precision
2 processes
1 machine
```

---

## 7. Run Florence-2 testing

To run testing directly:

```bash
python test_vlm.py
```

The output is saved as a CSV file with:

```text
pid
question
image
ground_truth
model_output
```

---

## 8. Run Florence-2 evaluation

After testing, run:

```bash
python evaluation_script.py /path/to/florence_test_results.csv /path/to/result_folder
```

The evaluation script saves:

```text
medvqa_results_cleaned.csv
diameter_metrics.txt
diameter_tolerance_plot.png
confusion_matrix_location.png
confusion_matrix_attenuation.png
confusion_matrix_margin.png
```

---

# Zephyr-7B Setup

The Zephyr-7B setup is used for Stage 2 of the framework. This stage uses structured radiological attributes to generate clinical-style outputs, such as nodule description, follow-up recommendation, and longitudinal analysis.

---

## 1. Load required modules

First, remove currently loaded modules:

```bash
module purge
```

Load the required modules:

```bash
module load python/3.11 gcc arrow/23.0.1
```

Check if the Arrow version is okay:

```bash
pip list | grep pyarrow
```

Check the most recent Arrow module:

```bash
module spider arrow
```

If there is any problem, load:

```bash
module load StdEnv/2023
```

---

## 2. Create the Zephyr-7B virtual environment

Create the virtual environment with system site packages:

```bash
virtualenv --system-site-packages llm_env
```

Activate the environment:

```bash
source /path/to/your/llm_env/bin/activate
```

Example:

```bash
source /home/pdutta/links/projects/def-erangauk-ab/pdutta/llm_training/llm_env/bin/activate
```

Use your own environment path when running this project.

---

## 3. Install required packages

Updated package installation:

```bash
pip install --no-index datasets==2.17.0 transformers==4.36.0 accelerate==0.25.0 trl==0.7.11
```

Older working version:

```bash
pip install --no-index datasets==2.16.0 transformers==4.36.0 accelerate==0.25.0 trl==0.7.4
```

Use the updated version unless there is a compatibility problem.

---

## 4. Download the Zephyr-7B model

The first step is to download the model. On a login node, run the following commands to download the model to your project directory using the `git-lfs` module.

This model download step should be done outside the virtual environment.

If an environment is already active, deactivate it first:

```bash
deactivate
```

Then download the model:

```bash
cd projects/account-name/user-name/

module load git-lfs

git clone https://huggingface.co/HuggingFaceH4/zephyr-7b-beta
```

---

## 5. Set the Zephyr-7B paths

Update the model path in `Zephyr7B/inference.py`.

```python
model_path = "# zephyr-7b-model-path"
```

Example:

```python
model_path = "/path/to/zephyr-7b-beta"
```

Also update the input JSONL file path:

```python
process_file(" # extracted-attribute-with-prompts.jsonl")
```

Example:

```python
process_file("/path/to/extracted_attribute_with_prompts.jsonl")
```

---

## 6. Run Zephyr-7B inference

Go to the Zephyr-7B folder:

```bash
cd Zephyr7B
```

Run directly:

```bash
python inference.py
```

Or submit the SLURM job:

```bash
sbatch accelerate-inference.sh
```

The script saves a CSV report such as:

```text
report_extracted_attribute_with_prompts_baseline.csv
```

---

# Running the Full Pipeline

The full pipeline script runs Florence-2 inference first and then runs the evaluation script.

```bash
cd Florence2

sbatch run_pipeline.sh
```

Before running, update the paths inside `Florence2/run_pipeline.sh`.

```bash
CSV_INPUT="# saved-result"
RESULTS_DIR="# Result Analysis Path"
```

Example:

```bash
CSV_INPUT="/path/to/florence_test_results.csv"
RESULTS_DIR="/path/to/result_analysis"
```

Also update the environment path:

```bash
source # environment-path
```

---

# Input and Output Formats

## Florence-2 Dataset Format

The Florence-2 dataset should be saved using the Hugging Face `datasets` format.

Expected splits:

```text
train
validation
test
```

Expected fields:

```text
question
answer
image
```

Example:

```python
{
    "question": "What is the location of the nodule?",
    "answer": "Right Upper Lobe",
    "image": PIL.Image
}
```

---

## Florence-2 Output CSV

The testing script saves:

```text
pid,question,image,ground_truth,model_output
```

---

## Zephyr-7B Input Format

The Zephyr-7B inference script expects a `.jsonl` file with chat-style messages.

Example structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a medical assistant that describes pulmonary nodules from structured attributes."
    },
    {
      "role": "user",
      "content": "Nodule characteristics: location: Left Upper Lobe longest diameter: 5 mm margin: Smooth attenuation: Soft Tissue. Based on the nodule characteristics provided, describe the nodule."
    }
  ]
}
```

---

## Zephyr-7B Output CSV

The Zephyr-7B inference script saves:

```text
Task
Input_Attributes
Model_Prediction
```


# Notes for Users

This repository is shared for research transparency and code availability.

Because the dataset and trained weights are restricted, users must:

* prepare their own approved dataset
* update all placeholder paths
* download the base models from Hugging Face
* run the training and inference scripts in their own environment

This code is not a clinical product. It is not approved for clinical diagnosis or patient care. The outputs must be reviewed by qualified clinical experts.

---

# Author

Pramit Dutta
University of Guelph

---

License

This repository is released under the MIT License for the source code only.

The license does not apply to:

the original medical imaging dataset
curated datasets derived from restricted data
patient-level metadata
trained model weights
generated outputs based on restricted data

The dataset used in this project was accessed under a data use agreement with the National Cancer Institute (NCI). Therefore, the dataset and trained model weights are not shared in this repository.

---

# Acknowledgements

This work was completed as part of a M.A.Sc. thesis project at the University of Guelph.

The author sincerely thanks Dr. Eranga Ukwatta for his supervision, guidance, and support throughout this research. The author also thanks the members of the **AI-Driven Medical Imaging & Diagnostic Lab** for their support, feedback, discussions, and encouragement during the development of this project.

The author is especially grateful to Dr. Jenita Manokaran for research support, dataset-related guidance, and helpful discussions throughout the project, and to Dr. Richa Mittal for clinical and radiological guidance during the evaluation of the framework.

The author acknowledges the National Cancer Institute (NCI) for providing access to the National Lung Screening Trial (NLST) data used in this research under an approved data use agreement. The author also acknowledges the **Natural Sciences and Engineering Research Council of Canada (NSERC)** and the **Digital Research Alliance of Canada** for supporting this research through funding and computational resources.

---

# Citation

If you use this code, please cite the related thesis or publication when available.

```text
Pramit Dutta. A Two Stage Florence-Zephyr Vision Language Model for Pulmonary Nodule Characterization and Clinical Decision Making from Lung CT in a Post Detection Framework. M.A.Sc. Thesis, University of Guelph, 2026 (In Preparation).
```
