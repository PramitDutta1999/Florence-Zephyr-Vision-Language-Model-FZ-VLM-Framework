# FZ-VLM: Florence-Zephyr Vision-Language Model Framework

A Two-Stage Florence-Zephyr Vision-Language Model (FZ-VLM) Framework for Pulmonary Nodule Characterization and Clinical Decision Making from Lung CT in a Post-Detection Framework.

This repository contains the source code developed by **Pramit Dutta** (University of Guelph) for the MASc thesis in Engineering (Collaborative Specialization in Artificial Intelligence). The framework automates the structured assessment of anatomical location, longest diameter, margin characteristics, and attenuation types of pulmonary nodules to support clinical risk assessment.

## ⚠️ Important Note on Data and Curation Scripts
> **Dataset Restrictions:** Due to the National Cancer Institute (NCI) Data Transfer Agreement restrictions, the original NLST CT images, patient-level data, and any protected datasets or trained weights are **not** publicly released with this repository. Researchers wishing to access the NLST dataset must apply directly through the NCI CDAS platform.
> 
> **Data Curation Code:** The data curation, prompt formation, and Hugging Face compatibility code included in this repository were built **specifically for curating my custom, restricted data**. While they provide a reference for converting DICOM to PNGs, formatting VQA, and generating Hugging Face `Dataset` objects, users will need to adapt these scripts to match the schema of their own datasets.

---

## 🛠️ Environment Setup

This project was developed and executed primarily on the Compute Canada / Digital Research Alliance of Canada **Rorqual cluster**, with some testing and configuration done in a **local multi-GPU environment**.

### 1. Rorqual Cluster Setup (SLURM)

To run the inference, training, and evaluation scripts on Rorqual, you must set up your virtual environment and load the appropriate modules. 

**One-time setup:**
```bash
# Load required modules
module purge
module load gcc arrow/21.0.0 python/3.11

# Create and activate the virtual environment
python -m venv ~/venvs/jlab
source ~/venvs/jlab/bin/activate

# Install dependencies
pip install --upgrade pip
pip install ipykernel torch torchvision torchaudio datasets transformers accelerate tqdm pandas pydicom pillow scikit-learn seaborn matplotlib
```
**Hugging Face Environment Configuration:**
Set your Hugging Face home and cache environments to your specific paths:

```bash
export HF_HOME=$SCRATCH/hf/cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_HUB_ENABLE_HF_TRANSFER=0

hf download microsoft/Florence-2-base-ft --revision refs/pr/6 --local-dir $SCRATCH/hf/models/florence2
```

### 2. Local Environment Setup

For the local machine environment, standard Python `venv` or `conda` setups apply. Multi-GPU training is supported via Hugging Face `accelerate`.

If using multiple GPUs locally, initialize your accelerate configuration (`accelerate_config.yaml`):

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
```

## 📂 Project Pipeline & SLURM Orchestration

### Stage 1: Data Curation & Preprocessing
*These scripts are tailored specifically to our data format.*

* **DICOM to PNG Conversion:** Maps expert-annotated DICOM slices and standardizes window level/width (WL: -700, WW: 1700).
* **VQA Format Generation:** Extracts parameters (location, diameter, attenuation, margin) and structures them into Question-Answer pairs (e.g., "What is the location of the nodule?").
* **Hugging Face Dataset Compilation:** Casts the CSV outputs into a formal `Dataset` object, reading image columns and utilizing the `datasets` library to save to disk for optimized data loading during training.

### Stage 2: VLM Training Orchestration
The vision-language models (e.g., Florence-2) are fine-tuned using `accelerate` via a dedicated SLURM configuration.

* **Resources:** The training script requests 2 GPUs per node, 8 CPUs per task, and an increased memory limit of 64G for full parameter tuning. The maximum time allocation is set to 10 hours (`10:00:00`).
* **Execution:**
* 
```bash
export HF_HOME=$PWD/hf_cache
export NCCL_IB_TIMEOUT=22
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

accelerate launch --config_file accelerate_config.yaml train_vlm.py
```

### Stage 3: Inference & Evaluation Workloads
The framework separates workloads based on processing needs, handled via distinct SLURM scripts:

**1. Unified VLM Inference and Evaluation**
This combined script manages the generation of textual descriptions and immediately processes the evaluation.
* **Resources:** Allocates 1 GPU, 4 CPUs, and 32G of memory to safely cover both the inference and evaluation scripts. The total combined time limit is 1 hour (`01:00:00`).
* **Execution Flow:** 1. Loads `gcc`, `arrow/21.0.0`, and `python/3.11`, activates the VLM environment, and runs `test_vlm.py`.
  2. Safely deactivates the inference environment.
  3. Loads the evaluation modules (`StdEnv/2023`, `gcc`, `python/3.11`, `arrow/23.0.1`), activates the evaluation environment, and executes `evaluation_script.py` against the generated CSV.

**2. LLM Evaluation Shell Script**
A standalone script specifically designed for processing Zephyr inferences.
* **Resources:** Allocates 1 GPU and 32G of memory. The time limit is set to 30 minutes (`0:30:00`).
* **Execution Flow:** Loads `python/3.11`, `gcc`, and `arrow/23.0.1`, activates the environment, and executes `inference.py`.

---

## ⚖️ Acknowledgements & Citation

If you use concepts or scripts from this repository in your research, please ensure your work complies with relevant ethical requirements and institutional policies. 

In publications and presentations arising from the use of the NLST data, **The National Cancer Institute (NCI)** must be acknowledged as the source of the data, and the CDAS project number must be referenced as required by the data-use agreement.
# Register environment for JupyterLab (if using the portal)
python -m ipykernel install --user --name jlab --display-name "Python (jlab)"
