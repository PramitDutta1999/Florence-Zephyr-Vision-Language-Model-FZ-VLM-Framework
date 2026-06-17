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

# Register environment for JupyterLab (if using the portal)
python -m ipykernel install --user --name jlab --display-name "Python (jlab)"
