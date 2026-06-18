import os
import sys
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# --- CONFIGURATION ---
# Strictly read arguments from the .sh execution
INPUT_FILE = sys.argv[1]    # First argument: path to CSV
OUTPUT_FOLDER = sys.argv[2] # Second argument: path to output folder

CLEANED_FILE = os.path.join(OUTPUT_FOLDER, "medvqa_results_cleaned.csv")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# --- 01. YOUR WORKING CLEANING FUNCTIONS ---
def clean_ground_truth(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    if val.startswith("[") and val.endswith("]"):
        val = val[1:-1]
    try:
        return float(val)
    except ValueError:
        return val

def clean_model_output(val):
    if pd.isna(val):
        return val
    val = str(val).strip()
    # Updated to match your working version with <MedVQA>
    if val.startswith("{") and "<MedVQA>" in val:
        try:
            parsed = ast.literal_eval(val)
            val = str(parsed.get("<MedVQA>", "")).strip()
        except Exception:
            pass
    if val.startswith("[") and val.endswith("]"):
        val = val[1:-1]
    try:
        return float(val)
    except ValueError:
        return val

# --- 02. CATEGORICAL EVALUATION (Confusion Matrices) ---
def run_categorical_eval(df):
    QUESTION_MAP = {
        "What is the location of the nodule?": "Location",
        "What is the attenuation of the nodule?": "Attenuation",
        "What is the margin of the nodule?": "Margin"
    }
    
    for q_text, title in QUESTION_MAP.items():
        sub = df[df["question"] == q_text].dropna(subset=["ground_truth_clean", "model_output_clean"])
        y_true = sub["ground_truth_clean"].astype(str).str.strip()
        y_pred = sub["model_output_clean"].astype(str).str.strip()
        
        labels = sorted(set(y_true) | set(y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt="d", cmap="viridis", linewidths=0.5)
        plt.title(f"Confusion Matrix – {title}")
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"confusion_matrix_{title.lower()}.png"), dpi=300)
        plt.close()

# --- 03-05. DIAMETER EVALUATION (Robust with MAE/Median/Plots) ---
def run_diameter_eval(df):
    sub = df[df["question"] == "What is the diameter of the nodule?"].copy()
    
    # Force conversion to numeric, turning empty strings or text into NaN
    sub["gt_num"] = pd.to_numeric(sub["ground_truth_clean"], errors='coerce')
    sub["pred_num"] = pd.to_numeric(sub["model_output_clean"], errors='coerce')
    
    # Drop rows that are not numbers to avoid crashes
    valid_sub = sub.dropna(subset=["gt_num", "pred_num"])
    
    y_true = valid_sub["gt_num"]
    y_pred = valid_sub["pred_num"]
    abs_error = np.abs(y_pred - y_true)

    # Metrics
    mae = abs_error.mean()
    std = abs_error.std(ddof=1)
    median_ae = np.median(abs_error)

    # Tolerance calculation
    thresholds = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    accuracies = [(abs_error <= t).mean() * 100 for t in thresholds]
    
    # --- UPDATED: Save detailed report to txt file ---
    with open(os.path.join(OUTPUT_FOLDER, "diameter_metrics.txt"), "w") as f:
        f.write(f"Diameter Evaluation (N={len(valid_sub)})\n")
        f.write("-" * 30 + "\n")
        f.write(f"MAE (mm): {mae:.2f}\n")
        f.write(f"Std Dev (mm): {std:.2f}\n")
        f.write(f"Median AE (mm): {median_ae:.2f}\n")
        f.write("-" * 30 + "\n")
        f.write("Accuracy Thresholds:\n")
        f.write(f"Exact match (0mm): {accuracies[0]:.2f}%\n")
        f.write(f"Within ±1 mm:      {accuracies[1]:.2f}%\n")
        f.write(f"Within ±2 mm:      {accuracies[2]:.2f}%\n")
        f.write(f"Within ±3 mm:      {accuracies[3]:.2f}%\n")
        f.write(f"Within ±4 mm:      {accuracies[4]:.2f}%\n")
        f.write(f"Within ±5 mm:      {accuracies[5]:.2f}%\n")

    # Plot Accuracy Tolerance with trend line
    labels = ['Exact match\n(0 mm)', '±1 mm', '±2 mm', '±3 mm', '±4 mm', '±5 mm']
    plt.figure(figsize=(7, 5))
    plt.bar(range(len(labels)), accuracies, width=0.6, zorder=2, color='skyblue')
    plt.plot(range(len(labels)), accuracies, color='black', marker='o', linewidth=1.8, markersize=5, zorder=3)
    
    for i, val in enumerate(accuracies):
        plt.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=9)
        
    plt.xticks(range(len(labels)), labels)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.grid(axis='y', linestyle='--', alpha=0.4, zorder=1)
    plt.title("Diameter Prediction Error Tolerance")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_FOLDER}/diameter_tolerance_plot.png", dpi=300)
    plt.close()

def main():
    print("Loading and cleaning data...")
    df = pd.read_csv(INPUT_FILE)
    df["ground_truth_clean"] = df["ground_truth"].apply(clean_ground_truth)
    df["model_output_clean"] = df["model_output"].apply(clean_model_output)
    df.to_csv(CLEANED_FILE, index=False)
    
    print("Generating Categorical Confusion Matrices...")
    run_categorical_eval(df)
    
    print("Generating Diameter Evaluation...")
    run_diameter_eval(df)
    
    print(f"Success! All results are in the '{OUTPUT_FOLDER}' folder.")

if __name__ == "__main__":
    main()