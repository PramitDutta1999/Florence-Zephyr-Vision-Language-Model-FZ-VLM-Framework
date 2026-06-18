import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pandas as pd
from tqdm import tqdm
import os

# Path to your saved fine-tuned model
model_path = "# zephyr-7b-model-path"

def process_file(test_filename):
    """Runs inference on a single jsonl file and saves a separate CSV report."""
    if not os.path.exists(test_filename):
        print(f"Skipping {test_filename}: File not found.")
        return

    print(f"\n Starting Inference on: {test_filename}")
    
    # Load model and tokenizer for this file
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    with open(test_filename, 'r') as f:
        lines = f.readlines()

    data_results = []
    
    for line in tqdm(lines, desc=f"Evaluating {test_filename}"):
        item = json.loads(line)
        messages = item['messages']
        
        # Identify the task type (Description vs Recommendation)
        system_content = next(msg['content'] for msg in messages if msg['role'] == 'system')
        task_type = "Longitudinal" if "longitudinal" in system_content.lower() else ("Followup" if "follow-up" in system_content.lower() else "Description")
        
        # Prepare inputs
        user_prompt = [msg for msg in messages if msg['role'] != 'assistant']

        inputs = tokenizer.apply_chat_template(
            user_prompt,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=3000,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )

        predicted = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).strip()

        data_results.append({
            "Task": task_type,
            "Input_Attributes": user_prompt[-1]['content'].replace('\n', ' '),
            "Model_Prediction": predicted
        })

    # Save to a unique CSV file
    df = pd.DataFrame(data_results)
    output_csv = f"report_{test_filename.replace('.jsonl', '')}_baseline.csv"
    df.to_csv(output_csv, index=False)
    
    print(f" Finished {test_filename}")
    print(f" Saved report to: {output_csv}")

    # Clear memory to avoid OOM when starting the next file
    del model
    del tokenizer
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # The two files will be processed one after the other
    process_file(" # extracted-attribute-with-prompts.jsonl")