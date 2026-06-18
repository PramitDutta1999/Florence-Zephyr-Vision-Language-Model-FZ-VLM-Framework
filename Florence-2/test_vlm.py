import os
import csv
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm.auto import tqdm
from PIL import Image

def run_testing():
    # 1. Setup paths and device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "# saved-model-path"
    dataset_path = "# dataset-path"
    
    # 2. Load Model and Processor
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device).eval()
    # use_fast=False is critical to avoid the AttributeError you saw earlier
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

    # 3. Load Dataset
    data = load_from_disk(dataset_path)
    test_data = data['test']
    
    results = []

    # 4. Helper Function for Inference
    def run_example(task_prompt, text_input, image):
        prompt = task_prompt + text_input
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, 
            task=task_prompt, 
            image_size=(image.width, image.height)
        )
        return parsed_answer

    # 5. Run Loop (using your specific range)
    print(f"Starting inference on {len(test_data)} samples...")
    
    # Using len(test_data) ensures we never go out of bounds
    for idx in tqdm(range(len(test_data))):
        pid = test_data[idx]['pid']
        question = test_data[idx]['question']
        image = test_data[idx]['image']
        answers = test_data[idx]['answer']

        # Task prompt must match what you used in training
        model_output = run_example("<MedVQA>", question, image)

        results.append({
            "pid": pid,
            "question": question,
            "image": f"sample_{idx}", # Storing the full image object in CSV is messy
            "ground_truth": ", ".join(answers) if isinstance(answers, list) else answers,
            "model_output": model_output
        })

    # 6. Save to CSV
    csv_filename = "# saved-result"
    with open(csv_filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["pid", "question", "image", "ground_truth", "model_output"])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Testing complete. Results saved to {csv_filename}")

if __name__ == "__main__":
    run_testing()