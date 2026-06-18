import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator, InitProcessGroupKwargs
from tqdm.auto import tqdm
from datetime import timedelta

# --- Dataset Class (from your notebook) ---
class DocVQADataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        example = self.data[idx]
        question = "<MedVQA>" + example['question']
        answer = example['answer']
        image = example['image'].convert("RGB") if example['image'].mode != "RGB" else example['image']
        return question, answer, image

def collate_fn(batch, processor):
    questions, answers, images = zip(*batch)
    inputs = processor(text=list(questions), images=list(images), return_tensors="pt", padding=True)
    labels = processor.tokenizer(text=list(answers), return_tensors="pt", padding=True, return_token_type_ids=False).input_ids
    # Important: Replace pad tokens with -100 so they are ignored by the loss function
    labels[labels == processor.tokenizer.pad_token_id] = -100
    return inputs, labels

def train():
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    # Initialize Accelerator
    accelerator = Accelerator(
        mixed_precision="fp16",
        kwargs_handlers=[timeout_kwargs]
    )
    
    # Paths
    model_path = "# pretrained-models-path"
    dataset_path = "# dataset-path"
    output_dir = "# saved-model-path"
    
    # Load Model & Processor
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    # Force use_fast=False to avoid the TokenizersBackend attribute error
    #processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    # Load Data
    data = load_from_disk(dataset_path)
    train_ds = DocVQADataset(data['train'])
    val_ds = DocVQADataset(data['validation'])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
    val_loader = DataLoader(val_ds, batch_size=2, collate_fn=lambda b: collate_fn(b, processor))

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-6)
    num_training_steps = 10 * len(train_loader)
    lr_scheduler = get_scheduler("cosine", optimizer, num_warmup_steps=2, num_training_steps=num_training_steps)

    # Prepare everything for multi-GPU
    model, optimizer, train_loader, val_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, lr_scheduler
    )

    # --- Training Loop (Fixed 15 Epochs) ---
    for epoch in range(10):
        model.train()
        total_train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", disable=not accelerator.is_local_main_process)
        
        for inputs, labels in train_pbar:
            outputs = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=labels)
            loss = outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            total_train_loss += loss.item()
            if accelerator.is_local_main_process:
                train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Validating Epoch {epoch}", disable=not accelerator.is_local_main_process):
                outputs = model(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], labels=labels)
                total_val_loss += outputs.loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        
        # Print progress after each epoch
        accelerator.print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- Final Save (Only at the end of Epoch 14) ---
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        accelerator.print(f"Training complete. Saving final model to {output_dir}...")
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    accelerator.print("Process group shut down successfully.")

if __name__ == "__main__":
    train()