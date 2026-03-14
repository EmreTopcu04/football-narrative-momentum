import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from model import setup_football_model


def train_model(dataset_files, model_name="gpt2", epochs=1, lr=2e-5, accumulation_steps=16):
    """
    Fine-tunes a QLoRA GPT-2 sequence classifier on an explicit list of
    match_*_dataset.pt files (train split only — caller controls the split).
    """
    dataset_files = [f for f in dataset_files if os.path.exists(f)]

    if not dataset_files:
        print("No dataset files found. Run the pipeline first.")
        return

    print(f"Loading train dataset from {len(dataset_files)} match(es)...")

    all_input_ids, all_labels = [], []
    for f in dataset_files:
        data = torch.load(f)
        all_input_ids.append(data['input_ids'])
        all_labels.append(data['labels'])

    input_ids = torch.cat(all_input_ids, dim=0)   # [total_windows, 128]
    labels    = torch.cat(all_labels,    dim=0)   # [total_windows]

    home_pct = labels.float().mean().item() * 100
    print(f"Combined dataset : {input_ids.shape[0]:,} windows x {input_ids.shape[1]} tokens")
    print(f"Label balance    : {home_pct:.1f}% home / {100 - home_pct:.1f}% away momentum\n")

    dataset    = TensorDataset(input_ids, labels)
    # batch_size=1 keeps peak VRAM under 8 GB; accumulation_steps=16 simulates a larger batch
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model  = setup_football_model(model_name=model_name, num_labels=2)
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr)

    model.train()
    print(f"Training on {device} | {len(dataloader):,} steps/epoch | "
          f"accumulation={accumulation_steps} (effective batch={accumulation_steps})\n")

    for epoch in range(epochs):
        total_loss = 0.0
        bar = tqdm(enumerate(dataloader), total=len(dataloader),
                   desc=f"Epoch {epoch + 1}/{epochs}")

        for step, (batch_ids, batch_labels) in bar:
            batch_ids    = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(input_ids=batch_ids, labels=batch_labels)
            loss    = outputs.loss / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            bar.set_postfix({'avg_loss': f"{total_loss / (step + 1):.4f}"})

    print("\n=== Training Complete! ===")
    weights_dir = os.path.join(tensor_dir, "football_lora_weights")
    model.save_pretrained(weights_dir)
    print(f"LoRA adapter weights saved to: {weights_dir}")
