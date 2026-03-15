import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import setup_football_model


def train_model(dataset_files, model_name="gpt2", epochs=3, lr=3e-5,
                accumulation_steps=16, weights_dir=None):
    """
    Fine-tunes a QLoRA GPT-2 sequence classifier with:
      - Class-weighted cross-entropy (handles label imbalance)
      - Cosine LR schedule with 5% warmup
      - Per-epoch checkpoint saves (never lose a full run again)
    """
    dataset_files = [f for f in dataset_files if os.path.exists(f)]

    if not dataset_files:
        print("No dataset files found. Run the pipeline first.")
        return

    if weights_dir is None:
        first_dir = os.path.dirname(os.path.abspath(dataset_files[0]))
        weights_dir = os.path.join(first_dir, "football_lora_weights")

    print(f"Loading train dataset from {len(dataset_files)} match(es)...")

    all_input_ids, all_labels = [], []
    for f in dataset_files:
        data = torch.load(f)
        all_input_ids.append(data['input_ids'])
        all_labels.append(data['labels'])

    input_ids = torch.cat(all_input_ids, dim=0)
    labels    = torch.cat(all_labels,    dim=0)

    home_pct = labels.float().mean().item() * 100
    print(f"Combined dataset : {input_ids.shape[0]:,} windows x {input_ids.shape[1]} tokens")
    print(f"Label balance    : {home_pct:.1f}% home / {100 - home_pct:.1f}% away momentum\n")

    dataset    = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    model  = setup_football_model(model_name=model_name, num_labels=2)
    device = next(model.parameters()).device

    # Class weights: inverse frequency so the minority class matters equally
    n_away = (labels == 0).sum().float()
    n_home = (labels == 1).sum().float()
    total  = n_away + n_home
    class_weights = torch.tensor([total / (2 * n_away),
                                  total / (2 * n_home)], device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Class weights    : away={class_weights[0]:.3f}  home={class_weights[1]:.3f}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Cosine schedule: 5% warmup then smooth decay to near-zero
    total_steps  = len(dataloader) * epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(optimizer,
                                                   num_warmup_steps=warmup_steps,
                                                   num_training_steps=total_steps)

    model.train()
    print(f"Training on {device} | {len(dataloader):,} steps/epoch x {epochs} epochs = "
          f"{total_steps:,} total steps")
    print(f"LR schedule      : warmup {warmup_steps:,} steps → cosine decay\n")

    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        bar = tqdm(enumerate(dataloader), total=len(dataloader),
                   desc=f"Epoch {epoch + 1}/{epochs}")

        for step, (batch_ids, batch_labels) in bar:
            batch_ids    = batch_ids.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(input_ids=batch_ids)
            logits  = outputs.logits.float()
            loss    = loss_fn(logits, batch_labels) / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            bar.set_postfix({
                'avg_loss': f"{total_loss / (step + 1):.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        avg = total_loss / len(dataloader)
        print(f"  Epoch {epoch + 1} avg loss: {avg:.4f}")

        # Save checkpoint after every epoch
        ckpt_dir = os.path.join(weights_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved: {ckpt_dir}\n")

    # Final save to the main weights dir
    model.save_pretrained(weights_dir)
    print(f"\n=== Training Complete ({epochs} epochs) ===")
    print(f"Final weights: {weights_dir}")
