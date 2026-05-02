"""
Training Module

This module manages the fine-tuning process of the large language model for sequence classification.
It handles building the datasets, computing dynamic class weights to address label imbalances,
configuring hardware optimizations (gradient accumulation, lazy data loading), and 
orchestrating the standard PyTorch training loop over the specified epochs.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm

from model import setup_football_model


class MatchDataset(Dataset):
    """
    Lazy-loading dataset for a single match_*_dataset.pt file.
    Each match stays as its own Dataset object; ConcatDataset stitches them
    together so we never torch.cat 2000 matches into one giant RAM tensor.
    """
    def __init__(self, path):
        data = torch.load(path, weights_only=True)
        self.input_ids = data['input_ids']   # [N, context_length]
        self.labels    = data['labels']       # [N]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx]


def train_model(dataset_files, model_name="gpt2", epochs=3, lr=3e-5,
                batch_size=1, accumulation_steps=16, weights_dir=None,
                num_workers=4, max_samples=300, num_labels=3):
    """
    Fine-tunes a QLoRA sequence classifier on the match dataset.
    
    This function utilizes gradient accumulation to simulate a larger batch size
    even if physical VRAM is limited (e.g. accumulating 16 steps of batch_size=1
    gives an effective batch size of 16).
    """
    dataset_files = [f for f in dataset_files if os.path.exists(f)]
    if not dataset_files:
        print("No dataset files found. Run the pipeline first.")
        return

    if weights_dir is None:
        first_dir   = os.path.dirname(os.path.abspath(dataset_files[0]))
        weights_dir = os.path.join(first_dir, "football_lora_weights")

    print(f"Building dataset from {len(dataset_files)} match file(s)...")
    datasets = [MatchDataset(f) for f in dataset_files]
    combined = ConcatDataset(datasets)
    print(f"Full dataset     : {len(combined):,} windows")

    # ── Random subsample ────────────────────────────────────────────────────
    if max_samples is not None and max_samples < len(combined):
        rng     = torch.Generator().manual_seed(42)
        indices = torch.randperm(len(combined), generator=rng)[:max_samples].tolist()
        combined = Subset(combined, indices)
        print(f"Subsampled to    : {len(combined):,} windows (max_samples={max_samples})")
    # ────────────────────────────────────────────────────────────────────────

    # Compute label distribution to calculate class weights for the loss function
    if isinstance(combined, Subset):
        # If subsampled, we can't easily vectorize without loading all, 
        # but combined.indices is small enough.
        all_labels = torch.stack([combined.dataset[i][1] for i in combined.indices])
    else:
        # Fast path for large ConcatDatasets
        all_labels = torch.cat([ds.labels for ds in datasets])
    
    n_away     = (all_labels == 0).sum().float()
    n_balanced = (all_labels == 1).sum().float()
    n_home     = (all_labels == 2).sum().float()
    total      = float(len(all_labels))

    away_pct     = (n_away / total * 100).item() if total > 0 else 0
    balanced_pct = (n_balanced / total * 100).item() if total > 0 else 0
    home_pct     = (n_home / total * 100).item() if total > 0 else 0

    print(f"Label balance    : {away_pct:.1f}% away / {balanced_pct:.1f}% balanced / {home_pct:.1f}% home\n")

    # num_workers=4 is safe and fast on Linux; pin_memory speeds up CPU->GPU transfer
    dataloader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,   # keeps worker processes alive between epochs
    )

    model  = setup_football_model(model_name=model_name, num_labels=num_labels)
    device = next(model.parameters()).device
    # Mandatory for gradient checkpointing
    model.config.use_cache = False

    # Using standard PyTorch eager mode for maximum stability rather than `torch.compile`
    # which can introduce memory/OOM overheads during the compilation step on smaller GPUs.
    print("Running in eager mode (stability optimization)...")

    if num_labels == 3:
        eps = 1e-6
        class_weights = torch.tensor([
            total / (3 * (n_away + eps)),
            total / (3 * (n_balanced + eps)),
            total / (3 * (n_home + eps))
        ], device=device)
        print(f"Class weights    : away={class_weights[0]:.3f}  balanced={class_weights[1]:.3f}  home={class_weights[2]:.3f}")
    else:
        class_weights = torch.tensor(
            [total / (2 * n_away), total / (2 * n_home)], device=device
        )
        print(f"Class weights    : away={class_weights[0]:.3f}  home={class_weights[1]:.3f}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    print(f"Batch size       : {batch_size}  (accumulation steps: {accumulation_steps})")
    print(f"Effective batch  : {batch_size * accumulation_steps}")

    optimizer    = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps  = (len(dataloader) // accumulation_steps) * epochs
    warmup_steps = int(0.05 * total_steps)
    scheduler    = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    
    actual_device = next(model.parameters()).device
    print(f"\n" + "!"*60)
    print(f"  DEVICE DEPLOYED : {actual_device}")
    if "cpu" in str(actual_device).lower():
        print("  WARNING: Training on CPU will be 100x slower! (450+ hours)")
        print("  Please check your CUDA/bitsandbytes installation.")
    print("!"*60 + "\n")

    print(f"Steps/epoch      : {len(dataloader):,}")
    print(f"Total opt. steps : {total_steps:,}  (warmup: {warmup_steps:,})\n")

    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()

        bar = tqdm(enumerate(dataloader), total=len(dataloader),
                   desc=f"Epoch {epoch + 1}/{epochs}")

        for step, (batch_ids, batch_labels) in bar:
            batch_ids    = batch_ids.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)

            outputs = model(input_ids=batch_ids)
            logits  = outputs.logits.float()
            loss    = loss_fn(logits, batch_labels) / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
                # Gradient clipping avoids occasional exploding-gradient spikes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            bar.set_postfix({
                'avg_loss': f"{total_loss / (step + 1):.4f}",
                'lr':       f"{scheduler.get_last_lr()[0]:.2e}",
            })

        avg = total_loss / len(dataloader)
        print(f"  Epoch {epoch + 1} avg loss: {avg:.4f}")

        ckpt_dir = os.path.join(weights_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(ckpt_dir, exist_ok=True)
        model.save_pretrained(ckpt_dir)
        print(f"  Checkpoint saved: {ckpt_dir}\n")

    model.save_pretrained(weights_dir)
    print(f"\n=== Training Complete ({epochs} epochs) ===")
    print(f"Final weights: {weights_dir}")
    return model