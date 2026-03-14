import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def _load_inference_model(model_name, weights_dir):
    """
    Rebuilds the base model + loads the saved LoRA adapter for inference.
    Tries 4-bit quantisation on GPU first; falls back to full-precision CPU.
    """
    print(f"Loading base model '{model_name}'...")
    try:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            quantization_config=bnb_config,
            device_map="auto",
        )
    except (ValueError, RuntimeError):
        print("  GPU/quantisation unavailable — loading on CPU.")
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )

    if base.config.pad_token_id is None:
        base.config.pad_token_id = base.config.eos_token_id

    print(f"Loading LoRA adapter from {weights_dir}...")
    model = PeftModel.from_pretrained(base, weights_dir, is_trainable=False)
    model.eval()
    return model


def evaluate_model(test_files, weights_dir, model_name="gpt2", batch_size=16):
    """
    Runs inference on an explicit list of match_*_dataset.pt test files and
    prints a full evaluation report:

      • Accuracy
      • Macro F1-score
      • ROC-AUC
      • Confusion matrix
      • Per-class precision/recall

    Parameters
    ----------
    test_files  : list[str]   paths to match_*_dataset.pt files (test split)
    weights_dir : str         directory produced by model.save_pretrained()
    model_name  : str         base HuggingFace model id (must match training)
    batch_size  : int         inference batch size (larger = faster on GPU)
    """
    if not os.path.isdir(weights_dir):
        print(f"Weights directory not found: {weights_dir}")
        print("Train the model first (Step 3 in main.py).")
        return

    test_files = [f for f in test_files if os.path.exists(f)]
    if not test_files:
        print("No test dataset files found.")
        return

    # ── Load test data ─────────────────────────────────────────────────────
    print(f"\nLoading test data from {len(test_files)} match(es)...")
    all_input_ids, all_labels = [], []
    for f in test_files:
        data = torch.load(f)
        all_input_ids.append(data['input_ids'])
        all_labels.append(data['labels'])

    input_ids    = torch.cat(all_input_ids, dim=0)
    true_labels  = torch.cat(all_labels,    dim=0)

    home_pct = true_labels.float().mean().item() * 100
    print(f"Test windows : {len(input_ids):,}")
    print(f"Label balance: {home_pct:.1f}% home / {100 - home_pct:.1f}% away\n")

    # ── Load model ─────────────────────────────────────────────────────────
    model  = _load_inference_model(model_name, weights_dir)
    device = next(model.parameters()).device

    # ── Inference loop ─────────────────────────────────────────────────────
    dataset    = TensorDataset(input_ids, true_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_probs, all_true = [], [], []

    print("Running inference...")
    with torch.no_grad():
        for batch_ids, batch_labels in dataloader:
            batch_ids = batch_ids.to(device)
            outputs   = model(input_ids=batch_ids)

            # Probabilities via softmax; column 1 = P(home momentum)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(batch_labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)

    # ── Metrics ────────────────────────────────────────────────────────────
    acc     = accuracy_score(all_true, all_preds)
    f1      = f1_score(all_true, all_preds, average='macro')
    roc_auc = roc_auc_score(all_true, all_probs)
    cm      = confusion_matrix(all_true, all_preds)
    report  = classification_report(all_true, all_preds,
                                    target_names=['Away momentum', 'Home momentum'])

    print("\n" + "=" * 50)
    print("         EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy   : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Macro F1   : {f1:.4f}")
    print(f"  ROC-AUC    : {roc_auc:.4f}")
    print()
    print("  Confusion Matrix")
    print("  (rows=actual, cols=predicted)")
    print(f"               Away   Home")
    print(f"  Actual Away  {cm[0,0]:5d}  {cm[0,1]:5d}")
    print(f"  Actual Home  {cm[1,0]:5d}  {cm[1,1]:5d}")
    print()
    print("  Per-class breakdown:")
    print(report)
    print("=" * 50)

    return {"accuracy": acc, "macro_f1": f1, "roc_auc": roc_auc}
