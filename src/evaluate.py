"""
Evaluation Module

This module is responsible for loading the trained Large Language Model (base model + LoRA adapter)
and evaluating its performance on the test dataset. It computes quantitative metrics such as 
Accuracy, F1 Score, and ROC-AUC for the predicted momentum classes.
"""
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
from tqdm import tqdm

NUM_LABELS = 3
CLASS_NAMES = ['Away Momentum', 'Balanced', 'Home Momentum']


def _load_inference_model(model_name, weights_dir, num_labels=NUM_LABELS):
    """
    Builds the base language model architecture and applies the trained LoRA adapter on top.
    
    Args:
        model_name (str): The identifier for the base HuggingFace model.
        weights_dir (str): Path to the saved LoRA adapter weights, or a Hugging Face repo ID.
        num_labels (int): Number of classes for the sequence classifier.
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
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"\n[!] GPU SETUP FAILED: {e}")
        print("    Loading standard model on CPU.")
        base = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, trust_remote_code=True
        )

    if base.config.pad_token_id is None:
        base.config.pad_token_id = base.config.eos_token_id

    print(f"Loading LoRA adapter from {weights_dir}...")
    model = PeftModel.from_pretrained(base, weights_dir, is_trainable=False)
    model.eval()
    return model


def evaluate_model(test_files, weights_dir, model_name="gpt2",
                   batch_size=16, max_samples=None, num_labels=NUM_LABELS):
    """
    Runs inference on test set and reports 3-class momentum metrics.
    """
    test_files = [f for f in test_files if os.path.exists(f)]
    if not test_files:
        print("No test dataset files found.")
        return

    print(f"\nLoading test data from {len(test_files)} match(es)...")
    all_input_ids, all_labels = [], []
    for f in test_files:
        data = torch.load(f, weights_only=True)
        all_input_ids.append(data['input_ids'])
        all_labels.append(data['labels'])

    input_ids   = torch.cat(all_input_ids, dim=0)
    true_labels = torch.cat(all_labels,    dim=0)

    # Subsampling
    if max_samples is not None and max_samples < len(input_ids):
        torch.manual_seed(42)
        indices = torch.randperm(len(input_ids))[:max_samples]
        input_ids   = input_ids[indices]
        true_labels = true_labels[indices]
        print(f"  Reduced test set to: {max_samples:,} random windows.")

    # Class distribution
    for i, name in enumerate(CLASS_NAMES):
        pct = (true_labels == i).float().mean().item() * 100
        print(f"  {name}: {pct:.1f}%")

    model  = _load_inference_model(model_name, weights_dir, num_labels=num_labels)
    device = next(model.parameters()).device

    dataset    = TensorDataset(input_ids, true_labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds, all_probs, all_true = [], [], []

    bar = tqdm(dataloader, desc="Evaluating", total=len(dataloader))
    with torch.no_grad():
        for batch_ids, batch_labels in bar:
            batch_ids = batch_ids.to(device)
            outputs   = model(input_ids=batch_ids)

            logits_f = outputs.logits.float()
            probs    = torch.softmax(logits_f, dim=-1).cpu().numpy()
            preds    = logits_f.argmax(dim=-1).cpu().numpy()

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_true.extend(batch_labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)

    acc    = accuracy_score(all_true, all_preds)
    f1_mac = f1_score(all_true, all_preds, average='macro')
    f1_w   = f1_score(all_true, all_preds, average='weighted')
    cm     = confusion_matrix(all_true, all_preds)
    report = classification_report(all_true, all_preds, target_names=CLASS_NAMES)

    # ROC-AUC (one-vs-rest for multiclass)
    try:
        roc_auc = roc_auc_score(all_true, all_probs, multi_class='ovr', average='macro')
    except Exception:
        roc_auc = None

    print("\n" + "=" * 55)
    print("           EVALUATION RESULTS (3-Class)")
    print("=" * 55)
    print(f"  Accuracy      : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  Macro F1      : {f1_mac:.4f}")
    print(f"  Weighted F1   : {f1_w:.4f}")
    if roc_auc is not None:
        print(f"  ROC-AUC (OvR) : {roc_auc:.4f}")
    print(f"\n  Confusion Matrix (rows=Actual, cols=Pred):")
    print(f"                {'  '.join(f'{n[:4]:>6}' for n in CLASS_NAMES)}")
    for i, row in enumerate(cm):
        print(f"  {CLASS_NAMES[i][:14]:<14} {'  '.join(f'{v:6d}' for v in row)}")
    print(f"\n{report}")
    print("=" * 55)

    return {"accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_w, "roc_auc": roc_auc}