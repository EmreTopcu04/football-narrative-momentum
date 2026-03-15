import os
import sys
import glob
import warnings

import pandas as pd
warnings.filterwarnings('ignore')

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(SRC_DIR)

from statsbombpy import sb
from data_pipeline import fetch_and_process_match
from tokenizer import tokenize_match_narrative
from train import train_model
from evaluate import evaluate_model

# ── Configuration ──────────────────────────────────────────────────────────────
COMPETITION_ID       = 2    # Premier League
SEASON_ID            = 27   # 2015/16
MODEL_NAME           = "Qwen/Qwen2.5-0.5B"
TRAIN_RATIO          = 0.8  # 80 % of matches → training, 20 % → held-out evaluation
# How many minutes either side of each token window to include when summing xG
# for label computation.  Higher = xG dominates more; 0 = within-window only.
LABEL_WINDOW_MINUTES = 5
PROJECT_ROOT         = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR           = os.path.join(PROJECT_ROOT, "data", "processed")
WEIGHTS_DIR          = os.path.join(OUTPUT_DIR, "football_lora_weights")  # LoRA adapter saved/loaded here only
# ───────────────────────────────────────────────────────────────────────────────

# Required columns added when LABEL_WINDOW_MINUTES support was introduced.
# If an existing meta CSV pre-dates this, it will be regenerated automatically.
_REQUIRED_META_COLS = {'minute', 'second', 'is_home', 'xg', 'event_type'}


_TOKENIZER_MARKER = os.path.join(OUTPUT_DIR, ".tokenizer_model.txt")


def _meta_is_valid(meta_file):
    """Returns True only if the meta CSV exists and has all required columns."""
    if not os.path.exists(meta_file):
        return False
    try:
        return _REQUIRED_META_COLS.issubset(pd.read_csv(meta_file, nrows=0).columns)
    except Exception:
        return False


def _check_tokenizer_changed():
    """
    If the model (and thus tokenizer) has changed since the last run, delete
    all _dataset.pt files so they get regenerated with the correct vocabulary.
    Narratives and meta CSVs are model-independent and stay untouched.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prev_model = ""
    if os.path.exists(_TOKENIZER_MARKER):
        with open(_TOKENIZER_MARKER, 'r') as f:
            prev_model = f.read().strip()

    if prev_model != MODEL_NAME:
        stale = glob.glob(os.path.join(OUTPUT_DIR, "match_*_dataset.pt"))
        if stale:
            print(f"Model changed ({prev_model or 'none'} → {MODEL_NAME}).")
            print(f"Deleting {len(stale)} stale _dataset.pt files for re-tokenization...\n")
            for f in stale:
                os.remove(f)
        with open(_TOKENIZER_MARKER, 'w') as f:
            f.write(MODEL_NAME)


def run_pipeline():
    print("=" * 60)
    print("  Premier League 2015/16 — Momentum Prediction Pipeline")
    print("=" * 60 + "\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    _check_tokenizer_changed()

    # ── STEP 1: Discover all matches ───────────────────────────────────────
    print("--- STEP 1: Fetching match list ---")
    matches   = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    match_ids = matches['match_id'].tolist()
    print(f"Found {len(match_ids)} matches in the 2015/16 Premier League.\n")

    # ── STEP 2: Per-match ingestion, formatting, and tokenisation ──────────
    print("--- STEP 2: Data ingestion & tokenisation ---")
    for i, match_id in enumerate(match_ids):
        row       = matches[matches['match_id'] == match_id].iloc[0]
        home_team = row['home_team']
        away_team = row['away_team']
        print(f"\n[{i + 1:3d}/{len(match_ids)}] {home_team} vs {away_team}  (id={match_id})")

        narrative_file = os.path.join(OUTPUT_DIR, f"match_{match_id}_narrative.txt")
        meta_file      = os.path.join(OUTPUT_DIR, f"match_{match_id}_meta.csv")
        dataset_file   = os.path.join(OUTPUT_DIR, f"match_{match_id}_dataset.pt")

        if not os.path.exists(narrative_file) or not _meta_is_valid(meta_file):
            fetch_and_process_match(match_id, home_team=home_team, output_dir=OUTPUT_DIR)
        else:
            print(f"  Narrative + meta exist — skipping fetch.")

        if not os.path.exists(dataset_file):
            tokenize_match_narrative(narrative_file, match_id=match_id,
                                     meta_file=meta_file, output_dir=OUTPUT_DIR,
                                     model_name=MODEL_NAME,
                                     label_window_minutes=LABEL_WINDOW_MINUTES)
        else:
            print(f"  Dataset file exists    — skipping tokenisation.")

    print(f"\nAll {len(match_ids)} matches ready in {OUTPUT_DIR}\n")

    # ── STEP 3: Train / test split (by match, not by window) ──────────────
    # Splitting by whole match prevents any single game's windows from
    # appearing in both sets, which would cause data leakage.
    print("--- STEP 3: Train / test split ---")
    split_idx    = int(len(match_ids) * TRAIN_RATIO)
    train_ids    = match_ids[:split_idx]
    test_ids     = match_ids[split_idx:]

    train_files  = [os.path.join(OUTPUT_DIR, f"match_{mid}_dataset.pt") for mid in train_ids]
    test_files   = [os.path.join(OUTPUT_DIR, f"match_{mid}_dataset.pt") for mid in test_ids]

    print(f"Train matches : {len(train_ids)}  ({len(train_files)} dataset files)")
    print(f"Test  matches : {len(test_ids)}  ({len(test_files)} dataset files)\n")

    # ── STEP 4: Fine-tuning ────────────────────────────────────────────────
    print("--- STEP 4: Training ---")
    train_model(dataset_files=train_files, model_name=MODEL_NAME,
                weights_dir=WEIGHTS_DIR, epochs=2)

    # ── STEP 5: Evaluation ────────────────────────────────────────────────
    print("\n--- STEP 5: Evaluation ---")
    evaluate_model(test_files=test_files, weights_dir=WEIGHTS_DIR, model_name=MODEL_NAME)


def _get_split():
    """Returns (train_files, test_files) using the same deterministic split."""
    matches   = sb.matches(competition_id=COMPETITION_ID, season_id=SEASON_ID)
    match_ids = matches['match_id'].tolist()
    split_idx = int(len(match_ids) * TRAIN_RATIO)
    train_ids = match_ids[:split_idx]
    test_ids  = match_ids[split_idx:]
    train_files = [os.path.join(OUTPUT_DIR, f"match_{mid}_dataset.pt") for mid in train_ids]
    test_files  = [os.path.join(OUTPUT_DIR, f"match_{mid}_dataset.pt") for mid in test_ids]
    return train_files, test_files


def run_train_only():
    """Skip data ingestion — just train on existing dataset files, then evaluate."""
    print("=" * 60)
    print("  Train + Evaluate (skipping data ingestion)")
    print("=" * 60 + "\n")

    train_files, test_files = _get_split()
    print(f"Train: {len(train_files)} matches  |  Test: {len(test_files)} matches\n")

    train_model(dataset_files=train_files, model_name=MODEL_NAME,
                weights_dir=WEIGHTS_DIR, epochs=2)

    print("\n--- Evaluation ---")
    evaluate_model(test_files=test_files, weights_dir=WEIGHTS_DIR, model_name=MODEL_NAME)


def run_evaluate_only():
    """Run only evaluation (Step 5): load saved LoRA weights and test set, no training."""
    print("=" * 60)
    print("  Evaluation only — loading weights & test set")
    print("=" * 60 + "\n")

    _, test_files = _get_split()
    print(f"Test set: {len(test_files)} matches (same 80/20 split as training)\n")
    evaluate_model(test_files=test_files, weights_dir=WEIGHTS_DIR, model_name=MODEL_NAME)


if __name__ == "__main__":
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if mode in ("evaluate", "--evaluate-only", "-e"):
        run_evaluate_only()
    elif mode in ("train", "--train-only", "-t"):
        run_train_only()
    else:
        run_pipeline()
