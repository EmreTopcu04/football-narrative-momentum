"""
Main execution script for the Football Narrative Momentum pipeline.

This script acts as the entry point for the entire project. It orchestrates the process
of fetching match events from StatsBomb, tokenizing these events into sliding windows,
training a LoRA-based sequence classifier to predict momentum, and evaluating the model.

Usage:
  python main.py [fetch|retokenize|train|evaluate|pipeline]
"""
import os
import sys
import glob
import warnings
import pandas as pd
import random
from dotenv import load_dotenv
from huggingface_hub import login



# Prevent CUDA memory fragmentation 
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'

warnings.filterwarnings('ignore')

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.append(SRC_DIR)

from statsbombpy import sb
from data_pipeline import fetch_and_process_match
from tokenizer import tokenize_match_narrative
from train import train_model
from evaluate import evaluate_model

# ── Configuration ──────────────────────────────────────────────────────────────


# LOCAL BASELINE CONFIG
MODEL_NAME           = "Qwen/Qwen2.5-1.5B"
CONTEXT_LENGTH       = 512
STRIDE               = 256

NUM_LABELS           = 3      # 0=Away, 1=Balanced, 2=Home
TRAIN_RATIO          = 0.8
# Random window subsample within each split (seed fixed in train.py / evaluate.py).
# None = use every window from train/test matches. Caps align with the ~20k / ~5k eval setup.
TRAIN_MAX_SAMPLES    = 20_000
EVAL_MAX_SAMPLES     = 5_000
LABEL_WINDOW_MINUTES = 5      
PROJECT_ROOT         = os.path.dirname(os.path.abspath(__file__))
# ── Directory layout ───────────────────────────────────────────────────────────
# data/raw/        → narrative .txt + meta .csv files (fetched from StatsBomb API)
# data/pt-files/   → tokenized .pt dataset files (input to training)
# data/processed/  → LoRA weights, checkpoints, and model artifacts
RAW_DATA_DIR         = os.path.join(PROJECT_ROOT, "data", "raw")
PT_FILES_DIR         = os.path.join(PROJECT_ROOT, "data", "pt-files")
PROCESSED_DIR        = os.path.join(PROJECT_ROOT, "data", "processed")
WEIGHTS_DIR          = os.path.join(PROCESSED_DIR, "football_lora_weights")
# ───────────────────────────────────────────────────────────────────────────────



_REQUIRED_META_COLS = {
    'minute', 'second', 'is_home', 'xg', 'event_type',
    'is_goal', 'home_score', 'away_score',
    'cumulative_home_xg', 'cumulative_away_xg',
}

_TOKENIZER_MARKER = os.path.join(PROCESSED_DIR, ".tokenizer_model.txt")


def _meta_is_valid(meta_file):
    """Returns True only if the meta CSV exists and has all required v2 columns."""
    if not os.path.exists(meta_file):
        return False
    try:
        return _REQUIRED_META_COLS.issubset(pd.read_csv(meta_file, nrows=0).columns)
    except Exception:
        return False


def upload_to_hf(model, repo_name="emretopcuu/football_narrative_momentum"):
    """Reads HF_TOKEN from .env and pushes the adapter weights to the hub."""
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("\n[HF] Skip: HF_TOKEN not found in .env. Model saved locally only.")
        return

    try:
        print(f"\n[HF] Logging into Hugging Face...")
        login(token=token)
        print(f"[HF] Pushing adapters to {repo_name}...")
        model.push_to_hub(repo_name)
        print(f"[HF] Success! Model available at: https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"[HF] Error uploading to hub: {e}")



def _check_tokenizer_changed():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    current_marker = f"{MODEL_NAME}_ctx{CONTEXT_LENGTH}"
    prev_marker = ""
    if os.path.exists(_TOKENIZER_MARKER):
        with open(_TOKENIZER_MARKER, 'r') as f:
            prev_marker = f.read().strip()

    if prev_marker != current_marker:
        stale = glob.glob(os.path.join(PT_FILES_DIR, "match_*_dataset.pt"))
        if stale:
            print(f"Config changed ({prev_marker or 'none'} → {current_marker}).")
            print(f"Deleting {len(stale)} stale _dataset.pt files...")
            for f in stale:
                os.remove(f)
        with open(_TOKENIZER_MARKER, 'w') as f:
            f.write(current_marker)


def run_pipeline():
    print("=" * 60)
    print("  StatsBomb Open Data — Momentum Prediction Pipeline v2")
    print("=" * 60 + "\n")

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(PT_FILES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    _check_tokenizer_changed()

    # ── STEP 1: Discover all competitions & matches ─────────────────────────
    print("--- STEP 1: Fetching competitions & match lists ---")
    competitions = sb.competitions()
    all_match_ids = []

    from transformers import AutoTokenizer
    print(f"Loading tokenizer ({MODEL_NAME})...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("Tokenizer ready.\n")

    for _, comp in competitions.iterrows():
        cid   = comp["competition_id"]
        sid   = comp["season_id"]
        cname = comp.get("competition_name", cid)
        sname = comp.get("season_name", sid)

        print(f"\n=== Competition {cid} ({cname}) — Season {sid} ({sname}) ===")
        matches = sb.matches(competition_id=cid, season_id=sid)
        if matches.empty:
            print("  No matches returned, skipping.")
            continue

        print(f"  Found {len(matches)} matches. Ingesting & tokenising...")

        # ── STEP 2: Per-match ingestion, formatting, and tokenisation ───────
        for _, row in matches.iterrows():
            match_id  = row["match_id"]
            home_team = row["home_team"]
            away_team = row["away_team"]
            all_match_ids.append(match_id)

            print(f"    → {home_team} vs {away_team}  (id={match_id})")

            narrative_file = os.path.join(RAW_DATA_DIR, f"match_{match_id}_narrative.txt")
            meta_file      = os.path.join(RAW_DATA_DIR, f"match_{match_id}_meta.csv")
            dataset_file   = os.path.join(PT_FILES_DIR,  f"match_{match_id}_dataset.pt")

            # Re-fetch if meta is stale (missing v2 columns) or narrative is missing
            if not os.path.exists(narrative_file) or not _meta_is_valid(meta_file):
                fetch_and_process_match(match_id, home_team=home_team, output_dir=RAW_DATA_DIR)
            else:
                print(f"      Narrative + meta exist (v2) — skipping fetch.")

            if not os.path.exists(dataset_file):
                tokenize_match_narrative(
                    narrative_file, match_id=match_id,
                    meta_file=meta_file, output_dir=PT_FILES_DIR,
                    context_length=CONTEXT_LENGTH,
                    stride=STRIDE,
                    label_window_minutes=LABEL_WINDOW_MINUTES,
                    tok=tok,
                )
            else:
                print(f"      Dataset file exists — skipping tokenisation.")

    all_match_ids = sorted(set(all_match_ids))
    print(f"\nAll {len(all_match_ids)} matches ready in {PT_FILES_DIR}\n")

    # ── STEP 3: Train / test split (by match, not by window) ────────────────
    print("--- STEP 3: Train / test split ---")
    split_idx   = int(len(all_match_ids) * TRAIN_RATIO)
    train_ids   = all_match_ids[:split_idx]
    test_ids    = all_match_ids[split_idx:]
    train_files = [os.path.join(PT_FILES_DIR, f"match_{mid}_dataset.pt") for mid in train_ids]
    test_files  = [os.path.join(PT_FILES_DIR, f"match_{mid}_dataset.pt") for mid in test_ids]
    print(f"Train: {len(train_ids)} matches  |  Test: {len(test_ids)} matches\n")

    model = train_model(
        dataset_files=train_files,
        model_name=MODEL_NAME,
        weights_dir=WEIGHTS_DIR,
        epochs=1,              
        batch_size=8,
        accumulation_steps=4, 
        max_samples=TRAIN_MAX_SAMPLES,
        num_labels=NUM_LABELS,
    )

    print("\n--- STEP 5: Hugging Face Upload ---")
    upload_to_hf(model)

    print("\n--- STEP 6: Evaluation ---")

    evaluate_model(test_files=test_files, weights_dir=WEIGHTS_DIR, model_name=MODEL_NAME, max_samples=EVAL_MAX_SAMPLES)


def _get_split():
    """
    Build train/test file lists purely from dataset files on disk.
    Looks in PT_FILES_DIR (data/pt-files/) for .pt dataset files.
    """
    dataset_files = sorted(glob.glob(os.path.join(PT_FILES_DIR, "match_*_dataset.pt")))
    if not dataset_files:
        print(f"No dataset files found in {PT_FILES_DIR}.")
        return [], []

    print(f"Using dataset files from: {PT_FILES_DIR}")

    def _extract_id(path: str) -> int:
        return int(os.path.basename(path).split("_")[1])

    dataset_files = sorted(dataset_files, key=_extract_id)
    split_idx = int(len(dataset_files) * TRAIN_RATIO)
    train_files = dataset_files[:split_idx]
    test_files  = dataset_files[split_idx:]
    return train_files, test_files


def run_train_only():
    print("=" * 60)
    print("  Train + Evaluate (skipping data ingestion)")
    print("=" * 60 + "\n")
    train_files, test_files = _get_split()
    print(f"Train: {len(train_files)} matches  |  Test: {len(test_files)} matches\n")
    model = train_model(
        dataset_files=train_files,
        model_name=MODEL_NAME,
        weights_dir=WEIGHTS_DIR,
        epochs=1,           
        batch_size=8,
        accumulation_steps=2,
        max_samples=TRAIN_MAX_SAMPLES,
        num_labels=NUM_LABELS,
    )

    print("\n--- Hugging Face Upload ---")
    upload_to_hf(model)

    print("\n--- Evaluation ---")

    evaluate_model(test_files=test_files, weights_dir=WEIGHTS_DIR, model_name=MODEL_NAME, max_samples=EVAL_MAX_SAMPLES)


def run_evaluate_only():
    print("=" * 60)
    print("  Evaluation only")
    print("=" * 60 + "\n")
    _, test_files = _get_split()
    print(f"Test set: {len(test_files)} matches\n")
    evaluate_model(test_files=test_files, weights_dir=WEIGHTS_DIR, model_name=MODEL_NAME, max_samples=EVAL_MAX_SAMPLES)


def run_fetch_only():
    """
    Step 1 & 2: Discover matches and fetch/format them.
    Skips matches that already have valid narrative and meta files.
    """
    print("=" * 60)
    print("  FETCHING Data from StatsBomb")
    print("=" * 60 + "\n")

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    competitions = sb.competitions()
    
    for _, comp in competitions.iterrows():
        cid, sid = comp["competition_id"], comp["season_id"]
        print(f"\n- Competition {cid} Season {sid} -")
        matches = sb.matches(competition_id=cid, season_id=sid)
        
        for _, row in matches.iterrows():
            match_id = row["match_id"]
            narrative_file = os.path.join(RAW_DATA_DIR, f"match_{match_id}_narrative.txt")
            meta_file      = os.path.join(RAW_DATA_DIR, f"match_{match_id}_meta.csv")

            # Skip if already processed
            if os.path.exists(narrative_file) and _meta_is_valid(meta_file):
                continue

            print(f"    Processing Match {match_id} ({row['home_team']} vs {row['away_team']})")
            fetch_and_process_match(match_id, home_team=row['home_team'], output_dir=RAW_DATA_DIR)

    print(f"\nDone. All raw data is in {RAW_DATA_DIR}")


worker_tokenizer = None

def _init_tokenizer_worker(model_name):
    """Initializes the tokenizer once per worker process with a jitter to avoid HF rate limits."""
    global worker_tokenizer
    import time
    import random
    import os
    from transformers import AutoTokenizer
    
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    time.sleep(random.random() * 2.0)
    
    try:
        worker_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=True
        )
        if worker_tokenizer.pad_token is None:
            worker_tokenizer.pad_token = worker_tokenizer.eos_token
    except Exception as e:
        print(f"  [ERROR] Failed to initialize tokenizer in worker: {e}")


def _tokenize_worker(args):
    """Worker function that uses the globally initialized tokenizer."""
    global worker_tokenizer
    narrative_file, match_id, meta_file, pt_files_dir, ctx_len, stride, lw_mins = args

    if worker_tokenizer is None:
        return None

    try:
        return tokenize_match_narrative(
            narrative_file, match_id=match_id,
            meta_file=meta_file, output_dir=pt_files_dir,
            context_length=ctx_len, stride=stride,
            label_window_minutes=lw_mins, tok=worker_tokenizer,
        )
    except Exception as e:
        print(f"  [ERROR] Match {match_id}: {e}")
        return None


def run_retokenize_only(resume=False):
    """
    Offline retokenization using all CPU cores (multiprocessing).

    Args:
        resume: If True, skips matches that already have a .pt file in PT_FILES_DIR.
                Use this after a crash to pick up where you left off.
    """
    import multiprocessing as mp

    print("=" * 60)
    print("  Offline Retokenization (Multiprocessing)")
    print("=" * 60 + "\n")

    # Find narrative files in raw dir; fallback to processed for old layout
    narratives = glob.glob(os.path.join(RAW_DATA_DIR, "match_*_narrative.txt"))
    if not narratives:
        narratives = glob.glob(os.path.join(PROCESSED_DIR, "match_*_narrative.txt"))
    if not narratives:
        print(f"No narrative files found in {RAW_DATA_DIR}. Cannot retokenize.")
        return

    os.makedirs(PT_FILES_DIR, exist_ok=True)

    # ── Pre-warm tokenizer cache in main process ──────────────────────────
    from transformers import AutoTokenizer
    print(f"Pre-warming tokenizer cache (Offline Mode)...")
    try:
        AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, local_files_only=True)
    except Exception:
        # If not in cache, let it download once then it will be local
        AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Cache ready.\n")

    # Build work list
    work = []
    skipped = 0
    for narrative_file in sorted(narratives):
        basename = os.path.basename(narrative_file)
        match_id = basename.split("_")[1]

        # Resume mode: skip if .pt already exists
        if resume and os.path.exists(os.path.join(PT_FILES_DIR, f"match_{match_id}_dataset.pt")):
            skipped += 1
            continue

        meta_file = os.path.join(RAW_DATA_DIR, f"match_{match_id}_meta.csv")
        if not os.path.exists(meta_file):
            meta_file = os.path.join(PROCESSED_DIR, f"match_{match_id}_meta.csv")
        if not os.path.exists(meta_file):
            print(f"  [Skip] Match {match_id}: meta.csv missing.")
            continue

        work.append((
            narrative_file, match_id, meta_file,
            PT_FILES_DIR,
            CONTEXT_LENGTH, STRIDE, LABEL_WINDOW_MINUTES,
        ))

    if resume and skipped:
        print(f"Resume mode: skipped {skipped} already-processed matches.")

    if not work:
        print("Nothing to do. All matches already tokenized.")
        return

    n_workers = min(8, mp.cpu_count(), len(work))
    print(f"Processing {len(work)} matches using {n_workers} CPU cores (Rate-Limited Mode)...\n")

    with mp.Pool(processes=n_workers, initializer=_init_tokenizer_worker, initargs=(MODEL_NAME,)) as pool:
        for i, result in enumerate(pool.imap_unordered(_tokenize_worker, work), 1):
            if i % 10 == 0 or i == len(work):
                print(f"  [{i}/{len(work)}] matches processed")

    print(f"\nRetokenization complete. Files saved to: {PT_FILES_DIR}")



if __name__ == "__main__":
    mode   = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    resume = "--resume" in sys.argv

    if mode in ("evaluate", "-e"):
        run_evaluate_only()
    elif mode in ("train", "-t"):
        run_train_only()
    elif mode in ("retokenize", "-r"):
        run_retokenize_only(resume=resume)
    elif mode in ("fetch", "-f"):
        run_fetch_only()
    elif mode == "pipeline":
        run_pipeline()
    else:
        print("Usage: python main.py [fetch|retokenize|train|evaluate|pipeline]")