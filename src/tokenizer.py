import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer


def _compute_xg_labels(lines, meta_df, tok, context_length, stride,
                       label_window_minutes=5):
    """
    Assigns a binary momentum label to each sliding window using a two-stage
    signal derived from a time-expanded context around the window.

    Label = 1  →  home team momentum
    Label = 0  →  away team momentum

    Stage 1 — find the window's time range:
        Tokenise each narrative line individually to get per-line token
        positions (exact for GPT-2, which inserts no BOS/EOS by default).
        The minute/second span of the lines overlapping the window gives
        the window's real-world time range.

    Stage 2 — expand and score:
        Extend that range by ±label_window_minutes on each side, then sum
        home and away xG over ALL events that fall in the wider range.
        Because shots are sparse within a 128-token window (~5-10 events),
        this expansion ensures nearby shots inform the label even when they
        don't happen to sit inside the exact token window.

    Primary signal  : xG delta in the expanded range
    Secondary signal: possession (pass count) in the expanded range,
                      used only when both teams have xG = 0 in that range.
    """
    # ── per-line token positions ───────────────────────────────────────────
    line_token_lengths = np.array([
        len(tok(line + "\n", add_special_tokens=False)['input_ids'])
        for line in lines
    ])
    line_starts  = np.concatenate([[0], np.cumsum(line_token_lengths[:-1])])
    line_ends    = line_starts + line_token_lengths
    total_tokens = int(line_ends[-1])

    # Pre-extract meta columns as numpy arrays for vectorised access
    is_home   = meta_df['is_home'].to_numpy(dtype=bool)
    xg_vals   = meta_df['xg'].to_numpy(dtype=float)
    is_pass   = (meta_df['event_type'] == 'Pass').to_numpy(dtype=bool)
    time_secs = (meta_df['minute'].to_numpy(dtype=float) * 60
                 + meta_df['second'].to_numpy(dtype=float))

    expand_secs  = label_window_minutes * 60
    labels       = []
    from_xg      = 0   # diagnostic counters
    from_poss    = 0

    for w_start in range(0, total_tokens - context_length, stride):
        w_end   = w_start + context_length
        overlap = (line_starts < w_end) & (line_ends > w_start)

        # ── window time range ──────────────────────────────────────────────
        if overlap.any():
            t_lo = time_secs[overlap].min() - expand_secs
            t_hi = time_secs[overlap].max() + expand_secs
        else:
            # Fallback (shouldn't happen): use the full match
            t_lo, t_hi = time_secs.min(), time_secs.max()

        # ── score events in the expanded range ────────────────────────────
        in_range = (time_secs >= t_lo) & (time_secs <= t_hi)

        home_xg     = xg_vals[in_range &  is_home].sum()
        away_xg     = xg_vals[in_range & ~is_home].sum()
        home_passes = int((in_range &  is_home & is_pass).sum())
        away_passes = int((in_range & ~is_home & is_pass).sum())

        if home_xg != away_xg:
            labels.append(1 if home_xg > away_xg else 0)
            from_xg += 1
        else:
            labels.append(1 if home_passes >= away_passes else 0)
            from_poss += 1

    total = len(labels)
    if total > 0:
        print(f"  Label source : {from_xg}/{total} ({100*from_xg/total:.1f}%) xG-delta  |  "
              f"{from_poss}/{total} ({100*from_poss/total:.1f}%) possession tie-break  "
              f"[window ±{label_window_minutes} min]")

    return torch.tensor(labels, dtype=torch.long)


def tokenize_match_narrative(input_file, match_id, meta_file,
                             output_dir="../data/processed",
                             model_name="gpt2",
                             context_length=128, stride=64,
                             label_window_minutes=5):
    """
    Tokenises a match narrative into overlapping sliding windows and generates
    an xG-delta momentum label for each window.

    model_name controls which tokenizer is used — must match the model that
    will be trained, since different models have different vocabularies.

    Saves a dict  {'input_ids': Tensor[N, 128], 'labels': Tensor[N]}
    to  match_{match_id}_dataset.pt
    """
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # ── input token windows ───────────────────────────────────────────────
    all_ids = tok(text, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    print(f"  Total tokens : {len(all_ids)}")

    windows = []
    for i in range(0, len(all_ids) - context_length, stride):
        windows.append(all_ids[i : i + context_length])
    input_ids = torch.stack(windows)           # [N, context_length]

    # ── xG-delta labels ───────────────────────────────────────────────────
    lines   = text.splitlines()
    meta_df = pd.read_csv(meta_file)
    labels  = _compute_xg_labels(lines, meta_df, tok, context_length, stride,
                                  label_window_minutes=label_window_minutes)

    assert len(input_ids) == len(labels), (
        f"Window/label count mismatch: {len(input_ids)} vs {len(labels)}"
    )

    home_pct = labels.float().mean().item() * 100
    print(f"  Windows      : {len(input_ids)} x {context_length} tokens")
    print(f"  Label balance: {home_pct:.1f}% home / {100 - home_pct:.1f}% away momentum")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"match_{match_id}_dataset.pt")
    torch.save({'input_ids': input_ids, 'labels': labels}, output_file)
    print(f"  Saved dataset → {output_file}")
    return output_file


if __name__ == "__main__":
    MATCH_ID   = 3754058
    INPUT_FILE = f"../data/processed/match_{MATCH_ID}_narrative.txt"
    META_FILE  = f"../data/processed/match_{MATCH_ID}_meta.csv"
    if os.path.exists(INPUT_FILE) and os.path.exists(META_FILE):
        tokenize_match_narrative(INPUT_FILE, match_id=MATCH_ID, meta_file=META_FILE,
                                 model_name="Qwen/Qwen2.5-0.5B", label_window_minutes=5)
    else:
        print("Error: run data_pipeline.py first to generate the narrative and meta files.")
