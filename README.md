# Narrative Momentum

Narrative-based tactical momentum prediction for football matches: StatsBomb open data → chronological text → sliding-window sequence classification with **Qwen2.5** + **QLoRA** (4-bit base model + LoRA adapters), with xG-delta–derived labels.

## Hugging Face model (LoRA adapters)

Published adapters (not a full fine-tuned base):

**[emretopcuu/football_narrative_momentum](https://huggingface.co/emretopcuu/football_narrative_momentum)**

Use them with the **same base checkpoint** you trained on (see `MODEL_NAME` in `main.py`, default `Qwen/Qwen2.5-1.5B`) and load via **[PEFT](https://github.com/huggingface/peft)** (`PeftModel.from_pretrained`). Training and evaluation in this repo already follow that pattern (`src/model.py`, `src/evaluate.py`).

Minimal inference sketch (same loading pattern as `src/evaluate.py`):

```python
import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

base_id = "Qwen/Qwen2.5-1.5B"
adapter_id = "emretopcuu/football_narrative_momentum"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
base = AutoModelForSequenceClassification.from_pretrained(
    base_id,
    num_labels=3,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
if base.config.pad_token_id is None:
    base.config.pad_token_id = base.config.eos_token_id
model = PeftModel.from_pretrained(base, adapter_id, is_trainable=False)
```

Use the same tokenizer as training (`AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)`). For evaluation end-to-end, run `python main.py evaluate` so paths and metrics stay aligned with this repo.

## Requirements

Python 3 with PyTorch; see `requirements.txt` (`transformers`, `peft`, `bitsandbytes`, `accelerate`, `statsbombpy`, etc.). A CUDA GPU is strongly recommended for training and evaluation.

Optional: create `.env` with `HF_TOKEN` if you use `upload_to_hf` in `main.py` to push adapters to the Hub.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```text
python main.py [fetch|retokenize|train|evaluate|pipeline]
```

Short flags: `-f` fetch, `-r` retokenize (`--resume` to skip finished matches), `-t` train, `-e` evaluate.

| Mode | What it does |
|------|----------------|
| `fetch` | Download/process StatsBomb data into `data/raw/` |
| `retokenize` | Build `data/pt-files/match_*_dataset.pt` from raw narratives |
| `train` | Train LoRA on the train split; optional HF upload; then evaluate |
| `evaluate` | Load adapters from `data/processed/football_lora_weights` and evaluate test split |
| `pipeline` | Full flow: discover competitions → ingest → tokenize → train → evaluate |

Train/test split is **by match** (80/20). Window subsampling caps are configured in `main.py` (`TRAIN_MAX_SAMPLES`, `EVAL_MAX_SAMPLES`).

Passing no mode prints usage and exits.

## Sample input and output

### Input (raw narrative)

After `fetch`, each match is stored under `data/raw/` as `match_<id>_narrative.txt` (chronological event text). Excerpt from `data/raw/match_69161_narrative.txt`:

```text
[00:00] [Spain Women's 0-0 Opponent] Spain Women's: Jennifer Hermoso Fuentes completed a pass (ground pass) from the middle third (center) to Irene Paredes Hernandez.
[00:02] [Spain Women's 0-0 Opponent] Spain Women's: Irene Paredes Hernandez attempted a incomplete pass (high pass) from the defensive third (right half-space) to Alexia Putellas Segura.
[00:05] [Spain Women's 0-0 Opponent] United States Women's: Kelley Maureen O'Hara cleared the ball in the defensive third (right flank) under pressure from Spain Women's.
…
```

`retokenize` turns these into sliding windows in `data/pt-files/match_*_dataset.pt` (token tensors + labels).

### What the model predicts

Each window is a **3-class** momentum label: **Away Momentum**, **Balanced**, **Home Momentum** (see `CLASS_NAMES` in `src/evaluate.py`). There is no natural-language prompt; the LM trunk sees tokenized narrative windows and the classification head outputs logits.

### Output from training (`python main.py train`)

The training loop prints per-step loss in the tqdm bar (`avg_loss`) and a per-epoch line, e.g. `Epoch 1 avg loss: 0.9601`, and writes LoRA weights under `data/processed/football_lora_weights/`.

### Output from evaluation (`python main.py evaluate`)

Evaluation aggregates predictions over the test windows and **prints** accuracy, macro/weighted F1, ROC-AUC (one-vs-rest), confusion matrix, and a per-class classification report—see `src/evaluate.py`. Example shape of the summary (figures vary by run and checkpoint):

```text
=======================================================
           EVALUATION RESULTS (3-Class)
=======================================================
  Accuracy      : 0.8000  (80.00%)
  Macro F1      : 0.7767
  Weighted F1   : 0.7996
  ROC-AUC (OvR) : 0.8748

  Confusion Matrix (rows=Actual, cols=Pred):
                  Away    Bala    Home
  Away Momentum     706     221      86
  Balanced          198    2319     222
  Home Momentum      56     217     975
…
```
