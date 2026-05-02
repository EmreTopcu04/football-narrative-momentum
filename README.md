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
