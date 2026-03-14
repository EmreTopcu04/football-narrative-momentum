# Narrative Momentum

Narrative-based tactical momentum prediction for Premier League matches: StatsBomb event data → chronological text → sliding-window sequence classification (GPT-2 + QLoRA), with xG-delta labels.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Pipeline: fetch 2015/16 PL matches → narratives + meta → tokenise + xG labels → train (80%) → evaluate (20%).
