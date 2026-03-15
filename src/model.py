import torch
from transformers import AutoModelForSequenceClassification, AutoConfig, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType


# Map model families to their linear layer names for LoRA targeting.
# PEFT requires explicit module names; "all" is not valid.
_LORA_TARGETS = {
    "gpt2":  ["c_attn", "c_proj"],
    "qwen":  ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi":   ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
}

_DEFAULT_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _get_lora_targets(model_name):
    name_lower = model_name.lower()
    for key in _LORA_TARGETS:
        if key in name_lower:
            return _LORA_TARGETS[key]
    return _DEFAULT_TARGETS


def setup_football_model(model_name="Qwen/Qwen2.5-0.5B", num_labels=2):
    """
    Loads a decoder-only LLM with a sequence classification head, applies
    4-bit quantization (QLoRA), and wraps it with LoRA adapters.
    Automatically detects the correct LoRA target modules for the architecture.
    """
    print(f"Setting up {model_name} for sequence classification...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    except (ValueError, RuntimeError):
        print("No compatible GPU found. Loading standard model on CPU...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            trust_remote_code=True,
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    targets = _get_lora_targets(model_name)
    print(f"LoRA target modules: {targets}")

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=targets,
    )

    football_model = get_peft_model(model, peft_config)
    football_model.print_trainable_parameters()

    return football_model
