import torch
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType

def setup_football_model(model_name="gpt2", num_labels=2):
    """
    Loads a base LLM, prepares it for 4-bit quantization (QLoRA), and attaches 
    a sequence classification head for predicting tactical momentum.
    """
    print(f"Setting up {model_name} for sequence classification...")

    # 1. Configure 4-bit Quantization (Saves massive VRAM)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 2. Load the Model with a Classification Head
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            quantization_config=bnb_config,
            device_map="auto" 
        )
    except ValueError:
        print("No compatible GPU found. Loading standard model on CPU...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )

    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    # 3. Configure LoRA (Freeze base model, train tiny adapters)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=8, 
        lora_alpha=32, 
        lora_dropout=0.1
    )

    # 4. Wrap the model with PEFT
    football_model = get_peft_model(model, peft_config)
    
    return football_model