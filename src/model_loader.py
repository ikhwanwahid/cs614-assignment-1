import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def get_bnb_config(config) -> BitsAndBytesConfig:
    """Create BitsAndBytesConfig for 4-bit QLoRA."""
    compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.load_in_4bit,
        bnb_4bit_quant_type=config.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
    )


def get_lora_config(config) -> LoraConfig:
    """Create LoRA configuration from experiment config."""
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type=config.task_type,
        bias="none",
    )


def load_base_model_and_tokenizer(config, for_training: bool = True) -> tuple:
    """Load quantized model + tokenizer.

    Args:
        config: ExperimentConfig
        for_training: If True, apply prepare_model_for_kbit_training + LoRA.
                      If False, load base model only (for baseline evaluation).

    Returns:
        (model, tokenizer)
    """
    bnb_config = get_bnb_config(config)

    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if for_training:
        model = prepare_model_for_kbit_training(model)
        lora_config = get_lora_config(config)
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

    return model, tokenizer


def load_finetuned_model(config, adapter_path: str) -> tuple:
    """Load base model + merge LoRA adapter for inference.

    Args:
        config: ExperimentConfig (for base model name and quant config).
        adapter_path: Path to the saved LoRA adapter directory.

    Returns:
        (model, tokenizer)
    """
    bnb_config = get_bnb_config(config)

    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # left padding for batched generation

    return model, tokenizer


def print_trainable_parameters(model) -> dict:
    """Print and return trainable vs total parameter counts."""
    trainable = 0
    total = 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    pct = 100 * trainable / total
    print(
        f"Trainable: {trainable:,} / {total:,} ({pct:.2f}%)"
    )
    return {"trainable": trainable, "total": total, "percent": pct}
