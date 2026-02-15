from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Single experiment configuration for MedQA fine-tuning."""

    # Identifiers
    name: str = "default"
    description: str = ""

    # Model
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.3"
    max_seq_length: int = 1024

    # QLoRA / BitsAndBytes
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    task_type: str = "CAUSAL_LM"

    # Training
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # effective batch = 16
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"
    fp16: bool = False
    bf16: bool = True

    # Logging / saving
    logging_steps: int = 25
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 2
    eval_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Data
    dataset_name: str = "GBaker/MedQA-USMLE-4-options-hf"
    train_subset_frac: float = 1.0
    seed: int = 42

    # Output
    output_dir: str = "results"


def get_all_configs() -> dict[str, ExperimentConfig]:
    """Return all experiment configurations.

    6 configs that systematically vary LoRA rank, learning rate,
    epochs, and dropout to explore the hyperparameter space.
    """
    configs = {}

    # Config 1: Standard QLoRA defaults -- serves as anchor
    configs["config_1_baseline"] = ExperimentConfig(
        name="config_1_baseline",
        description=(
            "Baseline: r=16, alpha=32, lr=2e-4, 2 epochs. "
            "Standard starting point from QLoRA literature."
        ),
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-4,
        num_train_epochs=2,
    )

    # Config 2: Lower rank -- test if fewer params suffice for MCQ
    configs["config_2_low_rank"] = ExperimentConfig(
        name="config_2_low_rank",
        description=(
            "Low rank: r=8, alpha=16. Tests whether fewer trainable "
            "parameters suffice for MCQ classification, reducing overfitting risk."
        ),
        lora_r=8,
        lora_alpha=16,
        learning_rate=2e-4,
        num_train_epochs=2,
    )

    # Config 3: Higher rank -- test if more capacity helps reasoning
    configs["config_3_high_rank"] = ExperimentConfig(
        name="config_3_high_rank",
        description=(
            "High rank: r=64, alpha=128. Tests whether increased capacity "
            "captures more nuanced medical reasoning. Lower LR to compensate."
        ),
        lora_r=64,
        lora_alpha=128,
        learning_rate=1e-4,
        num_train_epochs=2,
    )

    # Config 4: Lower learning rate -- test stability
    configs["config_4_low_lr"] = ExperimentConfig(
        name="config_4_low_lr",
        description=(
            "Lower LR: 5e-5 with r=16. Tests whether slower learning "
            "produces more stable convergence and better generalization. "
            "More epochs to compensate."
        ),
        lora_r=16,
        lora_alpha=32,
        learning_rate=5e-5,
        num_train_epochs=3,
    )

    # Config 5: Extended training -- find overfitting boundary
    configs["config_5_longer_training"] = ExperimentConfig(
        name="config_5_longer_training",
        description=(
            "Extended: 3 epochs at lr=2e-4. Tests benefit of additional "
            "passes vs. overfitting. Should reveal optimal epoch count."
        ),
        lora_r=16,
        lora_alpha=32,
        learning_rate=2e-4,
        num_train_epochs=3,
    )

    # Config 6: Aggressive -- high rank + higher LR + more dropout
    configs["config_6_aggressive"] = ExperimentConfig(
        name="config_6_aggressive",
        description=(
            "Aggressive: r=32, alpha=64, lr=3e-4, dropout=0.1. "
            "Pushes capacity and learning speed with dropout as regularizer."
        ),
        lora_r=32,
        lora_alpha=64,
        learning_rate=3e-4,
        lora_dropout=0.1,
        num_train_epochs=2,
    )

    return configs
