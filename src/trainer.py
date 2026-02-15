import json
import os
import time

from transformers import EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


def get_training_args(config) -> SFTConfig:
    """Build SFTConfig from ExperimentConfig."""
    output_dir = os.path.join(config.output_dir, config.name)
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        optim=config.optim,
        fp16=config.fp16,
        bf16=config.bf16,
        logging_steps=config.logging_steps,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        report_to="none",
        seed=config.seed,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # SFT-specific args (moved from SFTTrainer constructor)
        max_seq_length=config.max_seq_length,
        dataset_text_field="text",
        packing=False,
    )


def create_trainer(model, tokenizer, train_dataset, eval_dataset,
                   config) -> SFTTrainer:
    """Create and return a configured SFTTrainer."""
    training_args = get_training_args(config)

    return SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )


def train_and_save(trainer: SFTTrainer, config) -> dict:
    """Run training, save adapter + training log, return metrics.

    Returns:
        Dict with training log history, best eval loss, and timing info.
    """
    output_dir = os.path.join(config.output_dir, config.name)
    os.makedirs(output_dir, exist_ok=True)

    start = time.time()
    trainer.train()
    train_time = time.time() - start

    # Save adapter and tokenizer
    adapter_dir = os.path.join(output_dir, "adapter")
    trainer.model.save_pretrained(adapter_dir)
    trainer.processing_class.save_pretrained(adapter_dir)

    # Save training log
    log_path = os.path.join(output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # Extract summary metrics
    eval_losses = [
        e["eval_loss"] for e in trainer.state.log_history if "eval_loss" in e
    ]
    best_eval_loss = min(eval_losses) if eval_losses else None

    metrics = {
        "log_history": trainer.state.log_history,
        "best_eval_loss": best_eval_loss,
        "total_train_time_sec": round(train_time, 1),
        "config_name": config.name,
        "adapter_path": adapter_dir,
    }

    # Save summary
    summary_path = os.path.join(output_dir, "training_summary.json")
    summary = {k: v for k, v in metrics.items() if k != "log_history"}
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTraining complete: {config.name}")
    print(f"  Time: {train_time / 60:.1f} min")
    print(f"  Best eval loss: {best_eval_loss}")
    print(f"  Adapter saved to: {adapter_dir}")

    return metrics
