import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_device() -> torch.device:
    """Return available device and print GPU info."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("No GPU available, using CPU")
    return device


def save_results_json(results: dict, filepath: str):
    """Save results dict to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2, default=str)


def load_results_json(filepath: str) -> dict:
    """Load results dict from JSON."""
    with open(filepath) as f:
        return json.load(f)


def plot_training_curves(log_history: list, title: str = "Training Curves",
                         save_path: str | None = None):
    """Plot train loss and eval loss from a single trainer's log_history."""
    train_steps = [e["step"] for e in log_history if "loss" in e]
    train_loss = [e["loss"] for e in log_history if "loss" in e]
    eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
    eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_steps, train_loss, label="Train Loss", alpha=0.8)
    if eval_loss:
        ax.plot(eval_steps, eval_loss, label="Eval Loss", marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_all_configs_comparison(all_logs: dict[str, list],
                                save_path: str | None = None):
    """Overlay eval loss curves from all configs on one plot."""
    fig, ax = plt.subplots(figsize=(12, 6))
    for config_name, log_history in all_logs.items():
        eval_steps = [e["step"] for e in log_history if "eval_loss" in e]
        eval_loss = [e["eval_loss"] for e in log_history if "eval_loss" in e]
        if eval_loss:
            ax.plot(eval_steps, eval_loss, label=config_name, marker="o",
                    markersize=3)

    ax.set_xlabel("Step")
    ax.set_ylabel("Eval Loss")
    ax.set_title("Eval Loss Across All Configurations")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_per_topic_accuracy(per_topic: dict[str, float],
                            base_per_topic: dict[str, float] | None = None,
                            title: str = "Per-Topic Accuracy",
                            save_path: str | None = None):
    """Horizontal bar chart of accuracy by medical topic."""
    topics = sorted(per_topic.keys(), key=lambda t: per_topic[t])
    accs = [per_topic[t] for t in topics]

    fig, ax = plt.subplots(figsize=(10, max(6, len(topics) * 0.4)))
    y_pos = np.arange(len(topics))
    bar_height = 0.35

    if base_per_topic:
        base_accs = [base_per_topic.get(t, 0) for t in topics]
        ax.barh(y_pos + bar_height / 2, base_accs, bar_height, label="Base",
                color="#d4a0a0", alpha=0.8)
        ax.barh(y_pos - bar_height / 2, accs, bar_height, label="Fine-tuned",
                color="#4a90d9", alpha=0.8)
        ax.legend()
    else:
        ax.barh(y_pos, accs, color="#4a90d9", alpha=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(topics)
    ax.set_xlabel("Accuracy")
    ax.set_title(title)
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.25, color="red", linestyle="--", alpha=0.5, label="Random (25%)")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_error_taxonomy(error_counts: dict, save_path: str | None = None):
    """Bar chart of error categories."""
    categories = list(error_counts.keys())
    counts = list(error_counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#9b59b6"]
    ax.bar(categories, counts, color=colors[:len(categories)], alpha=0.8)
    ax.set_ylabel("Count")
    ax.set_title("Error Taxonomy")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig


def plot_calibration_curve(calibration_data: dict, save_path: str | None = None):
    """Reliability diagram: predicted confidence vs actual accuracy."""
    bins = calibration_data["calibration_bins"]
    bin_confs = [b["avg_confidence"] for b in bins if b["count"] > 0]
    bin_accs = [b["accuracy"] for b in bins if b["count"] > 0]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.bar(bin_confs, bin_accs, width=0.08, alpha=0.6, color="#4a90d9",
           edgecolor="black", label="Model")
    ax.set_xlabel("Mean Predicted Confidence")
    ax.set_ylabel("Fraction of Correct Predictions")
    ax.set_title(
        f"Calibration (ECE = {calibration_data['ece']:.3f})"
    )
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
