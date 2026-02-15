import re
from collections import Counter, defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from src.data_loader import LABEL_MAP, LETTER_TO_IDX, format_for_inference


def extract_answer_letter(generated_text: str) -> str | None:
    """Extract the answer letter (A/B/C/D) from generated text.

    Strategy (ordered):
      1. Exact single-letter match at start of generation (stripped)
      2. Regex for 'The answer is X' or 'Answer: X'
      3. First occurrence of A/B/C/D as a standalone letter
      4. None if extraction fails (counted as wrong)
    """
    text = generated_text.strip()

    # Strategy 1: First char is a valid answer letter
    if text and text[0] in "ABCD":
        return text[0]

    # Strategy 2: Common patterns
    patterns = [
        r"(?:the\s+)?answer\s*(?:is|:)\s*([A-D])",
        r"\b([A-D])\)",
        r"\b([A-D])\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Strategy 3: Any standalone A/B/C/D
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()

    return None


def run_inference_batch(model, tokenizer, prompts: list[str],
                        max_new_tokens: int = 5, batch_size: int = 8) -> list[str]:
    """Run batched inference and return raw generated strings."""
    model.eval()
    all_outputs = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=1.0,  # greedy (do_sample=False ignores this)
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        for j, output in enumerate(outputs):
            input_len = inputs["input_ids"][j].shape[0]
            new_tokens = output[input_len:]
            decoded = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_outputs.append(decoded)

    return all_outputs


def evaluate_on_dataset(model, tokenizer, dataset,
                        topic_labels: list[str] | None = None,
                        batch_size: int = 8,
                        max_new_tokens: int = 5) -> dict:
    """Full evaluation pipeline on a dataset split.

    Args:
        model: Fine-tuned or base model.
        tokenizer: Corresponding tokenizer.
        dataset: Raw HF dataset (formatting done here).
        topic_labels: Optional list of topic strings per example.
        batch_size: Inference batch size.
        max_new_tokens: Max tokens to generate.

    Returns dict with accuracy, F1, per-topic accuracy, predictions list, etc.
    """
    # Format prompts
    prompts = [format_for_inference(ex) for ex in dataset]
    gold_labels = [LABEL_MAP[ex["label"]] for ex in dataset]

    # Run inference
    print(f"Running inference on {len(prompts)} examples...")
    raw_outputs = run_inference_batch(
        model, tokenizer, prompts,
        max_new_tokens=max_new_tokens, batch_size=batch_size,
    )

    # Extract predictions
    predictions = []
    pred_labels = []
    extraction_failures = 0

    for i, (raw, gold) in enumerate(zip(raw_outputs, gold_labels)):
        pred = extract_answer_letter(raw)
        if pred is None:
            extraction_failures += 1
            pred = "X"  # placeholder for failed extraction
        pred_labels.append(pred)

        predictions.append({
            "idx": i,
            "gold": gold,
            "pred": pred,
            "raw_output": raw,
            "correct": pred == gold,
            "topic": topic_labels[i] if topic_labels else None,
        })

    # Compute metrics
    valid_mask = [p != "X" for p in pred_labels]
    # For accuracy, treat extraction failures as wrong
    correct = [p == g for p, g in zip(pred_labels, gold_labels)]
    overall_accuracy = sum(correct) / len(correct)

    # F1 (macro, treating extraction failures as a 5th class effectively)
    macro_f1 = f1_score(
        gold_labels, pred_labels, labels=["A", "B", "C", "D"],
        average="macro", zero_division=0,
    )

    # Per-topic accuracy
    per_topic_accuracy = {}
    if topic_labels:
        topic_correct = defaultdict(list)
        for pred_dict in predictions:
            topic = pred_dict["topic"]
            topic_correct[topic].append(pred_dict["correct"])
        per_topic_accuracy = {
            topic: sum(vals) / len(vals)
            for topic, vals in topic_correct.items()
            if len(vals) >= 3  # require minimum sample size
        }

    return {
        "overall_accuracy": round(overall_accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_topic_accuracy": per_topic_accuracy,
        "classification_report": classification_report(
            gold_labels, pred_labels, labels=["A", "B", "C", "D"],
            zero_division=0,
        ),
        "predictions": predictions,
        "extraction_failure_rate": round(extraction_failures / len(prompts), 4),
        "n_total": len(prompts),
        "n_correct": sum(correct),
    }


def compare_models(base_results: dict, finetuned_results: dict,
                   few_shot_results: dict | None = None) -> dict:
    """Build comparison summary between base, few-shot, and fine-tuned results."""
    comparison = {
        "zero_shot": {
            "accuracy": base_results["overall_accuracy"],
            "macro_f1": base_results["macro_f1"],
            "extraction_failures": base_results["extraction_failure_rate"],
        },
        "fine_tuned": {
            "accuracy": finetuned_results["overall_accuracy"],
            "macro_f1": finetuned_results["macro_f1"],
            "extraction_failures": finetuned_results["extraction_failure_rate"],
        },
        "delta": {
            "accuracy": round(
                finetuned_results["overall_accuracy"]
                - base_results["overall_accuracy"], 4
            ),
            "macro_f1": round(
                finetuned_results["macro_f1"]
                - base_results["macro_f1"], 4
            ),
        },
    }

    if few_shot_results:
        comparison["few_shot"] = {
            "accuracy": few_shot_results["overall_accuracy"],
            "macro_f1": few_shot_results["macro_f1"],
            "extraction_failures": few_shot_results["extraction_failure_rate"],
        }

    return comparison


def error_analysis(predictions: list[dict]) -> dict:
    """Classify errors into categories and compute error statistics.

    Categories:
      - extraction_failure: model didn't produce a valid A/B/C/D
      - other: all other errors (subdivided by topic if available)

    Also computes:
      - Most confused option pairs (gold -> pred)
      - Error rate by question length quartile
    """
    errors = [p for p in predictions if not p["correct"]]
    total_errors = len(errors)

    # Error categories
    extraction_failures = [e for e in errors if e["pred"] == "X"]
    substantive_errors = [e for e in errors if e["pred"] != "X"]

    # Confusion pairs
    confusion_pairs = Counter()
    for e in substantive_errors:
        confusion_pairs[(e["gold"], e["pred"])] += 1

    # Error rate by topic
    topic_errors = defaultdict(lambda: {"total": 0, "errors": 0})
    for p in predictions:
        topic = p.get("topic", "Unknown")
        topic_errors[topic]["total"] += 1
        if not p["correct"]:
            topic_errors[topic]["errors"] += 1

    topic_error_rates = {
        topic: round(d["errors"] / d["total"], 4) if d["total"] > 0 else 0
        for topic, d in topic_errors.items()
    }

    return {
        "total_errors": total_errors,
        "error_counts": {
            "extraction_failure": len(extraction_failures),
            "substantive_error": len(substantive_errors),
        },
        "most_confused_pairs": confusion_pairs.most_common(10),
        "topic_error_rates": topic_error_rates,
        "error_examples": {
            "extraction_failure": extraction_failures[:3],
            "substantive_error": substantive_errors[:5],
        },
    }


def confidence_calibration(model, tokenizer, dataset,
                           topic_labels: list[str] | None = None,
                           batch_size: int = 8) -> dict:
    """Compute confidence calibration by examining logits over A/B/C/D tokens.

    For each question, gets the softmax probability over the 4 option tokens
    at the first generated position and compares with actual correctness.

    Returns ECE (Expected Calibration Error), average confidence for correct
    and incorrect predictions, and binned calibration data.
    """
    model.eval()
    prompts = [format_for_inference(ex) for ex in dataset]
    gold_labels = [LABEL_MAP[ex["label"]] for ex in dataset]

    # Get token IDs for A, B, C, D
    option_token_ids = []
    for letter in ["A", "B", "C", "D"]:
        ids = tokenizer.encode(letter, add_special_tokens=False)
        option_token_ids.append(ids[-1])  # take last token if multi-token

    confidences = []
    pred_letters = []
    correctness = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Calibration"):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Get logits at the last position of each input
            logits = outputs.logits

        for j in range(len(batch_prompts)):
            # Last non-padding position
            attention_mask = inputs["attention_mask"][j]
            last_pos = attention_mask.sum().item() - 1
            token_logits = logits[j, last_pos, option_token_ids]
            probs = torch.softmax(token_logits, dim=-1).float().cpu().numpy()

            pred_idx = probs.argmax()
            pred_letter = "ABCD"[pred_idx]
            conf = probs[pred_idx]

            pred_letters.append(pred_letter)
            confidences.append(float(conf))
            gold_idx = i + j
            if gold_idx < len(gold_labels):
                correctness.append(pred_letter == gold_labels[gold_idx])

    confidences = np.array(confidences)
    correctness = np.array(correctness[:len(confidences)])

    # ECE computation with 10 bins
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    calibration_bins = []
    ece = 0.0

    for b in range(n_bins):
        mask = (confidences >= bin_boundaries[b]) & (confidences < bin_boundaries[b + 1])
        if b == n_bins - 1:  # include upper bound in last bin
            mask = mask | (confidences == bin_boundaries[b + 1])

        count = mask.sum()
        if count > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = correctness[mask].mean()
            ece += (count / len(confidences)) * abs(avg_acc - avg_conf)
            calibration_bins.append({
                "bin": b,
                "range": (float(bin_boundaries[b]), float(bin_boundaries[b + 1])),
                "count": int(count),
                "avg_confidence": float(avg_conf),
                "accuracy": float(avg_acc),
            })
        else:
            calibration_bins.append({
                "bin": b,
                "range": (float(bin_boundaries[b]), float(bin_boundaries[b + 1])),
                "count": 0,
                "avg_confidence": 0.0,
                "accuracy": 0.0,
            })

    # Average confidence for correct vs incorrect
    correct_mask = correctness.astype(bool)
    avg_conf_correct = float(confidences[correct_mask].mean()) if correct_mask.any() else 0.0
    avg_conf_incorrect = float(confidences[~correct_mask].mean()) if (~correct_mask).any() else 0.0

    return {
        "ece": round(float(ece), 4),
        "avg_confidence_correct": round(avg_conf_correct, 4),
        "avg_confidence_incorrect": round(avg_conf_incorrect, 4),
        "calibration_bins": calibration_bins,
        "n_samples": len(confidences),
    }
