import random

from src.data_loader import build_few_shot_prompt, format_for_inference
from src.evaluator import evaluate_on_dataset, extract_answer_letter, run_inference_batch
from src.data_loader import LABEL_MAP


def run_zero_shot_baseline(model, tokenizer, test_dataset,
                           topic_labels: list[str] | None = None,
                           batch_size: int = 8) -> dict:
    """Evaluate the base (non-fine-tuned) model with zero-shot prompting.

    Uses the same system prompt as fine-tuning but no exemplars.
    """
    return evaluate_on_dataset(
        model, tokenizer, test_dataset,
        topic_labels=topic_labels, batch_size=batch_size,
    )


def run_few_shot_baseline(model, tokenizer, test_dataset, train_dataset,
                          n_shots: int = 3,
                          topic_labels: list[str] | None = None,
                          batch_size: int = 4,
                          seed: int = 42) -> dict:
    """Evaluate the base model with n-shot prompting.

    Selects n_shots random examples from the training set as exemplars.
    Uses multi-turn Mistral instruct format.
    """
    rng = random.Random(seed)
    shot_indices = rng.sample(range(len(train_dataset)), n_shots)
    few_shot_examples = [train_dataset[i] for i in shot_indices]

    # Build prompts with exemplars
    prompts = [
        build_few_shot_prompt(ex, few_shot_examples, n_shots=n_shots)
        for ex in test_dataset
    ]
    gold_labels = [LABEL_MAP[ex["label"]] for ex in test_dataset]

    # Run inference
    print(f"Running {n_shots}-shot baseline on {len(prompts)} examples...")
    raw_outputs = run_inference_batch(
        model, tokenizer, prompts,
        max_new_tokens=5, batch_size=batch_size,
    )

    # Extract and score
    from collections import defaultdict
    predictions = []
    pred_labels = []
    extraction_failures = 0

    for i, (raw, gold) in enumerate(zip(raw_outputs, gold_labels)):
        pred = extract_answer_letter(raw)
        if pred is None:
            extraction_failures += 1
            pred = "X"
        pred_labels.append(pred)
        predictions.append({
            "idx": i,
            "gold": gold,
            "pred": pred,
            "raw_output": raw,
            "correct": pred == gold,
            "topic": topic_labels[i] if topic_labels else None,
        })

    correct = [p == g for p, g in zip(pred_labels, gold_labels)]
    overall_accuracy = sum(correct) / len(correct)

    from sklearn.metrics import classification_report, f1_score
    macro_f1 = f1_score(
        gold_labels, pred_labels, labels=["A", "B", "C", "D"],
        average="macro", zero_division=0,
    )

    per_topic_accuracy = {}
    if topic_labels:
        topic_correct = defaultdict(list)
        for p in predictions:
            topic_correct[p["topic"]].append(p["correct"])
        per_topic_accuracy = {
            topic: sum(vals) / len(vals)
            for topic, vals in topic_correct.items()
            if len(vals) >= 3
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
