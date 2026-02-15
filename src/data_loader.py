from datasets import load_dataset, Dataset

SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following USMLE-style "
    "multiple-choice question by selecting the single best answer. "
    "Respond with ONLY the letter (A, B, C, or D) of the correct answer."
)

LABEL_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_TO_IDX = {"A": 0, "B": 1, "C": 2, "D": 3}


def load_medqa_dataset(
    dataset_name: str = "GBaker/MedQA-USMLE-4-options-hf",
    subset_frac: float = 1.0,
    seed: int = 42,
) -> tuple[Dataset, Dataset, Dataset]:
    """Load and return train, validation, test splits.

    Args:
        dataset_name: HuggingFace dataset identifier.
        subset_frac: Fraction of training data to use (for debugging).
        seed: Random seed for subsetting.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    ds = load_dataset(dataset_name)
    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    if subset_frac < 1.0:
        n = max(1, int(len(train_ds) * subset_frac))
        train_ds = train_ds.shuffle(seed=seed).select(range(n))

    return train_ds, val_ds, test_ds


def format_question(example: dict) -> str:
    """Format a single MedQA example into the question + options text.

    Returns:
        Question text followed by labeled options A-D.
    """
    question = example["sent1"]
    options = (
        f"A) {example['ending0']}\n"
        f"B) {example['ending1']}\n"
        f"C) {example['ending2']}\n"
        f"D) {example['ending3']}"
    )
    return f"{question}\n\n{options}"


def format_for_training(example: dict) -> dict:
    """Format a single example into Mistral instruct format for SFTTrainer.

    Returns dict with 'text' key containing the full conversation:
        <s>[INST] {system + question} [/INST] {answer_letter}</s>
    """
    question_text = format_question(example)
    answer_letter = LABEL_MAP[example["label"]]
    user_content = f"{SYSTEM_PROMPT}\n\n{question_text}"
    full_text = f"<s>[INST] {user_content} [/INST] {answer_letter}</s>"
    return {"text": full_text}


def format_for_inference(example: dict) -> str:
    """Format for inference -- no answer appended.

    Returns:
        '<s>[INST] {system + question} [/INST]'
    """
    question_text = format_question(example)
    user_content = f"{SYSTEM_PROMPT}\n\n{question_text}"
    return f"<s>[INST] {user_content} [/INST]"


def build_few_shot_prompt(example: dict, few_shot_examples: list[dict],
                          n_shots: int = 3) -> str:
    """Build a few-shot prompt with n_shots exemplars before the target question.

    Uses multi-turn Mistral instruct format:
        <s>[INST] {system + exemplar_1_q} [/INST] {exemplar_1_a}</s>
        [INST] {exemplar_2_q} [/INST] {exemplar_2_a}</s>
        ...
        [INST] {target_q} [/INST]
    """
    shots = few_shot_examples[:n_shots]
    parts = []

    for i, shot in enumerate(shots):
        q = format_question(shot)
        a = LABEL_MAP[shot["label"]]
        if i == 0:
            parts.append(f"<s>[INST] {SYSTEM_PROMPT}\n\n{q} [/INST] {a}</s>")
        else:
            parts.append(f"[INST] {q} [/INST] {a}</s>")

    # Target question (no answer)
    target_q = format_question(example)
    parts.append(f"[INST] {target_q} [/INST]")

    return "\n".join(parts)


def prepare_datasets(config) -> tuple[Dataset, Dataset, Dataset]:
    """Load, format for training, and return train/val/test datasets.

    Note: test_ds is returned RAW (unformatted) -- formatting happens
    at inference time in the evaluator.
    """
    train_ds, val_ds, test_ds = load_medqa_dataset(
        dataset_name=config.dataset_name,
        subset_frac=config.train_subset_frac,
        seed=config.seed,
    )
    train_formatted = train_ds.map(format_for_training)
    val_formatted = val_ds.map(format_for_training)
    return train_formatted, val_formatted, test_ds
