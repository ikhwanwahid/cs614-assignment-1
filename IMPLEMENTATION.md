## Implementation Documentation

### Overview

This project fine-tunes **Mistral-7B-Instruct-v0.3** on the **MedQA (USMLE)** dataset — 12,723 US Medical Licensing Exam multiple-choice questions — using **QLoRA** (Quantized Low-Rank Adaptation). The goal is to improve the model's medical question-answering accuracy compared to zero-shot and few-shot baselines.

---

### Architecture & Data Flow

```
notebooks/main_notebook.ipynb (orchestrator)
    │
    ├── configs/hyperparams.py    ← experiment configurations
    │
    ├── src/data_loader.py        ← load & format MedQA dataset
    ├── src/model_loader.py       ← load Mistral-7B with QLoRA
    ├── src/trainer.py            ← fine-tune with SFTTrainer
    ├── src/evaluator.py          ← inference + metrics
    ├── src/baselines.py          ← zero-shot & few-shot evaluation
    ├── src/topic_classifier.py   ← tag questions by medical topic
    └── src/utils.py              ← plotting, seeding, I/O
```

The notebook calls into `src/` modules sequentially. Each module has a single responsibility.

---

### File-by-File Breakdown

#### `configs/hyperparams.py` — Experiment Configuration

Defines `ExperimentConfig`, a dataclass holding every parameter for one experiment: model name, LoRA rank/alpha/dropout, learning rate, epochs, batch size, quantization settings, etc.

`get_all_configs()` returns 6 named configurations that systematically vary one or two hyperparameters each:

| Config | What it varies | Key params |
|--------|---------------|------------|
| 1 — baseline | Nothing (anchor point) | r=16, lr=2e-4, 2 epochs |
| 2 — low_rank | Fewer trainable params | r=8, alpha=16 |
| 3 — high_rank | More capacity | r=64, alpha=128, lr=1e-4 |
| 4 — low_lr | Slower learning | lr=5e-5, 3 epochs |
| 5 — longer_training | More epochs | 3 epochs |
| 6 — aggressive | Speed + regularization | r=32, lr=3e-4, dropout=0.1 |

**Used in notebook:** Section 5 (training sweep) loads all 6 configs and iterates through them.

---

#### `src/data_loader.py` — Dataset Loading & Formatting

**What it does:** Loads the MedQA dataset from HuggingFace and formats questions into Mistral's instruct template.

**Key functions:**
- `load_medqa_dataset()` — Loads train/val/test splits from `GBaker/MedQA-USMLE-4-options-hf`. Supports `subset_frac` for debugging with less data.
- `format_question(example)` — Converts raw dataset fields (`sent1`, `ending0`–`ending3`) into readable "Question + A/B/C/D options" text.
- `format_for_training(example)` — Wraps question + answer into Mistral instruct format: `<s>[INST] {system_prompt}\n\n{question} [/INST] {answer_letter}</s>`. Returns `{"text": ...}` for SFTTrainer.
- `format_for_inference(example)` — Same template but **without** the answer, so the model generates it: `<s>[INST] {system_prompt}\n\n{question} [/INST]`
- `build_few_shot_prompt(example, shots)` — Multi-turn instruct format with n exemplars before the target question. Used for few-shot baseline.
- `prepare_datasets(config)` — Convenience wrapper: loads data, formats train/val for training, returns test raw (formatted at inference time).

**Used in notebook:** Sections 2 (data exploration), 3–4 (baselines), 5 (training).

---

#### `src/model_loader.py` — Model Initialization with QLoRA

**What it does:** Loads Mistral-7B in 4-bit quantization and optionally applies LoRA adapters.

**Key functions:**
- `get_bnb_config(config)` — Creates `BitsAndBytesConfig` for 4-bit NF4 quantization with double quantization and bfloat16 compute dtype. This is what compresses the 14.5GB model to ~4-5GB in GPU memory.
- `get_lora_config(config)` — Creates `LoraConfig` targeting all attention + MLP projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- `load_base_model_and_tokenizer(config, for_training)` — Core loading function:
  - Always loads with 4-bit quantization
  - If `for_training=True`: applies `prepare_model_for_kbit_training()` (freezes base weights, enables gradient for LoRA) then wraps with `get_peft_model()`
  - If `for_training=False`: loads base model only (for baseline evaluation)
- `load_finetuned_model(config, adapter_path)` — Loads base model + merges a saved LoRA adapter on top. Used after training to evaluate the best config.

**Used in notebook:** Sections 3 (load base for baselines), 5 (load with LoRA for training), 7 (load fine-tuned for evaluation).

---

#### `src/trainer.py` — Training Pipeline

**What it does:** Sets up and runs HuggingFace's `SFTTrainer` (Supervised Fine-Tuning Trainer from the `trl` library).

**Key functions:**
- `get_training_args(config)` — Translates `ExperimentConfig` into `TrainingArguments`. Notable settings: gradient checkpointing (saves VRAM), cosine LR schedule, bfloat16, no external reporting.
- `create_trainer(model, tokenizer, train_ds, eval_ds, config)` — Instantiates `SFTTrainer` with early stopping (patience=3 eval steps). Uses the `"text"` field from formatted dataset. No packing (each example is a separate sequence).
- `train_and_save(trainer, config)` — Runs `trainer.train()`, then:
  1. Saves LoRA adapter + tokenizer to `results/{config_name}/adapter/`
  2. Saves full training log (loss at every logging step) to `training_log.json`
  3. Saves summary metrics to `training_summary.json`
  4. Returns metrics dict including `log_history` for plotting

**Used in notebook:** Section 5 (training loop over all 6 configs).

---

#### `src/evaluator.py` — Evaluation & Analysis

**What it does:** Runs batched inference, extracts answer letters, computes metrics, and performs error/calibration analysis.

**Key functions:**
- `extract_answer_letter(text)` — Parses model output to find A/B/C/D using a 3-strategy cascade:
  1. First character is A/B/C/D
  2. Regex patterns like "The answer is C" or "B)"
  3. Any standalone A/B/C/D
  4. Returns `None` if all fail (counted as wrong)

- `run_inference_batch(model, tokenizer, prompts)` — Batched generation with greedy decoding, max 5 new tokens. Decodes only the newly generated tokens (not the prompt).

- `evaluate_on_dataset(model, tokenizer, dataset)` — Full evaluation pipeline:
  1. Formats all examples as inference prompts
  2. Runs batched inference
  3. Extracts predicted letters
  4. Computes: overall accuracy, macro F1, per-topic accuracy, classification report, extraction failure rate
  5. Returns dict with all metrics + per-example predictions

- `compare_models(base, finetuned, few_shot)` — Builds a comparison summary dict (accuracy/F1/extraction failures for each model variant + deltas).

- `error_analysis(predictions)` — Classifies errors into extraction failures vs substantive errors. Computes most confused answer pairs (e.g. "A predicted as C: 47 times") and error rate by medical topic.

- `confidence_calibration(model, tokenizer, dataset)` — Gets softmax probabilities over A/B/C/D token IDs at the generation position. Computes Expected Calibration Error (ECE) with 10 bins, plus average confidence for correct vs incorrect predictions.

**Used in notebook:** Sections 3–4 (baseline eval), 7 (fine-tuned eval), 8 (comparison), 10 (error analysis), 11 (calibration).

---

#### `src/baselines.py` — Baseline Evaluation

**What it does:** Evaluates the base (non-fine-tuned) Mistral model under two prompting strategies.

**Key functions:**
- `run_zero_shot_baseline(model, tokenizer, test_ds)` — Simply calls `evaluate_on_dataset()` with the base model. Uses system prompt + question, no exemplars.
- `run_few_shot_baseline(model, tokenizer, test_ds, train_ds, n_shots=3)` — Randomly samples 3 training examples, builds multi-turn prompts using `build_few_shot_prompt()`, runs inference, and computes the same metrics. Uses smaller batch size (4) since few-shot prompts are longer.

**Used in notebook:** Sections 3 (zero-shot) and 4 (few-shot).

---

#### `src/topic_classifier.py` — Medical Topic Tagging

**What it does:** Assigns each question to one of 13 medical topics (Cardiology, Neurology, Oncology, etc.) using keyword matching. Falls back to "Other" if no keywords match.

**How it works:** For each topic, a list of keywords is checked against the question text. The topic with the most keyword hits wins. This is a heuristic — not ML-based — so it's fast but can misclassify multi-topic questions.

**Used in notebook:** Section 2 (topic distribution plot), and passed as `topic_labels` to evaluation functions for per-topic accuracy breakdown in sections 9–10.

---

#### `src/utils.py` — Utilities

**What it does:** Reproducibility setup, device detection, result I/O, and all visualizations.

**Key functions:**
- `set_seed(42)` — Seeds Python, NumPy, PyTorch, and CUDA for reproducibility
- `setup_device()` — Detects GPU and prints name/VRAM
- `save_results_json()` / `load_results_json()` — JSON serialization for results
- `plot_training_curves(log_history)` — Train + eval loss over steps for a single config
- `plot_all_configs_comparison(all_logs)` — Overlays eval loss from all 6 configs on one plot
- `plot_per_topic_accuracy(per_topic)` — Horizontal bar chart comparing base vs fine-tuned accuracy by topic
- `plot_error_taxonomy(error_counts)` — Bar chart of error categories
- `plot_calibration_curve(calibration_data)` — Reliability diagram (predicted confidence vs actual accuracy)

**Used in notebook:** Sections 1 (seed/device), 6 (training curves), 9 (topic accuracy), 10 (errors), 11 (calibration).

---

### Notebook Flow (sections map to files)

| Section | What happens | Files called |
|---------|-------------|-------------|
| 0 | Colab setup: clone repo, install deps | — |
| 1 | Imports, set seed, detect GPU | `utils.py`, all `src/` imports |
| 2 | Load MedQA, show examples, plot distributions | `data_loader.py`, `topic_classifier.py` |
| 3 | Zero-shot baseline | `model_loader.py`, `baselines.py` |
| 4 | 3-shot baseline | `baselines.py`, `data_loader.py` |
| 5 | Train all 6 configs | `hyperparams.py`, `data_loader.py`, `model_loader.py`, `trainer.py`, `evaluator.py` |
| 6 | Plot training curves | `utils.py` |
| 7 | Evaluate best config on test set | `model_loader.py`, `evaluator.py` |
| 8 | Compare all three approaches | `evaluator.py` |
| 9 | Per-topic accuracy breakdown | `utils.py` |
| 10 | Error analysis | `evaluator.py`, `utils.py` |
| 11 | Confidence calibration | `evaluator.py`, `utils.py` |
| 12 | Written summary, limitations, ethics | — (markdown) |

---

### Why QLoRA?

Full fine-tuning Mistral-7B requires ~56GB VRAM (7B params × 8 bytes for optimizer states). QLoRA makes it feasible on a single A100 40GB by:
1. **4-bit quantization** — compresses base weights from 14GB to ~4GB
2. **LoRA adapters** — only trains ~0.5-2% of parameters (small low-rank matrices injected into attention/MLP layers)
3. **Gradient checkpointing** — trades compute for memory by recomputing activations during backward pass

The result: training fits in ~15-20GB VRAM instead of 56GB+.
