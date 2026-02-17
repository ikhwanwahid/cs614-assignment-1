# MedQA Fine-Tuning: Teaching an LLM to Pass a Medical Exam

Fine-tune Mistral-7B-Instruct-v0.3 on USMLE-style medical MCQs using QLoRA. CS614 - Generative AI with LLMs.

---

## Project Structure

```
cs614-assignment-1/
├── configs/
│   ├── __init__.py
│   └── hyperparams.py         # 6 experiment configurations (LoRA rank, LR, epochs, dropout)
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Load MedQA dataset, format prompts (train + inference)
│   ├── model_loader.py         # Load base model with 4-bit quantization + LoRA adapters
│   ├── trainer.py              # SFTTrainer setup with SFTConfig, train & save adapter
│   ├── evaluator.py            # Batched inference, answer extraction, metrics (accuracy, F1, ECE)
│   ├── baselines.py            # Zero-shot and few-shot evaluation pipelines
│   ├── topic_classifier.py     # Keyword-based medical topic tagger (13 specialties)
│   └── utils.py                # Plotting, JSON save/load, seed setting, GPU info
├── notebooks/
│   ├── main_notebook.ipynb     # Full pipeline: train + evaluate + analysis (run this on Colab)
│   └── report.ipynb            # Markdown-only report for submission (Section 2.3)
├── results/                    # Generated after training (JSON metrics, PNGs, adapters)
├── pyproject.toml              # Python dependencies
├── IMPLEMENTATION.md           # Detailed explanation of each module
└── README.md                   # This file
```

### What each file does

| File | Purpose |
|------|---------|
| `configs/hyperparams.py` | Defines the `ExperimentConfig` dataclass and 6 preset configurations that vary LoRA rank (8-64), learning rate (5e-5 to 3e-4), epochs (2-3), and dropout (0.05-0.1). |
| `src/data_loader.py` | Downloads MedQA from HuggingFace, formats questions into the Mistral `[INST]...[/INST]` template for training and inference, and builds few-shot prompts. |
| `src/model_loader.py` | Loads Mistral-7B with 4-bit NF4 quantization via BitsAndBytes. For training, attaches LoRA adapters. For inference, loads a saved adapter from disk. |
| `src/trainer.py` | Creates an `SFTTrainer` (from the `trl` library) with `SFTConfig`, early stopping, and gradient checkpointing. The `train_and_save` function runs training and saves the adapter + logs. |
| `src/evaluator.py` | Runs batched greedy inference, extracts answer letters (A/B/C/D) from generated text using regex, computes accuracy/F1/per-topic metrics, confusion analysis, and confidence calibration (ECE). |
| `src/baselines.py` | Wraps `evaluator.py` to run zero-shot and n-shot baselines on the base (non-fine-tuned) model. |
| `src/topic_classifier.py` | Tags each test question with one of 13 medical specialties (Cardiology, Neurology, etc.) based on keyword matching. |
| `src/utils.py` | Utility functions for plotting training curves, calibration diagrams, per-topic accuracy charts, error taxonomy, and JSON I/O. |
| `notebooks/main_notebook.ipynb` | The main orchestrator. Runs Sections 0-14: setup, data exploration, baselines, training (6 configs), evaluation, per-topic analysis, error analysis, calibration, position bias, prompt sensitivity, and conclusions. |
| `notebooks/report.ipynb` | A clean markdown-only report summarizing task, dataset, model choice, fine-tuning process, evaluation results, and analysis. No code cells. |

---

## How to Run (Google Colab)

### Prerequisites

- A Google account with access to Google Colab
- A GPU runtime (H100/A100 recommended for training; T4 works but is slower)
- ~10 GB of free Google Drive space (for saving model adapters and results)

### Option A: Clone from GitHub (this is how I ran it)

> This is the workflow used for this assignment. Code was edited locally, pushed to GitHub, then pulled into Colab for execution on GPU.

1. Open Google Colab: https://colab.research.google.com/

2. Create a **new notebook** (or upload `notebooks/main_notebook.ipynb` directly).

3. **Set the runtime to GPU:**
   - Go to `Runtime` > `Change runtime type` > Select **T4** or **A100** or **H100** > Save

4. Run the **setup cell** (Section 0 in `main_notebook.ipynb`). It will:
   - Mount your Google Drive
   - Ask for a GitHub Personal Access Token (PAT) — you need at least `repo` read access
   - Clone the repository to `/content/cs614-assignment-1`
   - Install all Python dependencies
   - Set up `RESULTS_DIR` on Google Drive for persistent storage

   ```python
   # This is already the first code cell in main_notebook.ipynb
   # It handles everything automatically
   ```

5. **Run all cells sequentially** from top to bottom (`Runtime` > `Run all`).

6. Results (JSON metrics, plots, model adapters) are saved to `/content/drive/MyDrive/cs614_results/` so they survive runtime disconnects.

### Option B: Upload as a ZIP

> Use this if you received the project as a zip file and don't have GitHub access.

1. **Unzip** the project on your local machine. You should see the folder structure shown above.

2. Open Google Colab: https://colab.research.google.com/

3. **Upload the project folder to Google Drive:**
   - Go to https://drive.google.com/
   - Upload the entire `cs614-assignment-1/` folder to your Drive root (or any location you prefer)

4. **Upload the notebook:**
   - In Colab, go to `File` > `Upload notebook`
   - Upload `notebooks/main_notebook.ipynb`

5. **Set the runtime to GPU:**
   - `Runtime` > `Change runtime type` > Select **T4** or **A100** or **H100** > Save

6. **Replace the setup cell** (the first code cell) with this:

   ```python
   import os
   import sys

   IN_COLAB = True

   # Mount Google Drive
   from google.colab import drive
   drive.mount("/content/drive")

   # Point to your uploaded project folder
   # CHANGE THIS PATH if you uploaded to a different location
   PROJECT_ROOT = "/content/drive/MyDrive/cs614-assignment-1"
   os.chdir(PROJECT_ROOT)

   # Install dependencies
   !pip install -q transformers datasets peft bitsandbytes trl accelerate
   !pip install -q scikit-learn matplotlib pandas tqdm scipy

   # Add project root to Python path
   if PROJECT_ROOT not in sys.path:
       sys.path.insert(0, PROJECT_ROOT)

   # Results directory (persistent on Drive)
   RESULTS_DIR = "/content/drive/MyDrive/cs614_results"
   os.makedirs(RESULTS_DIR, exist_ok=True)

   print(f"Project root: {PROJECT_ROOT}")
   print(f"Results dir: {RESULTS_DIR}")
   print(f"Running on Colab: {IN_COLAB}")
   ```

7. **Run all cells sequentially** from top to bottom.

---

## How to Run Locally (Option C)

> Use this if you have a local machine with an NVIDIA GPU (24+ GB VRAM recommended) or want to run analysis-only sections on CPU.

### Requirements

- Python 3.13+ (managed by `.python-version`)
- [uv](https://docs.astral.sh/uv/) package manager (recommended) or pip
- NVIDIA GPU with CUDA support (for training and inference)
  - **24 GB+ VRAM** (e.g., RTX 3090/4090) — can train with reduced batch size
  - **40 GB+ VRAM** (e.g., A100/H100) — can train with default settings
  - **No GPU** — can run position bias analysis (Section 12) and view the report, but not training or inference

### Setup with uv

```bash
# Clone the repo (or unzip the project folder)
git clone https://github.com/ikhwanwahid/cs614-assignment-1.git
cd cs614-assignment-1

# Create virtual environment and install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate

# Install Jupyter and register the kernel so the notebook can use this environment
uv pip install jupyter ipykernel
python -m ipykernel install --user --name medqa-finetune --display-name "MedQA Fine-Tune (Python 3.13)"
```

`uv sync` reads `pyproject.toml` and `uv.lock` to install exact dependency versions, ensuring reproducibility.

### Setup with pip (alternative)

```bash
git clone https://github.com/ikhwanwahid/cs614-assignment-1.git
cd cs614-assignment-1

python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

pip install transformers datasets peft bitsandbytes trl accelerate
pip install scikit-learn matplotlib pandas tqdm scipy
pip install jupyter ipykernel
python -m ipykernel install --user --name medqa-finetune --display-name "MedQA Fine-Tune (Python 3.13)"
```

### Running the notebook locally

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open** `notebooks/main_notebook.ipynb`

3. **Select the correct kernel:**
   - In Jupyter, go to `Kernel` > `Change kernel` > Select **MedQA Fine-Tune (Python 3.13)**
   - This ensures the notebook uses the virtual environment with all dependencies installed

4. **The setup cell auto-detects local vs Colab.** When `IN_COLAB` is `False`, it:
   - Skips Google Drive mounting and repo cloning
   - Sets `PROJECT_ROOT` to the parent of the current working directory
   - Sets `RESULTS_DIR` to `results/` within the project folder

5. **Run all cells sequentially.**

### Local GPU considerations

| GPU VRAM | What works | Adjustments needed |
|----------|-----------|-------------------|
| **80 GB** (H100/A100) | Everything, default settings | None |
| **40 GB** (A100-40GB) | Everything, default settings | None |
| **24 GB** (RTX 3090/4090) | Training + inference | Reduce `per_device_train_batch_size` to 2 and increase `gradient_accumulation_steps` to 8 in `configs/hyperparams.py` |
| **16 GB** (RTX 4080/T4) | Training may OOM | Reduce batch size to 1, gradient accumulation to 16. Consider `max_seq_length=512` |
| **No GPU (CPU only)** | Sections 12 (position bias) only | Skip all training and inference sections. Load pre-computed results from `results/` directory |

### Running on CPU with pre-computed results

If you don't have a GPU but want to explore the analysis, the `results/` directory contains all pre-computed outputs. You can load them directly:

```python
from src.utils import load_results_json

zero_shot_results = load_results_json("results/zero_shot_results.json")
few_shot_results = load_results_json("results/few_shot_results.json")
ft_results = load_results_json("results/finetuned_test_results.json")
calibration_data = load_results_json("results/calibration_data.json")
template_results = load_results_json("results/prompt_template_results.json")
```

Then run the analysis cells (Sections 8-12, 14) which only use these saved results — no GPU needed.

### macOS (Apple Silicon) note

BitsAndBytes 4-bit quantization requires CUDA and does **not** work on Apple Silicon (M1/M2/M3). You can:
- Use Google Colab for training (Options A or B above)
- Run analysis-only sections locally using pre-computed results

---

## What to Expect

### Runtime Estimates (H100 80GB)

| Section | Time |
|---------|------|
| Setup + data loading | ~2 min |
| Zero-shot baseline | ~3 min |
| 3-shot baseline | ~5 min |
| Training (6 configs) | ~3.5 hours |
| Evaluation + analysis | ~15 min |
| Position bias analysis | ~1 min (no GPU needed) |
| Prompt template sensitivity | ~10 min |
| **Total** | **~4 hours** |

On a T4 GPU, expect roughly 2-3x longer for training.

### Output Files

After a complete run, `RESULTS_DIR` will contain:

```
cs614_results/
├── zero_shot_results.json          # Zero-shot baseline metrics + predictions
├── few_shot_results.json           # 3-shot baseline metrics + predictions
├── finetuned_test_results.json     # Best config test set results
├── comparison_summary.json         # Zero-shot vs few-shot vs fine-tuned summary
├── calibration_data.json           # ECE and calibration bin data
├── prompt_template_results.json    # Accuracy per prompt template variant
├── all_configs_comparison.png      # Overlaid eval loss curves
├── per_topic_accuracy.png          # Per-topic bar chart (base vs fine-tuned)
├── calibration_curve.png           # Reliability diagram
├── error_taxonomy.png              # Error category breakdown
├── position_bias.png               # Gold vs predicted distribution
├── confusion_matrix_*.png          # Confusion matrices (zero-shot, 3-shot, fine-tuned)
├── prompt_template_comparison.png  # Accuracy by prompt template
├── config_1_baseline/              # Config-specific results
│   ├── adapter/                    # Saved LoRA adapter weights
│   ├── training_log.json           # Full training log history
│   ├── training_summary.json       # Best eval loss, training time
│   ├── training_curve.png          # Train/eval loss plot
│   └── val_results.json            # Validation set metrics
├── config_2_low_rank/              # (same structure)
├── config_3_high_rank/
├── config_4_low_lr/                # Best config
├── config_5_longer_training/
└── config_6_aggressive/
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Make sure you ran `os.chdir()` to the project root before importing. The project root must be in `sys.path`. |
| `NameError: name 'configs' is not defined` | Run the imports cell (Section 1) and `configs = get_all_configs()` before using `configs[...]`. |
| `TypeError: SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'` | The `trl` library v0.28+ renamed `tokenizer` to `processing_class`. Run `pip install -q trl>=0.28` and restart runtime. |
| `TypeError: Got unsupported ScalarType BFloat16` | Already fixed in `src/evaluator.py` — the code converts to float32 before calling `.numpy()`. If you see this, make sure you pulled the latest code. |
| Runtime disconnects during training | Results are saved to Google Drive after each config. Reconnect, re-run Sections 0-2, load results from Drive with `load_results_json()`, and resume from where you left off. |
| `CUDA out of memory` | Reduce `per_device_train_batch_size` from 4 to 2 in `configs/hyperparams.py` (and increase `gradient_accumulation_steps` to 8 to maintain effective batch size). |
| No GPU available | Training requires a GPU. For Colab: `Runtime` > `Change runtime type` > select a GPU. The position bias analysis (Section 12) can run on CPU since it only uses saved predictions. |
| Jupyter kernel not finding packages | Make sure you registered the kernel with `python -m ipykernel install --user --name medqa-finetune` and selected it in the notebook via `Kernel` > `Change kernel`. |

---

## Dependencies

Installed automatically by the setup cell (Colab) or `uv sync` / `pip install` (local):

```
transformers      # HuggingFace model loading and tokenization
datasets          # HuggingFace dataset loading
peft              # LoRA adapter management
bitsandbytes      # 4-bit quantization
trl               # SFTTrainer for supervised fine-tuning
accelerate        # Training acceleration
scikit-learn      # Classification metrics
matplotlib        # Plotting
pandas            # DataFrames for result tables
tqdm              # Progress bars
scipy             # Chi-squared statistical tests
```
