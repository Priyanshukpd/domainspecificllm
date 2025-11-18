# Modular Training Structure

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  train_modular.py (Main)                │
│                    [250 lines]                          │
│         • Orchestrates entire training pipeline        │
│         • Combines all modules                         │
└───────────────────┬─────────────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   modules/ folder     │
        └───────────┬───────────┘
                    │
        ┌───────────┼───────────┬──────────────┐
        │           │           │              │
        ▼           ▼           ▼              ▼
   ┌────────┐  ┌──────────┐ ┌────────────┐ ┌───────┐
   │config.py│  │datasets.py│ │callbacks.py│ │utils.py│
   │100 lines│  │150 lines│ │350 lines   │ │50 lines│
   └────────┘  └──────────┘ └────────────┘ └───────┘
   • Hardware    • Text      • Logging      • Checkpoints
   • Env setup   • Tokenize  • Plotting     • Summaries
                • Sampling  • Early stop
```

## 📁 File Organization

The training code has been split into modular components for better maintainability:

```
.
├── train_modular.py       # Main training script (entry point)
├── train.py               # Original monolithic script (backup)
├── domain.txt             # Domain-specific training data
├── general.txt            # General knowledge data
├── output/                # Training outputs
└── modules/               # 📁 Modular components folder
    ├── __init__.py        # Package initialization
    ├── config.py          # Configuration & hardware detection
    ├── datasets.py        # Dataset classes (TextFileDataset, TokenizedDataset)
    ├── callbacks.py       # Training callbacks (logging, plotting, early stopping)
    └── utils.py           # Utility functions (checkpoints, summaries)
```

## 🧩 Module Breakdown

### `modules/config.py` (100 lines)
- **Purpose**: Environment setup & hardware detection
- **Key Functions**:
  - `detect_hardware()`: Auto-detects GPU/CPU and returns optimal config
- **Key Variables**:
  - `MODEL_PATH`: Path to Phi-3.5-mini model
  - Hardware-specific batch sizes, gradient accumulation, workers

### `modules/datasets.py` (150 lines)
- **Purpose**: Data loading and tokenization
- **Key Classes**:
  - `TextFileDataset`: Loads text files with weighted sampling (80/20)
  - `TokenizedDataset`: Tokenizes with stride to prevent data loss
  - `SubsetDataset`: Creates train/test splits

### `modules/callbacks.py` (350 lines)
- **Purpose**: Training monitoring and control
- **Key Classes**:
  - `StepLoggingCallback`: Clean logging every 10 steps
  - `SafeEarlyStoppingCallback`: Early stopping with min_steps protection
  - `PlottingCallback`: Real-time 4-panel training plots
  - `EnhancedEvalCallback`: Perplexity tracking + sample generation

### `modules/utils.py` (50 lines)
- **Purpose**: Helper functions
- **Key Functions**:
  - `get_latest_checkpoint()`: Finds latest checkpoint for auto-resume
  - `print_training_summary()`: Final training statistics

### `train_modular.py` (250 lines)
- **Purpose**: Main orchestration script
- **Key Functions**:
  - `load_model_and_tokenizer()`: Model setup
  - `load_and_tokenize_data()`: Data pipeline
  - `create_train_test_split()`: Dataset splitting
  - `create_training_args()`: TrainingArguments setup
  - `main()`: Full training orchestration

## 🚀 Usage

### Quick Start
```bash
# Run the modular version
python train_modular.py

# Or use the original monolithic version
python train.py
```

### Import as Library
```python
from modules.config import detect_hardware
from modules.datasets import TextFileDataset, TokenizedDataset
from modules.callbacks import SafeEarlyStoppingCallback

# Use components in your own scripts
hw_config = detect_hardware()
dataset = TextFileDataset(["data.txt"], weights=[1.0])
```

## ✅ Benefits of Modular Structure

1. **Maintainability**: Each file has a single responsibility
2. **Reusability**: Import callbacks/datasets into other projects
3. **Testability**: Easy to unit test individual modules
4. **Readability**: ~250 lines per file vs 900 lines monolithic
5. **Extensibility**: Add new callbacks/datasets without touching core logic

## 🔄 Migration Path

Both versions are functionally identical:
- **`train.py`**: Original monolithic script (backup)
- **`train_modular.py`**: New modular version (recommended)

You can use either - they produce the same results!

## 📊 Code Size Comparison

| Version | Lines | Files |
|---------|-------|-------|
| Original | ~900 | 1 |
| Modular | ~900 | 5 |

Same total code, but split into logical components!
