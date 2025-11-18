# Domain-Adaptive Pretraining (DAPT) Project for Phi-3.5

## 🚀 Quick Start (3 Steps)
```bash
# 1. Verify setup (catches issues before long training)
python verify_setup.py

# 2. Start training (auto-detects A100-80GB or CPU server)
python train_simple.py

# 3. Monitor (GPU only)
watch -n 5 nvidia-smi
```

**Training Time**: 25-35 hours (A100-80GB) | 100-150 hours (CPU 16-core)

---

## Project Overview
This project performs domain-adaptive pretraining of Microsoft's Phi-3.5-mini-instruct (3.8B parameters) on insurance domain text (~200M tokens). The goal is to adapt the general-purpose model to insurance-specific terminology, policies, claims processing, and regulatory content.

**🚀 Ready to Run**: Code auto-detects A100-80GB or CPU server. No manual configuration needed.

**⚙️ Only 1 Thing to Change**: If model is not at `/Users/munishm/Documents/phi-3.5-mini-instruct/`, update `MODEL_PATH` in `train_simple.py` line 54.

## Core Architecture

### Model & Hardware
- **Base Model**: Phi-3.5-mini-instruct (3.8B params, decoder-only transformer)
- **Location**: `/Users/munishm/Documents/phi-3.5-mini-instruct/`
- **Hardware**: Single A100 GPU (40GB or 80GB variant)
- **Precision**: bfloat16 (A100 native support)

### Data Structure
- **Input Format**: Two `.txt` files
  - `general.txt`: General domain text (prevents catastrophic forgetting)
  - `domain.txt`: Insurance-specific corpus (~200M tokens)
- **Strategy**: Mixed training (80% domain, 20% general)
- **Token Length**: 512 tokens per sequence (truncated)

### Training Pipeline (`train_simple.py`)
```python
# Auto-detects hardware and optimizes for quality:
1. Load both text files using datasets library
2. Concatenate and shuffle for mixed training (80% domain, 20% general)
3. Tokenize with stride=50 to prevent 30-40% data loss on long paragraphs
4. Train with gradient accumulation (effective batch=32)
5. Early stopping (patience=3) monitors eval_loss
6. Conservative learning rate (2e-5) + weight decay (0.01) for quality
```

## Critical Parameters


## ⚠️ Data Preprocessing Strategy

### Problem: Data Loss with Simple Truncation

**Standard truncation discards everything beyond 512 tokens per line!**

If `domain.txt` has long paragraphs (>512 tokens), you're losing 30-40% of your 200M tokens.

### Solution: Tokenization with Stride

```python
def tokenize(examples):
    """Use stride to handle long texts without data loss"""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        stride=50,              # Overlap chunks by 50 tokens
        return_overflowing_tokens=True
    )

# Always use batched mapping with multiprocessing:
tokenized = dataset.map(
    tokenize, 
    batched=True, 
    remove_columns=["text"], 
    num_proc=4
)
```

### Batch Configuration (Adjust for GPU)
```python
per_device_train_batch_size=4      # Increase to 8 for A100-80GB
gradient_accumulation_steps=8      # Effective batch = 32
```

### Learning Rate Schedule
```python
learning_rate=2e-5                 # Conservative for domain shift
warmup_ratio=0.03                  # 3% warmup (good for 200M tokens)
lr_scheduler_type="cosine"         # Smooth decay
```

### Training Duration
```python
num_train_epochs=2                 # Sufficient for 200M tokens
early_stopping_patience=3          # Stop if no improvement for 3 evals
eval_steps=500                     # Evaluate every 500 steps
```



## Project-Specific Conventions

### Simple Over Complex
- **Philosophy**: Keep training script minimal (~80 lines)
- **No W&B by default**: `report_to="none"` (user adjusts if needed)
- **No fancy monitoring**: User handles with `nvidia-smi`
- **Direct file paths**: No config files, change parameters in-script

### Data Loading Pattern
```python
# For .txt files (not .jsonl):
dataset = load_dataset("text", data_files=PATH, split="train")

# For multiple files:
ds1 = load_dataset("text", data_files="file1.txt", split="train")
ds2 = load_dataset("text", data_files="file2.txt", split="train")
combined = concatenate_datasets([ds1, ds2])
```



## Expected Workflow

### 1. Setup
```bash
pip install transformers datasets torch accelerate
```

### 2. Prepare Data
- Place `general.txt` and `domain.txt` in project root
- Verify domain.txt has ~200M tokens (check file size: ~150-200MB)

### 2.5. Verify Setup (IMPORTANT - Run Before Training!)
```bash
python verify_setup.py
# This checks:
# - Hardware detection (GPU/CPU)
# - Model path exists
# - Data files present with token counts
# - Disk space for checkpoints
# - Catches issues BEFORE 25-150 hour training starts
```

### 3. Run Training

**On A100-80GB:**
```bash
python train_simple.py  # Auto-detects GPU, uses batch=8, ~25-35 hours
```

**On CPU server (16+ cores, 96GB RAM):**
```bash
# Recommended: Run in screen/tmux for long training
screen -S training
python train_simple.py  # Auto-detects CPU, uses batch=1, ~100-150 hours
# Detach: Ctrl+A then D
# Reattach: screen -r training
```

### 4. Monitor

**On GPU:**
```bash
# Separate terminal:
watch -n 5 nvidia-smi
```

**On CPU:**
```bash
# Monitor CPU usage:
htop
# Or check training logs:
tail -f logs/events.out.tfevents.*
```

### 5. Model Output
- Checkpoints: `./output/checkpoint-{step}/`
- Final model: `./output/final/`

## Performance Expectations

### A100-80GB (Recommended for Quality)
- Batch size: 8 (auto-detected)
- Gradient accumulation: 4
- Effective batch: 32
- Training time: 25-35 hours
- Tokens/sec: 50,000-70,000
- VRAM usage: 45-55 GB
- **Best for**: Fast + high quality

### A100-40GB
- Batch size: 4 (auto-detected)
- Gradient accumulation: 8
- Effective batch: 32
- Training time: 35-45 hours
- Tokens/sec: 35,000-50,000
- VRAM usage: 32-38 GB
- **Best for**: Budget-conscious quality training

### CPU Server (16+ cores, 96GB RAM)
- Batch size: 1 (auto-detected)
- Gradient accumulation: 32
- Effective batch: 32
- Training time: 100-150 hours
- Tokens/sec: 5,000-8,000
- RAM usage: 40-60 GB
- **Best for**: No GPU access but plenty of time

## Success Metrics

## Post-Training Evaluation

### Quick Domain Test

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/final/")
tokenizer = AutoTokenizer.from_pretrained("./output/final/")

# Test domain adaptation:
prompt = "An insurance claim for water damage must include"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0]))

# Compare with base model:
base_model = AutoModelForCausalLM.from_pretrained("/Users/munishm/Documents/phi-3.5-mini-instruct/")
base_outputs = base_model.generate(**inputs, max_length=100, do_sample=True, temperature=0.7)
print("Base model:", tokenizer.decode(base_outputs[0]))
```

### During Training
- **Training loss**: Should decrease steadily
- **Eval loss**: Should improve, trigger early stopping when plateaus
- **GPU utilization**: Should be 80-95%

### Post-Training
- **Perplexity improvement**: 40-60% on insurance test set
- **General knowledge retention**: General perplexity shouldn't explode
- **Insurance terminology**: Model should complete domain-specific prompts coherently

## Common Adjustments

### If Out of Memory
```python
per_device_train_batch_size=2          # Reduce from 4
gradient_accumulation_steps=16         # Increase to maintain effective batch
gradient_checkpointing=True            # Add to TrainingArguments
```

### If Training Too Slow
```python
per_device_train_batch_size=8          # Increase (A100-80GB only)
dataloader_num_workers=12              # More parallel data loading
```

### If Overfitting
```python
num_train_epochs=1                     # Reduce from 2
weight_decay=0.01                      # Add regularization
```

## Files Not to Modify
- Base model files in `/Users/munishm/Documents/phi-3.5-mini-instruct/`
- Only modify: `train_simple.py` and training parameters

## Key Dependencies
```python
transformers>=4.36.0    # For Phi-3.5 support
torch>=2.1.0           # CUDA 12.x compatible
datasets>=2.14.0       # For efficient data loading
```

## Anti-Patterns (Avoid)
- ❌ **Don't use truncation without stride** (loses 30-40% of long documents - see Data Preprocessing section)
- ❌ Don't use fp16 on A100 (use bf16 instead)
- ❌ Don't train on domain.txt only (mix with general.txt)
- ❌ Don't use small batch sizes if you have VRAM headroom
- ❌ Don't set num_train_epochs too high (2 is enough for 200M tokens)

## Debugging Tips

### If loss is NaN
- Check learning rate (try 1e-5)
- Enable gradient clipping: `max_grad_norm=1.0` (already default)

### If GPU underutilized
- Increase `dataloader_num_workers`
- Increase batch size
- Check data loading isn't bottleneck

### If general knowledge degraded
- Increase ratio of general.txt in mix
- Reduce learning rate
- Train for fewer epochs