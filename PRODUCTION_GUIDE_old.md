# Production Training Guide

## Your Setup Options

### Option 1: A100-80GB GPU Server (Recommended for Speed + Quality)
- **Time**: 25-35 hours
- **Cost**: ~$1.10/hour on Lambda Labs = $30-40 total
- **Quality**: Excellent (bf16 precision, batch=8)

### Option 2: CPU Server (16 cores, 96GB RAM)
- **Time**: 100-150 hours (4-6 days)
- **Cost**: If you own server, just electricity
- **Quality**: Excellent (fp32 precision, batch=1, high grad accumulation)

---

## Step-by-Step Instructions

### 1. Prepare Your Data Files

You need TWO files in project root:

#### `domain.txt` (Insurance domain - 200M tokens)
```
Target size: 150-200 MB plain text
Format: Plain text, UTF-8, one paragraph per line or continuous text
Content: Insurance policies, claims, regulations, underwriting docs
```

#### `general.txt` (General knowledge - 50M tokens)
```
Target size: 40-50 MB plain text
Format: Plain text, UTF-8
Content: Wikipedia, news, general knowledge (prevents forgetting)
```

**Creating these files:**
```bash
# If you have multiple source files:
cat insurance_doc1.txt insurance_doc2.txt ... > domain.txt
cat wikipedia_sample.txt news_articles.txt ... > general.txt

# Check sizes:
ls -lh domain.txt general.txt
```

---

### 2. Verify Setup (CRITICAL - Saves You From Wasted Time)

```bash
python verify_setup.py
```

**What it checks:**
- ✅ Hardware detected correctly (GPU or CPU)
- ✅ Model files exist at correct path
- ✅ domain.txt and general.txt exist with token counts
- ✅ Sufficient disk space for checkpoints (~30GB)
- ✅ Stride configuration will work

**Expected output:**
```
🔍 Pre-Training Verification
====================
1️⃣  Hardware Check:
   ✅ GPU: NVIDIA A100-SXM4-80GB (80GB)
   ✅ Recommended: A100-80GB detected, batch=8, ~25-35h

2️⃣  Model Check:
   ✅ Model found: /Users/munishm/Documents/phi-3.5-mini-instruct/
   ✅ Tokenizer vocab size: 32064
   ✅ Pad token: <|endoftext|> (id: 32000)

3️⃣  Data Files Check:
   ✅ domain.txt: 180.5MB, 1234 lines
   ✅ general.txt: 45.2MB, 456 lines
   📊 Estimated tokens: 210,456,789
      Domain: 168,365,431 (80.0%)
      General: 42,091,358 (20.0%)
   ✅ Good: 210.5M tokens

4️⃣  Stride Configuration Check:
   First line tokens: 1250
   ✅ Stride will prevent data loss (line > 512 tokens)

5️⃣  Disk Space Check:
   Available: 250.5GB
   ✅ Sufficient space for checkpoints

✅ Pre-Training Verification Complete
```

If you see ❌ or ⚠️ warnings, fix them BEFORE starting training!

---

### 3. Start Training

#### On A100-80GB:
```bash
# Direct run:
python train_simple.py

# Or with logging to file:
python train_simple.py 2>&1 | tee training.log
```

#### On CPU Server (Recommended: Use screen/tmux):
```bash
# Start screen session (so training continues if SSH disconnects)
screen -S insurance_training

# Start training
python train_simple.py

# Detach from screen: Press Ctrl+A, then D
# Training continues in background

# Reattach later:
screen -r insurance_training

# List all screen sessions:
screen -ls
```

**Expected startup output:**
```
🚀 GPU Detected: NVIDIA A100-SXM4-80GB (80GB)
   Batch: 8, Grad Accum: 4, Effective Batch: 32

Loading model and tokenizer...
✅ Pad token: <|endoftext|> (id: 32000)
✅ Model loaded: 3.82B parameters

Loading datasets...
  Domain examples: 1234
  General examples: 456
  Combined total: 1690

Tokenizing dataset with stride=50...
✅ Tokenized examples: 1859
   Expected with stride: ~1859 (10% overlap)
   Stride working: ✅ YES

📋 Training Configuration:
   Device: cuda
   Precision: bfloat16
   Per-device batch: 8
   Gradient accumulation: 4
   Effective batch size: 32
   Learning rate: 2e-05
   Epochs: 2
   Warmup: 3.0%
   Weight decay: 0.01

📊 Dataset Split:
   Training examples: 1673
   Validation examples: 186

🚀 Starting training...
```

---

### 4. Monitor Training

#### On GPU:
```bash
# Terminal 1: Watch GPU usage
watch -n 5 nvidia-smi

# Terminal 2: Tail logs
tail -f logs/events.out.tfevents.*

# Check progress in training output:
# You'll see:
{'loss': 2.456, 'learning_rate': 1.94e-05, 'epoch': 0.25}  # Step 100
{'eval_loss': 2.123, 'epoch': 0.5}                         # Eval checkpoint
{'loss': 1.987, 'learning_rate': 1.82e-05, 'epoch': 0.75} # Continuing...
```

#### On CPU:
```bash
# Monitor CPU/RAM usage
htop

# Or:
top

# Check training logs:
tail -f logs/events.out.tfevents.*
```

---

### 5. Training Progress Timeline

#### A100-80GB Timeline:
```
Hour 0:  Setup, tokenization (5 mins)
Hour 1:  Training starts, first eval checkpoint
Hour 5:  ~25% complete
Hour 12: ~50% complete (epoch 1 done)
Hour 18: ~75% complete
Hour 28: Training complete, saving model
```

#### CPU Server Timeline:
```
Day 1:   0-20% complete
Day 2:   20-40% complete
Day 3:   40-60% complete (epoch 1 done)
Day 4:   60-80% complete
Day 5:   80-95% complete
Day 6:   Complete, saving model
```

---

### 6. What Gets Saved

```
output/
├── checkpoint-500/          # Checkpoint at step 500
│   ├── model.safetensors
│   ├── config.json
│   └── ...
├── checkpoint-1000/         # Checkpoint at step 1000
├── checkpoint-1500/         # Checkpoint at step 1500
└── final/                   # Best model (loaded at end)
    ├── model.safetensors    # ~7.6 GB
    ├── config.json
    ├── tokenizer.json
    └── ...

logs/
└── events.out.tfevents.*    # TensorBoard logs
```

**Disk usage**: ~30-40 GB (3 checkpoints + final model)

---

### 7. After Training Completes

#### Test the model:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./output/final/")
tokenizer = AutoTokenizer.from_pretrained("./output/final/")

prompt = "An insurance claim for water damage must include"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0]))

# Should output insurance-specific terminology:
# "documentation of the damaged property, photographs, 
#  repair estimates, proof of policy coverage, and a detailed 
#  description of how the damage occurred..."
```

#### Compare with base model:
```python
base = AutoModelForCausalLM.from_pretrained("/Users/munishm/Documents/phi-3.5-mini-instruct/")
base_outputs = base.generate(**inputs, max_length=100)
print(tokenizer.decode(base_outputs[0]))

# Base model may give more generic response
```

---

## Troubleshooting

### Training stops with OOM (Out of Memory)
```python
# Edit train_simple.py, line 28:
batch_size = 4 if vram_gb >= 70 else 2  # Reduce from 8 to 4
```

### Training too slow on CPU
```python
# This is expected - CPU is 10-20x slower than GPU
# Options:
# 1. Let it run (4-6 days)
# 2. Move to GPU server
# 3. Reduce epochs to 1 in train_simple.py line 135
```

### Loss becomes NaN
```python
# Usually means learning rate too high
# Edit train_simple.py line 128:
learning_rate=1e-5,  # Reduce from 2e-5
```

### Disk full during training
```bash
# Remove old checkpoints manually:
rm -rf output/checkpoint-500/
rm -rf output/checkpoint-1000/
# Keep only latest and final
```

---

## Expected Results

### Training Loss Curve:
```
Start:  3.5-4.0 (random)
Step 500:   2.5-3.0
Step 1000:  2.0-2.5
Step 2000:  1.5-2.0
End (step 3000+): 1.2-1.8
```

### Evaluation Loss:
```
Should improve from ~3.0 to ~1.5-2.0
Early stopping triggers if plateaus for 3 evals
```

### Quality Indicators:
- ✅ Training loss decreases steadily
- ✅ Eval loss improves (may plateau, that's OK)
- ✅ Model outputs insurance-specific terms naturally
- ✅ Model still handles general prompts (not catastrophically forgotten)

---

## Cost Estimation

### A100-80GB (Lambda Labs):
```
$1.10/hour × 30 hours = $33
Storage: $0.10/GB/month × 30GB × 1 month = $3
Total: ~$36 for full training
```

### CPU Server (Owned):
```
Power: 500W × 150 hours = 75 kWh
Cost: 75 kWh × $0.12/kWh = $9 electricity
(Assumes $0.12/kWh average US rate)
```

---

## Quality vs Speed Trade-offs

Your current config is **optimized for quality**:
- ✅ Conservative learning rate (2e-5)
- ✅ Weight decay (0.01) for generalization
- ✅ Full 2 epochs on 200M tokens
- ✅ Stride preserves all training data
- ✅ Large effective batch (32) for stability

**Do NOT change** unless you have specific constraints.

If you need faster training:
- Reduce epochs to 1 (cuts time in half, slight quality loss)
- Remove weight decay (1-2% speedup, minor quality impact)

---

## Final Checklist Before Starting

- [ ] `verify_setup.py` shows all ✅
- [ ] `domain.txt` exists (150-200MB)
- [ ] `general.txt` exists (40-50MB)
- [ ] At least 30GB free disk space
- [ ] On CPU: Using screen/tmux
- [ ] On GPU: nvidia-smi shows A100
- [ ] You have 25-150 hours available
- [ ] Monitoring setup (watch nvidia-smi or htop)

**Once all checked, run:**
```bash
python train_simple.py
```

**And wait for quality results!** 🎯
