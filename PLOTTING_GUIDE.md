# Training Plots Guide 📊

## What Was Added

Your `train.py` now includes automatic plotting functionality to monitor DAPT (Domain-Adaptive Pre-Training) progress.

## Changes Made

### 1. Added Imports
```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
```

### 2. Added `PlottingCallback` Class
Automatically generates 4-panel plots showing:
- **Train vs Val Loss** (detects overfitting)
- **Validation Perplexity** (interpretable metric)
- **Learning Rate Schedule** (verifies optimizer config)
- **Gradient Norm** (training stability)

### 3. Integrated with Trainer
```python
callbacks=[
    EarlyStoppingCallback(early_stopping_patience=3),
    EnhancedEvalCallback(tokenizer, test_prompts),
    StepLoggingCallback(),  # Log every 10 steps
    PlottingCallback()  # ✅ Generate training plots
]
```

---

## Output Files

When you run training, plots will be saved to:

```
./output/plots/
├── training_metrics_step_100.png
├── training_metrics_step_200.png
├── training_metrics_step_300.png
├── ...
├── training_metrics_latest.png     ← Always current
└── training_metrics.json           ← Raw data
```

---

## What Each Plot Shows

### 📉 Plot 1: Train vs Val Loss (TOP LEFT)
**Most Important for DAPT!**

✅ **Good Pattern:**
```
Train: 2.5 → 2.0 → 1.8 → 1.7
Val:   2.6 → 2.1 → 1.9 → 1.8
Gap:   ~0.1 (stable, highlighted in yellow)
```

❌ **Overfitting Pattern:**
```
Train: 2.5 → 1.5 → 0.8
Val:   2.6 → 2.1 → 2.3 (increasing!)
Gap:   Growing (highlighted in orange/red)
```

**Action if overfitting:**
- Stop training early
- Add more general data
- Reduce epochs or learning rate

---

### 📊 Plot 2: Validation Perplexity (TOP RIGHT)
**Human-readable metric**

- Lower is better
- Red dashed line shows target (<15)
- Perplexity = exp(loss)

**Typical DAPT progression:**
```
Start:  Perplexity ~25-30
Good:   Perplexity ~10-15
Great:  Perplexity <10
```

---

### 📈 Plot 3: Learning Rate Schedule (BOTTOM LEFT)
**Verifies optimizer configuration**

Should show:
1. **Warmup phase:** LR increases from 0 to max (first 3% of steps)
2. **Cosine decay:** Smooth decrease to ~0

If it looks wrong, check `warmup_ratio` and `lr_scheduler_type` in TrainingArguments.

---

### ⚡ Plot 4: Gradient Norm (BOTTOM RIGHT)
**Training stability indicator**

✅ **Stable training:**
```
Grad Norm: 0.5 → 0.8 → 0.6 → 0.7 (oscillates around 0.5-1.0)
```

❌ **Unstable/exploding:**
```
Grad Norm: 0.5 → 2.5 → 10.3 → 50.2 (exponential growth!)
```

Red dashed line shows clip threshold (1.0). If norms frequently hit this, training may be unstable.

---

## Monitoring During Training

### View Latest Plot in Real-Time

```bash
# In a separate terminal, watch for updates
watch -n 10 ls -lh ./output/plots/

# Or view the latest plot (macOS)
open ./output/plots/training_metrics_latest.png

# Or view the latest plot (Linux with X11)
eog ./output/plots/training_metrics_latest.png
```

### Analyze Saved Metrics

```python
import json
import matplotlib.pyplot as plt

# Load saved metrics
with open('./output/plots/training_metrics.json', 'r') as f:
    data = json.load(f)

# Create custom analysis
plt.figure(figsize=(10, 6))
plt.plot(data['train_steps'], data['train_losses'], label='Train')
plt.plot(data['val_steps'], data['val_losses'], label='Val')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend()
plt.title('Custom Loss Analysis')
plt.savefig('custom_analysis.png')
```

---

## DAPT-Specific Warnings

### 🚨 Red Flags to Watch For

1. **Widening Gap**
   - Train-Val gap >0.5 and growing
   - **Action:** Stop training, use earlier checkpoint

2. **Val Loss Plateau or Increase**
   - Val loss stops decreasing after step 300
   - **Action:** Early stopping will trigger automatically

3. **Exploding Gradients**
   - Grad norm consistently >1.0 or spiking
   - **Action:** Reduce learning rate, enable gradient checkpointing

4. **Perplexity Plateau**
   - Perplexity stuck at 20+ for many steps
   - **Action:** Check if data is properly shuffled, try higher learning rate

---

## Troubleshooting

### Issue: Plots not generating
**Check:**
```bash
ls -la ./output/plots/
```
If empty, check terminal for errors related to matplotlib.

**Fix:**
```bash
# Install matplotlib if missing
pip install matplotlib
```

### Issue: "Waiting for data..." in all plots
**Reason:** Not enough steps completed yet.

**Wait for:** At least 100 steps (first evaluation).

### Issue: Plot looks garbled
**Reason:** File opened while being written.

**Fix:** Wait 2-3 seconds after plot save message, then open.

---

## Terminal Output Example

During training, you'll see:

```
Step    10 | Loss: 3.7821 | Grad Norm: 1.9876 | LR: 6.40e-06 | Epoch: 0.02
Step    20 | Loss: 3.7234 | Grad Norm: 1.8542 | LR: 9.60e-06 | Epoch: 0.04
...
Step   100 | Loss: 3.4012 | Grad Norm: 1.2987 | LR: 1.95e-05 | Epoch: 0.20

======================================================================
📊 Evaluation at Step 100:
======================================================================
   Eval Loss: 3.1234
   Perplexity: 22.76
======================================================================
📊 Plots saved: ./output/plots/training_metrics_step_100.png
```

---

## Best Practices

1. **Check plots every 2-3 evaluations** (~300 steps)
2. **Look for the gap** between train and val loss
3. **Monitor perplexity trend** - should steadily decrease
4. **Watch gradient norm** - should be stable <1.0
5. **Save good checkpoints** - if gap starts widening, revert to earlier checkpoint

---

## Next Steps

After training completes:
1. Check `./output/plots/training_metrics_latest.png`
2. Review `./output/plots/training_metrics.json` for detailed data
3. Compare train vs val loss to confirm no overfitting
4. Verify perplexity reached target (<15)

---

## Questions?

- **Gap too large?** Increase general data percentage (currently 20%)
- **Loss not decreasing?** Increase learning rate or check data quality
- **Training unstable?** Enable gradient checkpointing, reduce batch size
- **Perplexity plateau?** Check if enough training data, try more epochs

Happy training! 🚀
