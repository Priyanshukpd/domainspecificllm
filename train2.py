
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
import os
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
from pathlib import Path
 
os.environ["TOKENIZERS_PARALLELISM"] = "false" 

# ============================================================================
# OFFLINE MODE - Disable all HuggingFace Hub connections
# ============================================================================
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

# ============================================================================
# AUTO-DETECT HARDWARE
# ============================================================================

if torch.cuda.is_available():
    device = "cuda"
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_bf16 = True
    
    # Optimized for 1024 token sequences
    if vram_gb >= 70:  # A100 80GB, H100
        batch_size = 2
        grad_accum = 8
        gradient_checkpointing = False  # Not needed with 80GB
        workers = 4
    elif vram_gb >= 40:  # A6000, A40
        batch_size = 1
        grad_accum = 16
        gradient_checkpointing = True
        workers = 4
    elif vram_gb >= 20:  # RTX 4090, 3090
        batch_size = 1
        grad_accum = 16
        gradient_checkpointing = True
        workers = 2
    else:
        print("⚠️  GPU VRAM < 20GB - may struggle with 1024 tokens!")
        print("   Consider using max_length=512 instead")
        batch_size = 1
        grad_accum = 32
        gradient_checkpointing = True
        workers = 2
    
    print(f"🚀 GPU Detected: {device_name} ({vram_gb:.0f}GB)")
    print(f"   Batch: {batch_size}, Grad Accum: {grad_accum}, Effective Batch: {batch_size * grad_accum}")
    print(f"   Gradient Checkpointing: {'Enabled' if gradient_checkpointing else 'Disabled'}")
    print(f"   Workers: {workers}")
else:
    device = "cpu"
    use_bf16 = False
    batch_size = 1
    grad_accum = 32
    gradient_checkpointing = True
    workers = 8
    print(f"🖥️  CPU Mode Detected (16+ cores recommended)")
    print(f"   Batch: {batch_size}, Grad Accum: {grad_accum}, Effective Batch: {batch_size * grad_accum}")
    print(f"⚠️  Training with 1024 tokens will take 600+ hours. Consider GPU for faster training.")

print()

# ============================================================================
# LOAD MODEL & TOKENIZER (OFFLINE)
# ============================================================================

MODEL_PATH = "/data02/WorkingSLIM/PHI_SLIM/phi-3-pytorch-phi-3.5-mini-instruct-v2"

print("Loading model and tokenizer (OFFLINE MODE)...")

# Try flash attention, fallback to eager if not available
try:
    import flash_attn
    attn_impl = "flash_attention_2" if device == "cuda" else "eager"
except ImportError:
    attn_impl = "eager"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    device_map="auto" if device == "cuda" else None,
    trust_remote_code=True,
    attn_implementation=attn_impl,
    local_files_only=True  # ✅ OFFLINE: Only use local files
)

model.config.use_cache = False

if device == "cpu":
    model = model.to("cpu")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True  # ✅ OFFLINE: Only use local files
)

# Verify pad token
if tokenizer.pad_token is None:
    print("⚠️  No pad token found - setting to EOS token")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id
else:
    print(f"✅ Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")

print(f"✅ Model loaded: {model.num_parameters() / 1e9:.2f}B parameters\n")

# ============================================================================
# LOAD DATA (OFFLINE - Plain text files)
# ============================================================================

from torch.utils.data import Dataset as TorchDataset
import random

class TextFileDataset(TorchDataset):
    """Dataset that reads from text files with optional weighted sampling."""
    
    def __init__(self, file_paths, weights=None, shuffle=True, seed=42):
        """
        Args:
            file_paths: List of file paths to load
            weights: List of sampling weights (e.g., [0.8, 0.2] for 80/20 split)
                    Based on TOKEN COUNT, not line count
            shuffle: Whether to shuffle the combined dataset
            seed: Random seed for reproducibility
        """
        self.texts = []
        
        # Default to equal weights if not specified
        if weights is None:
            weights = [1.0] * len(file_paths)
        
        if len(weights) != len(file_paths):
            raise ValueError(f"Number of weights ({len(weights)}) must match number of files ({len(file_paths)})")
        
        # Read all files first and calculate token counts
        all_lines = []
        token_counts = []
        
        for file_path in file_paths:
            print(f"  Loading: {file_path}")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                all_lines.append(lines)
                
                # Estimate tokens (4 chars ≈ 1 token)
                total_chars = sum(len(line) for line in lines)
                approx_tokens = total_chars / 4
                token_counts.append(approx_tokens)
                
                print(f"    → {len(lines)} lines (~{approx_tokens:.0f} tokens)")
        
        # Track tokens for final summary
        domain_token_count = 0
        general_token_count = 0
        
        # NO OVERSAMPLING - Only undersample if needed
        if weights and len(set(weights)) > 1 and len(file_paths) >= 2:
            # Calculate target tokens based on domain (first file)
            domain_tokens = token_counts[0]
            target_general_tokens = domain_tokens * (weights[1] / weights[0])  # 20% of domain
            
            print(f"\n  📊 Dataset Balance Check:")
            print(f"     Domain tokens: {domain_tokens:.0f} (target: {weights[0]*100:.0f}%)")
            print(f"     General tokens available: {token_counts[1]:.0f}")
            print(f"     General tokens needed (for {weights[1]*100:.0f}%): {target_general_tokens:.0f}")
            
            random.seed(seed)
            
            # Add domain data (always use all of it)
            self.texts.extend(all_lines[0])
            domain_token_count = domain_tokens
            print(f"\n  ✅ Using all {len(all_lines[0])} domain examples (~{domain_token_count:.0f} tokens)")
            
            # Handle general data
            general_lines = all_lines[1]
            current_general_tokens = token_counts[1]
            
            if current_general_tokens < target_general_tokens:
                # Not enough general data - use what we have and warn
                shortage = target_general_tokens - current_general_tokens
                shortage_pct = (shortage / target_general_tokens) * 100
                
                print(f"\n  ⚠️  WARNING: Insufficient general data!")
                print(f"     Missing: ~{shortage:.0f} tokens ({shortage_pct:.1f}% short)")
                print(f"     📝 RECOMMENDATION: Add more text to general.txt")
                print(f"        Target: {target_general_tokens:.0f} tokens")
                print(f"        Current: {current_general_tokens:.0f} tokens")
                print(f"\n  ℹ️  Continuing training with available data...")
                
                # Calculate actual token ratios (FIXED)
                total_actual_tokens = domain_tokens + current_general_tokens
                actual_domain_pct = 100 * domain_tokens / total_actual_tokens
                actual_general_pct = 100 * current_general_tokens / total_actual_tokens
                print(f"     Actual ratio will be ~{actual_domain_pct:.1f}% domain / ~{actual_general_pct:.1f}% general (by tokens)")
                
                self.texts.extend(general_lines)
                general_token_count = current_general_tokens
                print(f"  ✅ Using all {len(general_lines)} general examples (~{general_token_count:.0f} tokens)")
                
            else:
                # Enough or excess general data - sample exactly what we need
                # Calculate how many lines to sample based on token ratio
                target_sample_ratio = target_general_tokens / current_general_tokens
                target_line_count = int(len(general_lines) * target_sample_ratio)
                
                sampled = random.sample(general_lines, min(target_line_count, len(general_lines)))
                general_token_count = sum(len(line) for line in sampled) / 4
                
                excess = current_general_tokens - target_general_tokens
                excess_pct = (excess / current_general_tokens) * 100
                
                print(f"\n  ✅ General data sufficient!")
                print(f"     Excess: ~{excess:.0f} tokens ({excess_pct:.1f}%)")
                print(f"  ✅ Sampled {len(sampled)}/{len(general_lines)} general examples (~{general_token_count:.0f} tokens)")
                
                self.texts.extend(sampled)
        else:
            # No weighting - just combine all lines
            for lines, tokens in zip(all_lines, token_counts):
                self.texts.extend(lines)
                domain_token_count += tokens
        
        # Shuffle if requested
        if shuffle:
            random.seed(seed)
            random.shuffle(self.texts)
        
        # Validation
        if len(self.texts) == 0:
            raise ValueError(f"No valid text found in {file_paths}")
        
        # Final summary with CORRECT token counts (calculated before shuffle)
        total_tokens = domain_token_count + general_token_count
        print(f"\n  📦 Total examples for training: {len(self.texts)} (~{total_tokens:.0f} tokens)")
        
        if len(all_lines) >= 2 and weights and len(set(weights)) > 1:
            if total_tokens > 0:
                actual_domain_pct = 100 * domain_token_count / total_tokens
                actual_general_pct = 100 * general_token_count / total_tokens
                print(f"     Actual domain ratio: {actual_domain_pct:.1f}% (~{domain_token_count:.0f} tokens)")
                print(f"     Actual general ratio: {actual_general_pct:.1f}% (~{general_token_count:.0f} tokens)")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

print("Loading datasets (OFFLINE MODE)...")

# Combine datasets with 80% domain, 20% general weighting
combined_texts = TextFileDataset(
    ["domain.txt", "general.txt"],
    weights=[0.8, 0.2],  # 80% domain, 20% general
    shuffle=True,
    seed=42
)
print()

# ============================================================================
# TOKENIZE WITH STRIDE (PREVENTS 30-40% DATA LOSS)
# ============================================================================

class TokenizedDataset(TorchDataset):
    """SIMPLIFIED: Pre-tokenize everything, store as plain lists."""
    
    def __init__(self, text_dataset, tokenizer, max_length=512, stride=50):
        self.tokenizer = tokenizer
        self.chunks = []
        
        print(f"  Tokenizing {len(text_dataset)} examples...")
        
        for idx in range(len(text_dataset)):
            if idx % 1000 == 0 and idx > 0:
                print(f"    {idx}/{len(text_dataset)}...")
            
            text = text_dataset[idx]["text"]
            
            # Simple tokenization - no fancy stuff
            tokens = tokenizer.encode(text, add_special_tokens=True)
            tokens = list(tokens)  # Force to plain Python list
            
            # Split into chunks with stride
            for i in range(0, len(tokens), max_length - stride):
                chunk = list(tokens[i:i + max_length])  # Force to plain list
                
                if len(chunk) < 10:  # Skip tiny chunks
                    continue
                
                self.chunks.append({
                    "input_ids": chunk,
                    "attention_mask": [1] * len(chunk)
                })
        
        print(f"  ✅ Created {len(self.chunks)} chunks from {len(text_dataset)} texts")
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        # Return COPIES to avoid any reference issues
        return {
            "input_ids": list(chunk["input_ids"]),
            "attention_mask": list(chunk["attention_mask"]),
            "labels": list(chunk["input_ids"])
        }

print("Tokenizing dataset...")
tokenized_dataset = TokenizedDataset(
    combined_texts,
    tokenizer,
    max_length=1024,  # Full context for production-quality DAPT
    stride=256        # 25% overlap for context preservation
)
print()

# ============================================================================
# TRAINING ARGUMENTS
# ============================================================================

# Calculate steps per epoch
examples_per_epoch = len(tokenized_dataset) * 0.9
steps_per_epoch = int(examples_per_epoch / (batch_size * grad_accum))

# More frequent evaluation and checkpointing for better monitoring
eval_steps = 50  # Changed from 200 - evaluate every 50 steps
save_strategy = "steps"
save_steps = 100  # Changed from 600 - save every 100 steps

# Ensure save_steps is a multiple of eval_steps
save_steps = ((save_steps + eval_steps - 1) // eval_steps) * eval_steps
save_total_limit = 3

training_args = TrainingArguments(
    output_dir="./output",
    
    # Batch configuration
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    
    # Learning rate schedule
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Training duration
    num_train_epochs=2,
    max_steps=-1,
    
    # Evaluation & checkpointing
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    
    # Optimization
    bf16=use_bf16,
    fp16=False,
    dataloader_num_workers=workers,  # MUST be 0 with custom collator to avoid multiprocessing issues
    gradient_checkpointing=gradient_checkpointing,  # Dynamic based on VRAM
    
    # Monitoring
    logging_steps=10,  # Log every 10 steps
    logging_first_step=True,
    logging_dir="./logs",
    report_to="tensorboard",
    disable_tqdm=False,
    
    # Stability
    max_grad_norm=1.0,
    weight_decay=0.01,
)

print("="*70)
print("📋 Training Configuration (OFFLINE MODE):")
print(f"   Device: {device}")
print(f"   Precision: {'bfloat16' if use_bf16 else 'float32'}")
print(f"   Per-device batch: {batch_size}")
print(f"   Gradient accumulation: {grad_accum}")
print(f"   Effective batch size: {batch_size * grad_accum}")
print(f"   Learning rate: {training_args.learning_rate}")
print(f"   Epochs: {training_args.num_train_epochs}")
print(f"   Steps per epoch: ~{steps_per_epoch}")
print(f"   Total steps: ~{steps_per_epoch * 2}")
print(f"   Checkpoint every: {save_steps} steps")
print(f"   Eval every: {training_args.eval_steps} steps")
print("="*70)
print()

# ============================================================================
# ENHANCED EVALUATION CALLBACK
# ============================================================================

from transformers import TrainerCallback
import numpy as np

class StepLoggingCallback(TrainerCallback):
    """Prints training metrics every N steps in a clean format with timestamps."""
    
    def __init__(self):
        from datetime import datetime
        self.start_time = None
        self.datetime = datetime
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Record training start time."""
        from datetime import datetime
        self.start_time = datetime.now()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called whenever logging happens (controlled by logging_steps)."""
        if logs and state.global_step > 0:
            from datetime import datetime
            
            # Extract relevant metrics
            loss = logs.get('loss')
            grad_norm = logs.get('grad_norm')
            learning_rate = logs.get('learning_rate')
            epoch = logs.get('epoch')
            
            # Only print if we have the core metrics
            if loss is not None:
                # Calculate elapsed time
                current_time = datetime.now()
                if self.start_time:
                    elapsed = current_time - self.start_time
                    hours, remainder = divmod(int(elapsed.total_seconds()), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                else:
                    time_str = "00:00:00"
                
                # Format current timestamp
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                
                output = f"[{timestamp}] [{time_str}] Step {state.global_step:>5d} | "
                output += f"Loss: {loss:.4f} | "
                
                if grad_norm is not None:
                    output += f"Grad Norm: {grad_norm:.4f} | "
                
                if learning_rate is not None:
                    output += f"LR: {learning_rate:.2e} | "
                
                if epoch is not None:
                    output += f"Epoch: {epoch:.2f}"
                
                print(output)

class SafeEarlyStoppingCallback(TrainerCallback):
    """
    Early stopping that only activates after minimum training steps.
    Prevents stopping before the model has seen the full dataset at least once.
    """
    
    def __init__(self, min_steps=None, patience=3):
        """
        Args:
            min_steps: Minimum steps before early stopping can trigger
                      If None, will be calculated as steps_per_epoch (1 full epoch)
            patience: Number of evaluations without improvement before stopping
        """
        self.min_steps = min_steps
        self.patience = patience
        self.patience_counter = 0
        self.best_metric = None
        self.best_step = None
        self.early_stopping_triggered = False
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Calculate minimum steps if not provided and display protection info."""
        if self.min_steps is None:
            # Calculate steps per epoch from training dataset
            train_dataloader = kwargs.get('train_dataloader')
            if train_dataloader:
                steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
                self.min_steps = steps_per_epoch
            else:
                # Fallback: use eval_steps as minimum
                self.min_steps = args.eval_steps
        
        print(f"🛡️  Early Stopping Protection:")
        print(f"   Minimum training steps: {self.min_steps} (≥1 full epoch)")
        print(f"   Patience after minimum: {self.patience} evaluations")
        print(f"   Early stopping will be active after step {self.min_steps}")
        print()
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Check if early stopping should trigger."""
        if not metrics or 'eval_loss' not in metrics:
            return
        
        current_step = state.global_step
        current_metric = metrics['eval_loss']
        
        # SAFETY: Don't allow early stopping before minimum steps
        if current_step < self.min_steps:
            if self.best_metric is None or current_metric < self.best_metric:
                self.best_metric = current_metric
                self.best_step = current_step
            print(f"   🛡️  Early stopping protected (step {current_step}/{self.min_steps})")
            return
        
        # Standard early stopping logic (after minimum steps)
        if self.best_metric is None or current_metric < self.best_metric:
            self.best_metric = current_metric
            self.best_step = current_step
            self.patience_counter = 0
            print(f"   ✅ New best! Validation loss: {current_metric:.4f}")
        else:
            self.patience_counter += 1
            print(f"   ⚠️  No improvement for {self.patience_counter} eval(s). Patience: {self.patience_counter}/{self.patience}")
            
            if self.patience_counter >= self.patience:
                print(f"\n{'='*70}")
                print(f"🛑 EARLY STOPPING TRIGGERED at step {current_step}")
                print(f"   Best validation loss: {self.best_metric:.4f} (step {self.best_step})")
                print(f"   Current validation loss: {current_metric:.4f}")
                print(f"   No improvement for {self.patience} consecutive evaluations")
                print(f"   Loading best model from step {self.best_step}...")
                print(f"{'='*70}\n")
                control.should_training_stop = True
                self.early_stopping_triggered = True

class PlottingCallback(TrainerCallback):
    """Plots training metrics in real-time and saves figures."""
    
    def __init__(self, output_dir="./output/plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Store metrics
        self.train_losses = []
        self.train_steps = []
        self.val_losses = []
        self.val_steps = []
        self.learning_rates = []
        self.grad_norms = []
        self.perplexities = []
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Collect training metrics."""
        if logs:
            step = state.global_step
            
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
                self.train_steps.append(step)
            
            if 'learning_rate' in logs:
                self.learning_rates.append(logs['learning_rate'])
            
            if 'grad_norm' in logs:
                self.grad_norms.append(logs['grad_norm'])
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Collect validation metrics and generate plots."""
        if metrics:
            step = state.global_step
            
            if 'eval_loss' in metrics:
                self.val_losses.append(metrics['eval_loss'])
                self.val_steps.append(step)
            
            if 'eval_perplexity' in metrics:
                self.perplexities.append(metrics['eval_perplexity'])
            
            # Generate plots after each evaluation
            self._generate_plots(step)
    
    def _generate_plots(self, current_step):
        """Generate comprehensive training plots."""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics (Step {current_step})', fontsize=16, fontweight='bold')
        
        # ============= PLOT 1: Train vs Val Loss (MOST IMPORTANT) =============
        ax1 = axes[0, 0]
        if self.train_losses and self.val_losses:
            ax1.plot(self.train_steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
            ax1.plot(self.val_steps, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
            
            # Highlight the gap
            if len(self.val_steps) > 1:
                gap = self.train_losses[-1] - self.val_losses[-1]
                # Calculate overlapping region for fill_between
                min_len = min(len(self.train_steps), len(self.val_losses))
                if min_len > 0:
                    ax1.fill_between(
                        self.train_steps[-min_len:],
                        self.train_losses[-min_len:],
                        self.val_losses[-min_len:],
                        alpha=0.2,
                        color='yellow' if abs(gap) < 0.3 else 'orange' if abs(gap) < 0.5 else 'red',
                        label=f'Gap: {gap:.3f}'
                    )
            
            ax1.set_xlabel('Steps', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('📉 Training vs Validation Loss', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax1.transAxes)
        
        # ============= PLOT 2: Perplexity =============
        ax2 = axes[0, 1]
        if self.perplexities:
            ax2.plot(self.val_steps, self.perplexities, 'g-', label='Val Perplexity', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Steps', fontsize=11)
            ax2.set_ylabel('Perplexity', fontsize=11)
            ax2.set_title('📊 Validation Perplexity', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Add horizontal line for "good" perplexity threshold
            good_threshold = 15.0
            ax2.axhline(y=good_threshold, color='r', linestyle='--', alpha=0.5, label=f'Target: <{good_threshold}')
            ax2.legend(loc='upper right')
        else:
            ax2.text(0.5, 0.5, 'Waiting for evaluation...', ha='center', va='center', transform=ax2.transAxes)
        
        # ============= PLOT 3: Learning Rate Schedule =============
        ax3 = axes[1, 0]
        if self.learning_rates:
            ax3.plot(self.train_steps, self.learning_rates, 'm-', label='Learning Rate', linewidth=2)
            ax3.set_xlabel('Steps', fontsize=11)
            ax3.set_ylabel('Learning Rate', fontsize=11)
            ax3.set_title('📈 Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        else:
            ax3.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax3.transAxes)
        
        # ============= PLOT 4: Gradient Norm (Stability) =============
        ax4 = axes[1, 1]
        if self.grad_norms:
            ax4.plot(self.train_steps, self.grad_norms, 'c-', label='Gradient Norm', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Steps', fontsize=11)
            ax4.set_ylabel('Gradient Norm', fontsize=11)
            ax4.set_title('⚡ Gradient Norm (Stability)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # Add warning line for gradient explosion
            max_norm = 1.0  # From training_args.max_grad_norm
            ax4.axhline(y=max_norm, color='r', linestyle='--', alpha=0.5, label=f'Clip Threshold: {max_norm}')
            ax4.legend(loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'Waiting for data...', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.output_dir / f"training_metrics_step_{current_step}.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Also save as "latest" for easy viewing
        latest_path = self.output_dir / "training_metrics_latest.png"
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics (Step {current_step})', fontsize=16, fontweight='bold')
        
        # Re-create plots for latest
        ax1 = axes[0, 0]
        if self.train_losses and self.val_losses:
            ax1.plot(self.train_steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, alpha=0.7)
            ax1.plot(self.val_steps, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='o', markersize=4)
            if len(self.val_steps) > 1:
                gap = self.train_losses[-1] - self.val_losses[-1]
                min_len = min(len(self.train_steps), len(self.val_losses))
                if min_len > 0:
                    ax1.fill_between(
                        self.train_steps[-min_len:],
                        self.train_losses[-min_len:],
                        self.val_losses[-min_len:],
                        alpha=0.2,
                        color='yellow' if abs(gap) < 0.3 else 'orange' if abs(gap) < 0.5 else 'red',
                        label=f'Gap: {gap:.3f}'
                    )
            ax1.set_xlabel('Steps', fontsize=11)
            ax1.set_ylabel('Loss', fontsize=11)
            ax1.set_title('📉 Training vs Validation Loss', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        if self.perplexities:
            ax2.plot(self.val_steps, self.perplexities, 'g-', label='Val Perplexity', linewidth=2, marker='s', markersize=4)
            ax2.set_xlabel('Steps', fontsize=11)
            ax2.set_ylabel('Perplexity', fontsize=11)
            ax2.set_title('📊 Validation Perplexity', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            good_threshold = 15.0
            ax2.axhline(y=good_threshold, color='r', linestyle='--', alpha=0.5, label=f'Target: <{good_threshold}')
            ax2.legend(loc='upper right')
        
        ax3 = axes[1, 0]
        if self.learning_rates:
            ax3.plot(self.train_steps, self.learning_rates, 'm-', label='Learning Rate', linewidth=2)
            ax3.set_xlabel('Steps', fontsize=11)
            ax3.set_ylabel('Learning Rate', fontsize=11)
            ax3.set_title('📈 Learning Rate Schedule', fontsize=12, fontweight='bold')
            ax3.legend(loc='upper right')
            ax3.grid(True, alpha=0.3)
            ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
        
        ax4 = axes[1, 1]
        if self.grad_norms:
            ax4.plot(self.train_steps, self.grad_norms, 'c-', label='Gradient Norm', linewidth=2, alpha=0.7)
            ax4.set_xlabel('Steps', fontsize=11)
            ax4.set_ylabel('Gradient Norm', fontsize=11)
            ax4.set_title('⚡ Gradient Norm (Stability)', fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            max_norm = 1.0
            ax4.axhline(y=max_norm, color='r', linestyle='--', alpha=0.5, label=f'Clip Threshold: {max_norm}')
            ax4.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig(latest_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Plots saved: {plot_path}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate final comprehensive plot and save metrics."""
        self._generate_plots(state.global_step)
        
        # Also save metrics as JSON
        metrics_file = self.output_dir / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump({
                'train_steps': self.train_steps,
                'train_losses': self.train_losses,
                'val_steps': self.val_steps,
                'val_losses': self.val_losses,
                'learning_rates': self.learning_rates,
                'grad_norms': self.grad_norms,
                'perplexities': self.perplexities
            }, f, indent=2)
        
        print(f"✅ Final metrics saved: {metrics_file}")

class EnhancedEvalCallback(TrainerCallback):
    """Combined perplexity tracking + sample generation callback."""
    
    def __init__(self, tokenizer, prompts):
        self.tokenizer = tokenizer
        self.prompts = prompts
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            perplexity = np.exp(metrics['eval_loss'])
            metrics['eval_perplexity'] = perplexity
            
            print(f"\n{'='*70}")
            print(f"📊 Evaluation at Step {state.global_step}:")
            print(f"{'='*70}")
            print(f"   Eval Loss: {metrics['eval_loss']:.4f}")
            print(f"   Perplexity: {perplexity:.2f}")
            
            # Generate samples every 1000 steps
            if state.global_step % 1000 == 0 and model is not None:
                print(f"\n📝 Sample Generations:")
                print(f"{'-'*70}")
                model.eval()
                
                for i, prompt in enumerate(self.prompts[:3], 1):
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=150,  # Allow longer responses with 1024 context
                            do_sample=True,
                            temperature=0.7,
                            top_p=0.9,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    continuation = generated[len(prompt):].strip()
                    
                    print(f"\n{i}. Prompt: \"{prompt}\"")
                    print(f"   Model: \"{continuation[:200]}...\"" if len(continuation) > 200 else f"   Model: \"{continuation}\"")
                
                print(f"\n{'-'*70}")
                model.train()
            
            print(f"{'='*70}\n")

# Test prompts (customize these to match YOUR specific domain data)
test_prompts = [
    "What is the difference between whole life and term life insurance?",
    "Explain how underwriting risk assessment works in commercial property insurance:",
    "Define loss ratio and its significance for insurance companies:"
]

# ============================================================================
# DATA COLLATOR (with dynamic padding)
# ============================================================================

# Create a simple wrapper for better compatibility across transformers versions
class SimpleDataCollator:
    """Simple data collator that works across all transformers versions."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Find max length in batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of is not None:
            max_length = ((max_length + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # Pad each feature
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }
        
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        
        for f in features:
            input_ids = list(f["input_ids"])
            attention_mask = list(f["attention_mask"])
            labels = list(f["labels"])
            
            # Calculate padding needed
            padding_length = max_length - len(input_ids)
            
            # Pad input_ids and attention_mask
            input_ids = input_ids + [pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            labels = labels + [-100] * padding_length  # -100 is ignored in loss
            
            batch["input_ids"].append(input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(labels)
        
        # Convert to tensors
        import torch
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }

data_collator = SimpleDataCollator(tokenizer=tokenizer)

# ============================================================================
# SPLIT DATASET & TRAIN
# ============================================================================

# Manual train/test split (90/10)
dataset_size = len(tokenized_dataset)
train_size = int(0.9 * dataset_size)
test_size = dataset_size - train_size

# Create indices for split
indices = list(range(dataset_size))
random.seed(42)
random.shuffle(indices)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

# Create subset datasets
class SubsetDataset(TorchDataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

train_dataset = SubsetDataset(tokenized_dataset, train_indices)
eval_dataset = SubsetDataset(tokenized_dataset, test_indices)

print(f"📊 Dataset Split:")
print(f"   Training examples: {len(train_dataset)}")
print(f"   Validation examples: {len(eval_dataset)}")
print()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    callbacks=[
        SafeEarlyStoppingCallback(min_steps=steps_per_epoch, patience=3),  # Safe early stopping
        EnhancedEvalCallback(tokenizer, test_prompts),
        StepLoggingCallback(),  # Log every 10 steps
        PlottingCallback()  # Generate training plots
    ]
)

print("="*70)
print("🚀 Starting training (OFFLINE MODE)...")
print("="*70)
print()

# Auto-resume from checkpoint
from pathlib import Path
import re

def get_latest_checkpoint(output_dir):
    """Find the latest checkpoint automatically."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    
    checkpoints = []
    for item in output_path.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            match = re.match(r'checkpoint-(\d+)', item.name)
            if match:
                step_num = int(match.group(1))
                checkpoints.append((step_num, str(item)))
    
    if not checkpoints:
        return None
    
    checkpoints.sort(reverse=True)
    return checkpoints[0][1]

latest_checkpoint = get_latest_checkpoint("./output")
if latest_checkpoint:
    print(f"🔄 Auto-resuming from: {latest_checkpoint}\n")

try:
    trainer.train(resume_from_checkpoint=latest_checkpoint)
    
    # Save final model in multiple precisions
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)
    print("\nSaving final model in multiple formats...")
    
    # 1. Save in original training precision (bfloat16 or float32)
    print("📦 Saving model in training precision...")
    trainer.save_model("./output/final")
    tokenizer.save_pretrained("./output/final")
    print(f"   ✅ Saved to: ./output/final/ ({model.dtype})")
    
    # 2. Save in FP32 (full precision)
    print("\n📦 Converting and saving model in FP32...")
    model_fp32 = model.float()  # Convert to FP32
    model_fp32.save_pretrained("./output/final_fp32")
    tokenizer.save_pretrained("./output/final_fp32")
    print(f"   ✅ Saved to: ./output/final_fp32/ (float32)")
    
    # 3. Save in FP16 (half precision)
    print("\n📦 Converting and saving model in FP16...")
    model_fp16 = model.half()  # Convert to FP16
    model_fp16.save_pretrained("./output/final_fp16")
    tokenizer.save_pretrained("./output/final_fp16")
    print(f"   ✅ Saved to: ./output/final_fp16/ (float16)")
    
    print("\n" + "="*70)
    print("💾 Model saved in 3 formats:")
    print(f"   1. ./output/final/ ({model.dtype})")
    print(f"   2. ./output/final_fp32/ (float32)")
    print(f"   3. ./output/final_fp16/ (float16)")
    print("="*70)
    
    print("\n📊 Training Summary:")
    print(f"   Total steps: {trainer.state.global_step}")
    
    # Safe formatting for best metric
    if trainer.state.best_metric is not None:
        print(f"   Best eval loss: {trainer.state.best_metric:.4f}")
    else:
        print(f"   Best eval loss: N/A (insufficient evaluation steps)")
    
    # Safe formatting for final loss
    if trainer.state.log_history:
        last_log = trainer.state.log_history[-1]
        final_loss = last_log.get('loss')
        if final_loss is not None:
            print(f"   Final training loss: {final_loss:.4f}")
        else:
            eval_loss = last_log.get('eval_loss')
            if eval_loss is not None:
                print(f"   Final eval loss: {eval_loss:.4f}")
            else:
                print(f"   Final loss: N/A")
    else:
        print(f"   Final loss: N/A (no logs available)")
    
    print("="*70)
    
except KeyboardInterrupt:
    print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
    print("   Partial checkpoints saved in: ./output/")
    print("   Run the script again to auto-resume!")
    
except Exception as e:
    print(f"\n\n❌ Error during training: {e}")
    print("   Check logs in: ./logs/")
    import traceback
    traceback.print_exc() 
 