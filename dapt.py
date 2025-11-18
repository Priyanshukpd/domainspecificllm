
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
    """Dataset that reads from text files without any sampling or weighting."""
    
    def __init__(self, file_paths, shuffle=True, seed=42):
        """
        Args:
            file_paths: List of file paths to load
            shuffle: Whether to shuffle the combined dataset
            seed: Random seed for reproducibility
        """
        self.texts = []
        
        # Read all files and combine them as-is
        for file_path in file_paths:
            print(f"  Loading: {file_path}")
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                self.texts.extend(lines)
                print(f"    → {len(lines)} lines loaded")
        
        # Shuffle if requested
        if shuffle:
            random.seed(seed)
            random.shuffle(self.texts)
        
        # Validation
        if len(self.texts) == 0:
            raise ValueError(f"No valid text found in {file_paths}")
        
        print(f"  Total examples loaded: {len(self.texts)}")
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return {"text": self.texts[idx]}

print("Loading datasets (OFFLINE MODE)...")

# Combine datasets without any oversampling or weighting - use data as-is
combined_texts = TextFileDataset(
    ["domain.txt", "general.txt"],
    shuffle=True,
    seed=42
)
print()

# ============================================================================
# TOKENIZE WITH STRIDE (PREVENTS 30-40% DATA LOSS)
# ============================================================================

class TokenizedDataset(TorchDataset):
    """Tokenizes text on-the-fly with stride to prevent data loss."""
    
    def __init__(self, text_dataset, tokenizer, max_length=512, stride=50):
        self.text_dataset = text_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        
        # Pre-tokenize to create index (for proper length calculation)
        print(f"  Pre-tokenizing {len(text_dataset)} examples (max_length={max_length}, stride={stride})...")
        self.tokenized_chunks = []
        
        for idx in range(len(text_dataset)):
            if idx % 1000 == 0 and idx > 0:
                print(f"    Processed {idx}/{len(text_dataset)} examples...")
            
            text = text_dataset[idx]["text"]
            
            # Tokenize with stride
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                return_overflowing_tokens=True,
                padding=False
            )
            
            # Store all chunks from this text
            if "input_ids" in encoding:
                if isinstance(encoding["input_ids"][0], list):
                    # Multiple chunks
                    for i in range(len(encoding["input_ids"])):
                        self.tokenized_chunks.append({
                            "input_ids": encoding["input_ids"][i],
                            "attention_mask": encoding["attention_mask"][i]
                        })
                else:
                    # Single chunk
                    self.tokenized_chunks.append({
                        "input_ids": encoding["input_ids"],
                        "attention_mask": encoding["attention_mask"]
                    })
        
        # Statistics
        avg_chunks = len(self.tokenized_chunks) / len(text_dataset)
        print(f"  ✅ Tokenization complete:")
        print(f"     Original texts: {len(text_dataset)}")
        print(f"     Tokenized chunks: {len(self.tokenized_chunks)}")
        print(f"     Average chunks per text: {avg_chunks:.2f}x")
    
    def __len__(self):
        return len(self.tokenized_chunks)
    
    def __getitem__(self, idx):
        chunk = self.tokenized_chunks[idx]
        return {
            "input_ids": torch.tensor(chunk["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(chunk["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(chunk["input_ids"], dtype=torch.long)  # For causal LM
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

# Hybrid checkpointing
save_strategy = "steps"
save_steps = max(steps_per_epoch // 2, 600)
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
    eval_steps=200,  # Evaluate every 200 steps (slower with 1024 tokens)
    save_strategy=save_strategy,
    save_steps=save_steps,
    save_total_limit=save_total_limit,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    
    # Optimization
    bf16=use_bf16,
    fp16=False,
    dataloader_num_workers=workers,
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
    """Prints training metrics every N steps in a clean format."""
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called whenever logging happens (controlled by logging_steps)."""
        if logs and state.global_step > 0:
            # Extract relevant metrics
            loss = logs.get('loss')
            grad_norm = logs.get('grad_norm')
            learning_rate = logs.get('learning_rate')
            epoch = logs.get('epoch')
            
            # Only print if we have the core metrics
            if loss is not None:
                output = f"Step {state.global_step:>5d} | "
                output += f"Loss: {loss:.4f} | "
                
                if grad_norm is not None:
                    output += f"Grad Norm: {grad_norm:.4f} | "
                
                if learning_rate is not None:
                    output += f"LR: {learning_rate:.2e} | "
                
                if epoch is not None:
                    output += f"Epoch: {epoch:.2f}"
                
                print(output)

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

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

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
        EarlyStoppingCallback(early_stopping_patience=3),
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
    
    # Save final model
    print("\n" + "="*70)
    print("✅ Training complete!")
    print("="*70)
    print("\nSaving final model...")
    trainer.save_model("./output/final")
    tokenizer.save_pretrained("./output/final")
    
    print(f"✅ Model saved to: ./output/final/")
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
 