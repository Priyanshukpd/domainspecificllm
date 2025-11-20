#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LoRA Fine-tuning Script for Phi-3.5 Mini Instruct
Supports: A100 80GB GPU and 96GB CPU Server with auto-detection
"""

import os
import json
import math
import random
import numpy as np
import torch
from typing import List, Dict, Optional, Any
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments


# ===============================
# Config
# ===============================
MODEL_PATH = "./MODEL_DIR"                 # local Phi-3.5 Mini Instruct checkpoint dir
OUT_DIR    = "./phi3.5-lora-sft"           # output directory for LoRA adapter + tokenizer
DATA_PATH  = "phi_test_instruction_data.json"  # JSON list of {"instruction", "input"(optional), "output"}

NUM_EPOCHS = 3


# ===============================
# Hardware Auto-Detection
# ===============================
def detect_hardware() -> Dict[str, Any]:
    """
    Auto-detect hardware and return optimized configs.
    A100 80GB GPU or 96GB CPU Server
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print("=" * 60)
        print(f"🚀 GPU DETECTED: {device_name} ({vram_gb:.1f} GB)")
        print("=" * 60)
        
        if vram_gb >= 70:  # A100 80GB
            config = {
                "device": "cuda",
                "dtype": torch.bfloat16,
                "batch_size": 4,              # 4x larger than CPU
                "grad_accum": 4,              # Effective batch = 16
                "gradient_checkpointing": False,  # Not needed with 80GB
                "num_workers": 4,
                "use_bf16": True,
                "use_fp16": False,
                "learning_rate": 2e-4,        # Higher LR for GPU
                "max_length": 512,            # Full sequences
                "lora_r": 16,                 # Higher rank
                "lora_alpha": 32,
            }
            print(f"✅ A100 80GB Configuration:")
            print(f"   • Batch Size: {config['batch_size']}")
            print(f"   • Gradient Accumulation: {config['grad_accum']}")
            print(f"   • Effective Batch: {config['batch_size'] * config['grad_accum']}")
            print(f"   • Max Sequence Length: {config['max_length']}")
            print(f"   • BF16: Enabled")
            print(f"   • Gradient Checkpointing: Disabled (plenty of VRAM)")
            print(f"   • LoRA Rank: {config['lora_r']}")
        else:
            # Fallback for smaller GPUs
            config = {
                "device": "cuda",
                "dtype": torch.bfloat16,
                "batch_size": 2,
                "grad_accum": 4,
                "gradient_checkpointing": True,
                "num_workers": 2,
                "use_bf16": True,
                "use_fp16": False,
                "learning_rate": 1e-4,
                "max_length": 256,
                "lora_r": 8,
                "lora_alpha": 16,
            }
            print(f"⚠️  GPU < 70GB - Using conservative settings")
    else:
        # CPU Server (96GB RAM)
        print("=" * 60)
        print(f"💻 CPU DETECTED (96GB RAM assumed)")
        print("=" * 60)
        config = {
            "device": "cpu",
            "dtype": torch.float32,          # CPU doesn't support bf16 efficiently
            "batch_size": 2,                 # Can afford larger batch on 96GB RAM
            "grad_accum": 8,                 # Effective batch = 16 (same as GPU)
            "gradient_checkpointing": True,  # Save memory
            "num_workers": 16,               # Utilize all CPU cores (96GB server likely has 32+ cores)
            "use_bf16": False,
            "use_fp16": False,
            "learning_rate": 1e-4,           # Conservative for CPU
            "max_length": 256,               # Shorter sequences for CPU speed
            "lora_r": 8,
            "lora_alpha": 16,
        }
        print(f"✅ CPU 96GB Configuration:")
        print(f"   • Batch Size: {config['batch_size']}")
        print(f"   • Gradient Accumulation: {config['grad_accum']}")
        print(f"   • Effective Batch: {config['batch_size'] * config['grad_accum']}")
        print(f"   • Max Sequence Length: {config['max_length']}")
        print(f"   • Workers: {config['num_workers']} (multi-core CPU)")
        print(f"   • Gradient Checkpointing: Enabled")
    
    print("=" * 60)
    return config


# ===============================
# Reproducibility
# ===============================
def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ===============================
# Utilities
# ===============================
def load_json(json_path: str) -> List[Dict[str, str]]:
    """Load JSON dataset from file."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_prompt(example: Dict[str, str], tokenizer: Any) -> str:
    """
    Build a simple instruction + optional input + answer format using Phi-style chat markers.
    Keep this consistent for training and inference.
    **Appends EOS token for proper sequence termination.**
    """
    instruction = example["instruction"]
    input_text = example.get("input", "").strip()
    output = example["output"]

    if input_text:
        prompt = (
            "<|user|>\n"
            f"{instruction}\n\n"
            f"Input:\n{input_text}\n"
            "<|assistant|>\n"
        )
    else:
        prompt = (
            "<|user|>\n"
            f"{instruction}\n"
            "<|assistant|>\n"
        )
    # Append EOS token for proper sequence termination
    return prompt + output + tokenizer.eos_token


class SFTDataset(Dataset):
    """
    Pre-tokenized dataset that returns input_ids, attention_mask, labels.
    Labels are masked (-100) on pad tokens to avoid training on padding.
    """
    def __init__(self, encodings: Dict[str, List[List[int]]], pad_token_id: int):
        self.encodings = encodings
        self.pad_id = pad_token_id

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        input_ids = torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long)
        attention_mask = torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long)
        labels = input_ids.clone()
        labels[labels == self.pad_id] = -100  # ignore pad tokens in loss

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ===============================
# Training Visualization Callback
# ===============================
class MetricsCallback(TrainerCallback):
    """
    Callback to track and plot training metrics in real-time.
    Includes gradient norm monitoring.
    """
    def __init__(self, output_dir: str, grad_log_frequency: int = 100):
        self.output_dir = output_dir
        self.grad_log_frequency = grad_log_frequency
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.steps = []
        self.eval_steps = []
        self.grad_steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging occurs."""
        if logs:
            if "loss" in logs:
                self.train_losses.append(logs["loss"])
                self.steps.append(state.global_step)
            if "learning_rate" in logs:
                self.learning_rates.append(logs["learning_rate"])
            if "eval_loss" in logs:
                self.eval_losses.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
    
    def on_step_end(self, args, state, control, **kwargs):
        """Log gradient norm every N steps."""
        if state.global_step % self.grad_log_frequency == 0:
            model = kwargs.get("model")
            if model is not None:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                self.grad_norms.append(total_norm)
                self.grad_steps.append(state.global_step)
                
                # Warn if gradient norm is suspiciously high or low
                if total_norm > 10.0:
                    print(f"⚠️  Warning: High gradient norm at step {state.global_step}: {total_norm:.4f}")
                elif total_norm < 0.001:
                    print(f"⚠️  Warning: Very low gradient norm at step {state.global_step}: {total_norm:.4f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Generate plots at the end of training."""
        self.plot_metrics()
    
    def plot_metrics(self) -> None:
        """Generate and save training metric plots."""
        if not self.train_losses:
            print("⚠️  No metrics to plot")
            return
        
        # Create figure with light background matching the reference
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor('#E8E8E8')
        
        # Determine current step for title
        current_step = self.steps[-1] if self.steps else 0
        fig.suptitle(f'Training Metrics (Step {current_step})', 
                     fontsize=18, fontweight='bold', y=0.98)
        
        # Create 2x2 grid
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25, 
                             left=0.08, right=0.98, top=0.93, bottom=0.08)
        
        # 1. Training vs Validation Loss (Top Left)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor('#F5F5F5')
        ax1.plot(self.steps, self.train_losses, color='#4472C4', linewidth=2.5, 
                label='Train Loss', alpha=0.9)
        if self.eval_losses:
            ax1.plot(self.eval_steps, self.eval_losses, color='#C55A11', linewidth=2.5, 
                    label='Val Loss', alpha=0.9)
            # Calculate and display gap
            if len(self.eval_losses) > 0:
                final_gap = abs(self.train_losses[-1] - self.eval_losses[-1])
                ax1.text(0.02, 0.98, f'Gap: {final_gap:.3f}', 
                        transform=ax1.transAxes, fontsize=11,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='white', alpha=0.8))
        ax1.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=11, fontweight='bold')
        ax1.set_title('⬜ Training vs Validation Loss', fontsize=12, 
                     fontweight='bold', pad=10, loc='left')
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.legend(loc='upper right', framealpha=0.9)
        
        # 2. Validation Perplexity (Top Right)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('#F5F5F5')
        if self.eval_losses:
            perplexities = [math.exp(min(loss, 20)) for loss in self.eval_losses]
            ax2.plot(self.eval_steps, perplexities, color='#70AD47', linewidth=3, alpha=0.9)
            # Add target line
            target_ppl = 15.0
            ax2.axhline(y=target_ppl, color='#C55A11', linestyle='--', 
                       linewidth=1.5, alpha=0.6)
            ax2.text(ax2.get_xlim()[1] * 0.98, target_ppl, f'Target: <{target_ppl:.1f}', 
                    fontsize=10, color='#C55A11', ha='right', va='bottom')
            ax2.fill_between(self.eval_steps, 0, perplexities, alpha=0.1, color='#70AD47')
        else:
            ax2.text(0.5, 0.5, 'No evaluation data', ha='center', va='center', 
                    fontsize=12, transform=ax2.transAxes)
        ax2.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
        ax2.set_title('⬜ Validation Perplexity', fontsize=12, 
                     fontweight='bold', pad=10, loc='left')
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # 3. Learning Rate Schedule (Bottom Left)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor('#F5F5F5')
        if self.learning_rates:
            ax3.plot(self.steps, self.learning_rates, color='#9E4AD4', linewidth=2.5)
            ax3.fill_between(self.steps, 0, self.learning_rates, alpha=0.2, color='#9E4AD4')
        ax3.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Learning Rate', fontsize=11, fontweight='bold')
        ax3.set_title('⬜ Learning Rate Schedule', fontsize=12, 
                     fontweight='bold', pad=10, loc='left')
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax3.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
        
        # 4. Gradient Norm (Stability) (Bottom Right)
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor('#F5F5F5')
        if self.grad_norms:
            # Plot gradient norm with cyan color
            ax4.plot(self.grad_steps, self.grad_norms, color='#00B0F0', 
                    linewidth=1.5, alpha=0.7)
            ax4.fill_between(self.grad_steps, 0, self.grad_norms, 
                           alpha=0.2, color='#00B0F0')
            # Add threshold line
            clip_threshold = 1.0  # max_grad_norm from training args
            ax4.axhline(y=clip_threshold, color='#C55A11', linestyle='--', 
                       linewidth=1.5, alpha=0.6, label=f'Clip Threshold: {clip_threshold}')
            ax4.legend(loc='upper right', framealpha=0.9, fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'No gradient norm data', ha='center', va='center', 
                    fontsize=12, transform=ax4.transAxes)
        ax4.set_xlabel('Steps', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Gradient Norm', fontsize=11, fontweight='bold')
        ax4.set_title('⬜ Gradient Norm (Stability)', fontsize=12, 
                     fontweight='bold', pad=10, loc='left')
        ax4.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        
        # Save plot
        plot_path = os.path.join(self.output_dir, "training_metrics.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#E8E8E8')
        print(f"\n📊 Training plots saved to: {plot_path}")
        plt.close()
        
        # Also save metrics as JSON for later analysis
        metrics_data = {
            "steps": self.steps,
            "train_losses": self.train_losses,
            "eval_steps": self.eval_steps,
            "eval_losses": self.eval_losses,
            "learning_rates": self.learning_rates,
            "grad_steps": self.grad_steps,
            "grad_norms": self.grad_norms
        }
        json_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(json_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"📊 Training metrics saved to: {json_path}")


# ===============================
# Main
# ===============================
def main() -> None:
    """Main training function."""
    # Set random seed for reproducibility
    set_seed(42)
    
    # Auto-detect hardware and get optimal config
    hw_config = detect_hardware()
    
    # Save hardware config for reproducibility
    os.makedirs(OUT_DIR, exist_ok=True)
    config_path = os.path.join(OUT_DIR, "hardware_config.json")
    with open(config_path, 'w') as f:
        # Convert torch dtypes to strings for JSON serialization
        config_export = {k: (str(v) if isinstance(v, torch.dtype) else v) 
                        for k, v in hw_config.items()}
        json.dump(config_export, f, indent=2)
    print(f"💾 Hardware config saved to: {config_path}")
    
    # Override max_length with hardware-specific value
    max_length = hw_config["max_length"]

    # ---- Tokenizer
    print(f"\n📦 Loading tokenizer from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=False,
        local_files_only=True
    )

    # Ensure pad token exists (use EOS as pad for causal LM)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # ---- Model
    print(f"\n📦 Loading model with dtype: {hw_config['dtype']}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto" if hw_config["device"] == "cuda" else {"": "cpu"},
        torch_dtype=hw_config["dtype"],
        trust_remote_code=False,
        local_files_only=True
    )

    # During training, disable KV cache to avoid any cache API edge cases
    model.config.use_cache = False
    model.config.pad_token_id = pad_id

    # ---- LoRA (target Phi-3.5 fused layers)
    print(f"\n🔧 Applying LoRA configuration (rank={hw_config['lora_r']})...")
    lora = LoraConfig(
        r=hw_config["lora_r"],
        lora_alpha=hw_config["lora_alpha"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "qkv_proj",      # fused Q/K/V
            "o_proj",        # attention output projection
            "gate_up_proj",  # fused gate + up in MLP
            "down_proj"      # MLP down projection
        ]
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # ---- Data
    print(f"\n📊 Loading dataset from: {DATA_PATH}")
    dataset = load_json(DATA_PATH)
    assert isinstance(dataset, list) and len(dataset) > 0, "Dataset must be a non-empty list of samples."
    print(f"   • Total samples: {len(dataset)}")

    texts = [format_prompt(item, tokenizer) for item in dataset]

    # Ensure at least 1 validation sample
    split_index = max(1, int(len(texts) * 0.9))
    train_texts = texts[:split_index]
    val_texts   = texts[split_index:]
    
    if len(val_texts) == 0:
        print("⚠️  WARNING: No validation data. Using last training sample.")
        val_texts = [train_texts[-1]]

    print(f"   • Train samples: {len(train_texts)}")
    print(f"   • Validation samples: {len(val_texts)}")

    # Tokenize
    print(f"\n🔤 Tokenizing (max_length={max_length})...")
    train_enc = tokenizer(train_texts, truncation=True, max_length=max_length, padding=True)
    val_enc   = tokenizer(val_texts,   truncation=True, max_length=max_length, padding=True)

    train_ds = SFTDataset(train_enc, pad_token_id=pad_id)
    val_ds   = SFTDataset(val_enc,   pad_token_id=pad_id)

    steps_per_epoch = max(1, len(train_ds) // (hw_config["batch_size"] * hw_config["grad_accum"]))
    print(f"   • Steps per epoch: {steps_per_epoch}")

    # ---- Check for existing checkpoint to resume training
    checkpoint_path = None
    checkpoints = [d for d in os.listdir(OUT_DIR) if d.startswith("checkpoint-")] if os.path.exists(OUT_DIR) else []
    if checkpoints:
        # Get the latest checkpoint
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[1]))
        checkpoint_path = os.path.join(OUT_DIR, latest_checkpoint)
        print(f"\n🔄 Found existing checkpoint: {checkpoint_path}")
        print(f"   Training will resume from this checkpoint")
    
    # ---- Training Arguments
    print(f"\n⚙️  Configuring training arguments...")
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=hw_config["batch_size"],
        gradient_accumulation_steps=hw_config["grad_accum"],
        num_train_epochs=NUM_EPOCHS,

        learning_rate=hw_config["learning_rate"],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        logging_steps=max(10, steps_per_epoch // 10),
        eval_steps=max(50, steps_per_epoch // 2),
        eval_strategy="steps",
        save_strategy="epoch",

        fp16=hw_config["use_fp16"],
        bf16=hw_config["use_bf16"],

        report_to="none",
        logging_dir=os.path.join(OUT_DIR, "logs"),

        max_grad_norm=1.0,
        weight_decay=0.01,

        group_by_length=True,
        dataloader_pin_memory=hw_config["device"] == "cuda",
        dataloader_num_workers=hw_config["num_workers"],
        
        gradient_checkpointing=hw_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if hw_config["gradient_checkpointing"] else None,
    )

    # ---- Initialize Callbacks
    metrics_callback = MetricsCallback(output_dir=OUT_DIR, grad_log_frequency=100)

    # ---- Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=None,  # Using custom dataset with pre-masked labels
        tokenizer=tokenizer,
        callbacks=[metrics_callback],  # All-in-one metrics callback with gradient norm monitoring
    )

    # Optional: Quick sanity check (disable in production)
    if os.getenv("DEBUG_MODE"):
        batch = next(iter(trainer.get_train_dataloader()))
        model.train()
        out = model(**{k: v.to(model.device) for k, v in batch.items()})
        print("Sanity check — loss.requires_grad:", out.loss.requires_grad)
        out.loss.backward()

    # ---- Train
    print("\n" + "=" * 60)
    if checkpoint_path:
        print("🚀 RESUMING TRAINING FROM CHECKPOINT")
    else:
        print("🚀 STARTING TRAINING")
    print("=" * 60)
    
    trainer.train(resume_from_checkpoint=checkpoint_path)

    # ---- Evaluate
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION")
    print("=" * 60)
    metrics = trainer.evaluate()
    print(metrics)
    if "eval_loss" in metrics:
        ppl = math.exp(metrics["eval_loss"]) if metrics["eval_loss"] < 20 else float('inf')
        print(f"   • Perplexity: {ppl:.4f}")

    # ---- Save LoRA adapter + tokenizer
    print("\n💾 Saving model...")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"✅ DONE: LoRA adapter + tokenizer saved to: {OUT_DIR}")
    
    # ---- Training Summary
    print("\n" + "=" * 60)
    print("📋 TRAINING SUMMARY")
    print("=" * 60)
    print(f"✅ Model: Phi-3.5 Mini Instruct")
    print(f"✅ LoRA Rank: {hw_config['lora_r']} (α={hw_config['lora_alpha']})")
    print(f"✅ Training Samples: {len(train_texts)}")
    print(f"✅ Validation Samples: {len(val_texts)}")
    print(f"✅ Epochs: {NUM_EPOCHS}")
    print(f"✅ Effective Batch Size: {hw_config['batch_size'] * hw_config['grad_accum']}")
    print(f"✅ Max Sequence Length: {max_length}")
    print(f"✅ Final Perplexity: {ppl:.4f}" if "eval_loss" in metrics else "✅ No eval metrics")
    print(f"\n📁 Output Directory: {OUT_DIR}")
    print(f"   ├── adapter_model.bin (LoRA weights)")
    print(f"   ├── adapter_config.json (LoRA config)")
    print(f"   ├── training_metrics.png (📊 visualizations)")
    print(f"   ├── training_metrics.json (📄 raw data)")
    print(f"   └── hardware_config.json (⚙️ hardware settings)")
    print("=" * 60)
    print("\n🎉 Training completed successfully!")


if __name__ == "__main__":
    # Optional: ensure HF offline behavior if that’s your environment
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # os.environ['HF_DATASETS_OFFLINE'] = '1'
    # os.environ['TRANSFORMERS_OFFLINE'] = '1'
    # os.environ['HF_HUB_OFFLINE'] = '1'
    main()

