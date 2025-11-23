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
import logging
import numpy as np
import torch
from typing import List, Dict, Optional, Any, Iterator
from torch.utils.data import Dataset, IterableDataset
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

# Setup logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# ===============================
# Config
# ===============================
MODEL_PATH = "./MODEL_DIR"                 # local Phi-3.5 Mini Instruct checkpoint dir
OUT_DIR    = "./phi3.5-lora-sft"           # output directory for LoRA adapter + tokenizer
DATA_PATH  = "phi_test_instruction_data.jsonl"  # JSONL file: one JSON object per line (instruction or chat format)

NUM_EPOCHS = 3

# ⭐ NEW: Large dataset config
MAX_SAMPLES = None  # Set to limit (e.g., 1000000 for 1M samples), None = all
CHUNK_SIZE = 5000  # Process in chunks of 10k to avoid memory spikes
CACHE_FILE = "./tokenized_cache.pt"  # Cache tokenized data


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
                "batch_size": 2,              # Reduced due to long sequences
                "grad_accum": 8,             # Increased to maintain effective batch=16
                "gradient_checkpointing": False,  # Enable for long sequences
                "num_workers": 4,
                "use_bf16": True,
                "use_fp16": False,
                "learning_rate": 2e-4,        # Higher LR for GPU
                "max_length": 5120,           # Increased for 17k char instructions (~4.25k tokens + overhead)
                "lora_r": 16,                 # Higher rank
                "lora_alpha": 32,
            }
            print(f"✅ A100 80GB Configuration:")
            print(f"   • Batch Size: {config['batch_size']}")
            print(f"   • Gradient Accumulation: {config['grad_accum']}")
            print(f"   • Effective Batch: {config['batch_size'] * config['grad_accum']}")
            print(f"   • Max Sequence Length: {config['max_length']} (supports ~17k chars)")
            print(f"   • BF16: Enabled")
            print(f"   • Gradient Checkpointing: Enabled (for long sequences)")
            print(f"   • LoRA Rank: {config['lora_r']}")
        else:
            # Fallback for smaller GPUs
            config = {
                "device": "cuda",
                "dtype": torch.bfloat16,
                "batch_size": 1,
                "grad_accum": 8,
                "gradient_checkpointing": True,
                "num_workers": 2,
                "use_bf16": True,
                "use_fp16": False,
                "learning_rate": 1e-4,
                "max_length": 4096,           # Reduced from 6144 for smaller GPUs
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
            "batch_size": 1,                 # Reduced for long sequences
            "grad_accum": 16,                # Increased to maintain effective batch = 16
            "gradient_checkpointing": True,  # Save memory
            "num_workers": 16,               # Utilize all CPU cores (96GB server likely has 32+ cores)
            "use_bf16": False,
            "use_fp16": False,
            "learning_rate": 1e-4,           # Conservative for CPU
            "max_length": 4096,              # Increased but kept lower than GPU for speed
            "lora_r": 8,
            "lora_alpha": 16,
        }
        print(f"✅ CPU 96GB Configuration:")
        print(f"   • Batch Size: {config['batch_size']}")
        print(f"   • Gradient Accumulation: {config['grad_accum']}")
        print(f"   • Effective Batch: {config['batch_size'] * config['grad_accum']}")
        print(f"   • Max Sequence Length: {config['max_length']} (supports ~11k chars)")
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
def format_prompt(instruction: str, input_text: str = "") -> str:
    """Phi-3.5 chat template format for instruction/input/output format."""
    if input_text:
        return f"<|user|>\n{instruction}\n\nInput:\n{input_text}\n<|assistant|>\n"
    else:
        return f"<|user|>\n{instruction}\n<|assistant|>\n"


def format_chat_messages(messages: List[Dict[str, str]]) -> tuple[str, str]:
    """
    Format chat messages into Phi-3.5 chat template.
    Returns (prompt, output) tuple.
    
    Handles format: {"messages": [{"role": "system/user/assistant", "content": "..."}]}
    """
    prompt_parts = []
    output = ""
    
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            # System message goes first
            prompt_parts.append(f"<|system|>\n{content}<|end|>")
        elif role == "user":
            # User message
            prompt_parts.append(f"<|user|>\n{content}<|end|>")
        elif role == "assistant":
            # Assistant message is the output (last message should be assistant)
            output = content
    
    # Join all prompt parts and add assistant tag
    prompt = "\n".join(prompt_parts) + "\n<|assistant|>\n"
    
    return prompt, output


def detect_sample_format(sample: Dict) -> str:
    """
    Detect the format of a sample.
    Returns: 'instruction' or 'chat'
    """
    if "messages" in sample:
        return "chat"
    elif "instruction" in sample:
        return "instruction"
    else:
        logger.warning(f"Unknown format for sample: {sample.keys()}")
        return "unknown"


def format_list_output(output: Any) -> str:
    """
    Convert various output formats (string, list, dict) to properly formatted text.
    Handles cases where output is a JSON list, string list, or nested structure.
    """
    if isinstance(output, str):
        return output.strip()
    
    elif isinstance(output, list):
        # Format list items with bullet points
        formatted_items = []
        for item in output:
            if isinstance(item, dict):
                # Handle dict items (e.g., {"point": "text"})
                item_text = str(item.get("point", item.get("text", item)))
                formatted_items.append(f"• {item_text}")
            elif isinstance(item, (list, tuple)):
                # Handle nested lists
                formatted_items.append(f"• {' → '.join(str(x) for x in item)}")
            else:
                # Simple string/number items
                formatted_items.append(f"• {str(item).strip()}")
        return "\n".join(formatted_items) if formatted_items else ""
    
    elif isinstance(output, dict):
        # Handle dict outputs (convert to readable format)
        formatted_items = []
        for key, value in output.items():
            if isinstance(value, list):
                formatted_items.append(f"{key}:\n" + "\n".join(f"  • {v}" for v in value))
            else:
                formatted_items.append(f"{key}: {value}")
        return "\n".join(formatted_items) if formatted_items else ""
    
    else:
        # Fallback for other types (int, float, None, etc.)
        return str(output).strip() if output else ""


def load_json_streamed(json_path: str, max_samples: Optional[int] = None) -> Iterator[Dict]:
    """
    ⭐ Generator that yields samples from JSON one by one.
    Supports both list format and line-by-line JSON (JSONL).
    Memory-efficient for large datasets.
    """
    count = 0
    
    try:
        # Try standard JSON list format first
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = data.get('data', data.get('samples', []))
        
        logger.info(f"📊 Loaded JSON format with {len(data)} samples")
        for sample in data:
            if max_samples and count >= max_samples:
                break
            yield sample
            count += 1
    except (json.JSONDecodeError, MemoryError):
        # Fall back to line-by-line JSON (JSONL format)
        logger.info("📊 Standard JSON failed, trying line-by-line JSONL format...")
        with open(json_path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_samples and count >= max_samples:
                    break
                try:
                    sample = json.loads(line.strip())
                    if sample:  # Skip empty lines
                        yield sample
                        count += 1
                except json.JSONDecodeError:
                    logger.warning(f"⚠️  Skipping malformed JSON line: {line[:100]}")
                    continue


class StreamingTokenizedDataset(IterableDataset):
    """
    ⭐ CRITICAL: Memory-efficient streaming dataset for large files (5GB+).
    
    For 12M samples:
    - Old approach: Load full dataset → tokenize all → crash with OOM
    - New approach: Load CHUNK_SIZE (10k) samples → tokenize chunk → yield items
    - Memory saved: ~20-30GB → ~2GB peak!
    
    Tokenizes samples on-the-fly in chunks to avoid OOM errors.
    """
    
    def __init__(
        self,
        json_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        chunk_size: int = 10000,
        max_samples: Optional[int] = None,
    ):
        self.json_path = json_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self._sample_count = None
        self.device = "cpu"  # For IterableDataset, we don't move tensors here
        self.truncated_count = 0  # Track truncated sequences
    
    def _count_samples(self) -> int:
        """Count total samples without loading everything into memory."""
        if self._sample_count is not None:
            return self._sample_count
        
        logger.info("📊 Counting samples (large file—this may take a moment)...")
        count = 0
        for _ in load_json_streamed(self.json_path, self.max_samples):
            count += 1
            if count % 100000 == 0:
                logger.info(f"   └─ Counted {count:,} samples...")
        
        self._sample_count = count
        logger.info(f"✅ Total samples: {count:,}")
        return count
    
    def _load_samples_batch(self, start_idx: int, batch_size: int) -> List[Dict]:
        """Load a batch of samples from JSON stream."""
        samples = []
        current_idx = 0
        
        for sample in load_json_streamed(self.json_path, self.max_samples):
            if current_idx >= start_idx + batch_size:
                break
            if current_idx >= start_idx:
                samples.append(sample)
            current_idx += 1
        
        return samples
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """
        Main iterator: yields tokenized samples in chunks.
        Each chunk is ~500MB (10k samples) instead of 20-30GB for full dataset.
        Supports both instruction/input/output and chat message formats.
        """
        total_samples = self._count_samples()
        
        for chunk_start in range(0, total_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_samples)
            logger.info(f"📦 Loading chunk: {chunk_start:,} → {chunk_end:,} ({chunk_end - chunk_start:,} samples)...")
            
            chunk_samples = self._load_samples_batch(chunk_start, chunk_end - chunk_start)
            
            for sample in chunk_samples:
                # Detect sample format
                sample_format = detect_sample_format(sample)
                
                if sample_format == "chat":
                    # Chat format: {"messages": [{"role": "...", "content": "..."}]}
                    messages = sample.get('messages', [])
                    prompt, output = format_chat_messages(messages)
                elif sample_format == "instruction":
                    # Instruction format: {"instruction": "...", "input": "...", "output": "..."}
                    instruction = sample.get('instruction', '')
                    input_text = sample.get('input', '')
                    output_raw = sample.get('output', '')
                    
                    # Handle list/dict outputs
                    output = format_list_output(output_raw)
                    prompt = format_prompt(instruction, input_text)
                else:
                    # Unknown format - skip
                    logger.warning(f"⚠️  Skipping sample with unknown format")
                    continue
                
                # Validate output is not empty
                if not output or not output.strip():
                    logger.warning(f"⚠️  Skipping sample with empty output")
                    continue
                
                # Tokenize prompt and output separately to track lengths
                prompt_enc = self.tokenizer(prompt, truncation=False, add_special_tokens=False)
                output_enc = self.tokenizer(output, truncation=False, add_special_tokens=False)
                
                # Combine: prompt + output + EOS
                input_ids = prompt_enc['input_ids'] + output_enc['input_ids'] + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                
                # Create labels: -100 for prompt (ignored in loss), real IDs for output
                prompt_len = len(prompt_enc['input_ids'])
                labels = [-100] * prompt_len + output_enc['input_ids'] + [self.tokenizer.eos_token_id]
                
                # Truncate if exceeds max_length
                if len(input_ids) > self.max_length:
                    self.truncated_count += 1
                    if self.truncated_count <= 10:  # Log first 10 truncations
                        logger.warning(f"⚠️  Sequence truncated: {len(input_ids)} tokens → {self.max_length} tokens")
                    elif self.truncated_count == 100 or self.truncated_count % 1000 == 0:
                        logger.info(f"📊 Total truncated sequences so far: {self.truncated_count:,}")
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    labels = labels[:self.max_length]
                
                # Pad to max_length
                pad_len = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len  # Padding tokens ignored in loss
                
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                }
        
        # Log final truncation summary
        if self.truncated_count > 0:
            logger.info(f"📊 Dataset processing complete. Total truncated sequences: {self.truncated_count:,} ({self.truncated_count/total_samples*100:.2f}%)")


class TrainDataset(StreamingTokenizedDataset):
    """Training split of streaming dataset (first 90%)."""
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate only over training samples (90% of data). Supports both formats."""
        total_samples = self._count_samples()
        train_samples = int(total_samples * 0.9)
        
        logger.info(f"🎓 [TRAIN SPLIT] Processing {train_samples:,} samples (90%)...")
        
        for chunk_start in range(0, train_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, train_samples)
            logger.info(f"   ├─ Chunk: {chunk_start:,} → {chunk_end:,}")
            
            samples = self._load_samples_batch(chunk_start, chunk_end - chunk_start)
            for sample in samples:
                # Detect sample format
                sample_format = detect_sample_format(sample)
                
                if sample_format == "chat":
                    messages = sample.get('messages', [])
                    prompt, output = format_chat_messages(messages)
                elif sample_format == "instruction":
                    instruction = sample.get('instruction', '')
                    input_text = sample.get('input', '')
                    output_raw = sample.get('output', '')
                    
                    # Handle list/dict outputs
                    output = format_list_output(output_raw)
                    prompt = format_prompt(instruction, input_text)
                else:
                    continue
                
                # Validate output is not empty
                if not output or not output.strip():
                    continue
                
                # Tokenize prompt and output
                prompt_enc = self.tokenizer(prompt, truncation=False, add_special_tokens=False)
                output_enc = self.tokenizer(output, truncation=False, add_special_tokens=False)
                
                input_ids = prompt_enc['input_ids'] + output_enc['input_ids'] + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                prompt_len = len(prompt_enc['input_ids'])
                labels = [-100] * prompt_len + output_enc['input_ids'] + [self.tokenizer.eos_token_id]
                
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    labels = labels[:self.max_length]
                
                pad_len = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
                
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                }


class ValDataset(StreamingTokenizedDataset):
    """Validation split of streaming dataset (last 10%)."""
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate only over validation samples (last 10% of data). Supports both formats."""
        total_samples = self._count_samples()
        train_samples = int(total_samples * 0.9)
        
        logger.info(f"📝 [VAL SPLIT] Processing {total_samples - train_samples:,} samples (10%)...")
        
        for chunk_start in range(train_samples, total_samples, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_samples)
            logger.info(f"   ├─ Chunk: {chunk_start:,} → {chunk_end:,}")
            
            samples = self._load_samples_batch(chunk_start, chunk_end - chunk_start)
            for sample in samples:
                # Detect sample format
                sample_format = detect_sample_format(sample)
                
                if sample_format == "chat":
                    messages = sample.get('messages', [])
                    prompt, output = format_chat_messages(messages)
                elif sample_format == "instruction":
                    instruction = sample.get('instruction', '')
                    input_text = sample.get('input', '')
                    output_raw = sample.get('output', '')
                    
                    # Handle list/dict outputs
                    output = format_list_output(output_raw)
                    prompt = format_prompt(instruction, input_text)
                else:
                    continue
                
                # Validate output is not empty
                if not output or not output.strip():
                    continue
                
                # Tokenize prompt and output
                prompt_enc = self.tokenizer(prompt, truncation=False, add_special_tokens=False)
                output_enc = self.tokenizer(output, truncation=False, add_special_tokens=False)
                
                input_ids = prompt_enc['input_ids'] + output_enc['input_ids'] + [self.tokenizer.eos_token_id]
                attention_mask = [1] * len(input_ids)
                prompt_len = len(prompt_enc['input_ids'])
                labels = [-100] * prompt_len + output_enc['input_ids'] + [self.tokenizer.eos_token_id]
                
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[:self.max_length]
                    attention_mask = attention_mask[:self.max_length]
                    labels = labels[:self.max_length]
                
                pad_len = self.max_length - len(input_ids)
                input_ids += [self.tokenizer.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len
                
                yield {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                }


# ===============================
# Training Visualization Callback
# ===============================
class MetricsCallback(TrainerCallback):
    """
    Callback to track and plot training metrics in real-time.
    Includes gradient norm monitoring.
    Generates charts at specific intervals during training.
    """
    def __init__(self, output_dir: str, grad_log_frequency: int = 100, plot_frequency: int = 500):
        self.output_dir = output_dir
        self.grad_log_frequency = grad_log_frequency
        self.plot_frequency = plot_frequency  # Generate chart every N steps
        self.train_losses = []
        self.eval_losses = []
        self.learning_rates = []
        self.grad_norms = []
        self.steps = []
        self.eval_steps = []
        self.grad_steps = []
        self.last_plot_step = 0  # Track when we last generated a plot
        
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
            
            # Generate chart at specified intervals
            if state.global_step - self.last_plot_step >= self.plot_frequency:
                print(f"\n📊 Generating training chart at step {state.global_step}...")
                self.plot_metrics()
                self.last_plot_step = state.global_step
    
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
        fig.suptitle(f'Training Metrics (Step {current_step:,})', 
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
        
        # Save plot with step number in filename
        current_step = self.steps[-1] if self.steps else 0
        plot_path = os.path.join(self.output_dir, f"training_metrics_step_{current_step}.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='#E8E8E8')
        print(f"📊 Training plot saved to: {plot_path}")
        
        # Also save a "latest" version for easy access
        latest_plot_path = os.path.join(self.output_dir, "training_metrics_latest.png")
        plt.savefig(latest_plot_path, dpi=300, bbox_inches='tight', facecolor='#E8E8E8')
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


# ===============================
# Main
# ===============================
def main() -> None:
    """Main training function."""
    # Validate critical paths before starting
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH}\n   Please check MODEL_PATH in config section.")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Data file not found at: {DATA_PATH}\n   Please check DATA_PATH in config section.")
    
    print(f"✅ Model path verified: {MODEL_PATH}")
    print(f"✅ Data path verified: {DATA_PATH}")
    
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

    # ---- Data (⭐ Streaming for large datasets)
    print(f"\n📊 Loading dataset from: {DATA_PATH}")
    print(f"   ⭐ Using StreamingTokenizedDataset for memory efficiency")
    print(f"   └─ CHUNK_SIZE: {CHUNK_SIZE:,} samples per chunk (~500MB each)")
    
    # Create streaming datasets
    train_ds = TrainDataset(
        json_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=max_length,
        chunk_size=CHUNK_SIZE,
        max_samples=MAX_SAMPLES
    )
    
    val_ds = ValDataset(
        json_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=max_length,
        chunk_size=CHUNK_SIZE,
        max_samples=MAX_SAMPLES
    )
    
    # For IterableDataset, estimate steps based on approximate sample count
    total_samples = train_ds._count_samples()
    train_samples = int(total_samples * 0.9)
    val_samples = total_samples - train_samples
    
    print(f"   • Total samples: {total_samples:,}")
    print(f"   • Train samples: {train_samples:,} (90%)")
    print(f"   • Validation samples: {val_samples:,} (10%)")
    
    steps_per_epoch = max(1, train_samples // (hw_config["batch_size"] * hw_config["grad_accum"]))
    print(f"   • Estimated steps per epoch: {steps_per_epoch:,}")


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
        max_steps=steps_per_epoch * NUM_EPOCHS,  # Fix #1: Required for IterableDataset

        learning_rate=hw_config["learning_rate"],
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",

        logging_steps=max(10, steps_per_epoch // 10),
        eval_steps=max(50, steps_per_epoch // 2),
        eval_strategy="steps",
        save_strategy="epoch",
        save_total_limit=2,  # Keep only last 2 checkpoints to save disk space

        fp16=hw_config["use_fp16"],
        bf16=hw_config["use_bf16"],

        report_to="none",
        logging_dir=os.path.join(OUT_DIR, "logs"),

        max_grad_norm=1.0,
        weight_decay=0.01,

        group_by_length=False,  # Fix #2: Must be False for IterableDataset
        dataloader_pin_memory=False,  # Fix #3: Reduce memory usage
        dataloader_num_workers=0,  # Fix #3: Prevent OOM with large datasets
        
        gradient_checkpointing=hw_config["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False} if hw_config["gradient_checkpointing"] else None,
    )

    # ---- Initialize Callbacks
    # Generate chart every 500 steps (adjust plot_frequency as needed)
    # For 12M samples: ~500 steps = ~2-3 times per epoch on A100
    metrics_callback = MetricsCallback(
        output_dir=OUT_DIR, 
        grad_log_frequency=100,
        plot_frequency=500  # Change this to control chart generation frequency
    )

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
    print(f"✅ Training Samples: {train_samples:,}")
    print(f"✅ Validation Samples: {val_samples:,}")
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

