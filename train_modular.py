"""
Main Training Script - Modular Version
Domain-Adaptive Pre-Training (DAPT) for Phi-3.5-mini
"""
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import random

# Import custom modules from the modules folder
from modules.config import MODEL_PATH, detect_hardware
from modules.datasets import TextFileDataset, TokenizedDataset, SubsetDataset
from modules.callbacks import (
    StepLoggingCallback,
    SafeEarlyStoppingCallback,
    PlottingCallback,
    EnhancedEvalCallback
)
from modules.utils import get_latest_checkpoint, print_training_summary


def load_model_and_tokenizer(model_path, device, use_bf16):
    """Load model and tokenizer in offline mode."""
    print("Loading model and tokenizer (OFFLINE MODE)...")
    
    # Try flash attention, fallback to eager if not available
    try:
        import flash_attn
        attn_impl = "flash_attention_2" if device == "cuda" else "eager"
    except ImportError:
        attn_impl = "eager"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
        attn_implementation=attn_impl,
        local_files_only=True
    )
    
    model.config.use_cache = False
    
    if device == "cpu":
        model = model.to("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    # Verify pad token
    if tokenizer.pad_token is None:
        print("⚠️  No pad token found - setting to EOS token")
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    else:
        print(f"✅ Pad token: {tokenizer.pad_token} (id: {tokenizer.pad_token_id})")
    
    print(f"✅ Model loaded: {model.num_parameters() / 1e9:.2f}B parameters\n")
    
    return model, tokenizer


def load_and_tokenize_data(tokenizer, max_length=1024, stride=256):
    """Load text files and tokenize with weighted sampling."""
    print("Loading datasets (OFFLINE MODE)...")
    
    # Combine datasets with 80% domain, 20% general weighting
    combined_texts = TextFileDataset(
        ["domain.txt", "general.txt"],
        weights=[0.8, 0.2],
        shuffle=True,
        seed=42
    )
    print()
    
    print("Tokenizing dataset...")
    tokenized_dataset = TokenizedDataset(
        combined_texts,
        tokenizer,
        max_length=max_length,
        stride=stride
    )
    print()
    
    return tokenized_dataset


def create_train_test_split(dataset, train_ratio=0.9, seed=42):
    """Split dataset into train and test sets."""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    
    # Create indices for split
    indices = list(range(dataset_size))
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # Create subset datasets
    train_dataset = SubsetDataset(dataset, train_indices)
    eval_dataset = SubsetDataset(dataset, test_indices)
    
    print(f"📊 Dataset Split:")
    print(f"   Training examples: {len(train_dataset)}")
    print(f"   Validation examples: {len(eval_dataset)}")
    print()
    
    return train_dataset, eval_dataset


def create_training_args(batch_size, grad_accum, use_bf16, workers, 
                         gradient_checkpointing, steps_per_epoch):
    """Create training arguments."""
    
    # Hybrid checkpointing
    save_strategy = "steps"
    save_steps = max(steps_per_epoch // 2, 500)
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
        eval_steps=200,
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
        gradient_checkpointing=gradient_checkpointing,
        
        # Monitoring
        logging_steps=10,
        logging_first_step=True,
        logging_dir="./logs",
        report_to="tensorboard",
        disable_tqdm=False,
        
        # Stability
        max_grad_norm=1.0,
        weight_decay=0.01,
    )
    
    return training_args


def print_training_config(device, use_bf16, batch_size, grad_accum, 
                         training_args, steps_per_epoch):
    """Print training configuration summary."""
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
    print(f"   Checkpoint every: {training_args.save_steps} steps")
    print(f"   Eval every: {training_args.eval_steps} steps")
    print("="*70)
    print()


def main():
    """Main training function."""
    
    # ========== HARDWARE DETECTION ==========
    hw_config = detect_hardware()
    
    # ========== LOAD MODEL & TOKENIZER ==========
    model, tokenizer = load_model_and_tokenizer(
        MODEL_PATH,
        hw_config['device'],
        hw_config['use_bf16']
    )
    
    # ========== LOAD & TOKENIZE DATA ==========
    tokenized_dataset = load_and_tokenize_data(
        tokenizer,
        max_length=1024,
        stride=256
    )
    
    # ========== TRAIN/TEST SPLIT ==========
    train_dataset, eval_dataset = create_train_test_split(tokenized_dataset)
    
    # ========== CALCULATE STEPS ==========
    examples_per_epoch = len(tokenized_dataset) * 0.9
    steps_per_epoch = int(examples_per_epoch / (hw_config['batch_size'] * hw_config['grad_accum']))
    
    # ========== TRAINING ARGS ==========
    training_args = create_training_args(
        hw_config['batch_size'],
        hw_config['grad_accum'],
        hw_config['use_bf16'],
        hw_config['workers'],
        hw_config['gradient_checkpointing'],
        steps_per_epoch
    )
    
    print_training_config(
        hw_config['device'],
        hw_config['use_bf16'],
        hw_config['batch_size'],
        hw_config['grad_accum'],
        training_args,
        steps_per_epoch
    )
    
    # ========== TEST PROMPTS ==========
    test_prompts = [
        "What is the difference between whole life and term life insurance?",
        "Explain how underwriting risk assessment works in commercial property insurance:",
        "Define loss ratio and its significance for insurance companies:"
    ]
    
    # ========== DATA COLLATOR ==========
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # ========== CREATE TRAINER ==========
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[
            SafeEarlyStoppingCallback(min_steps=steps_per_epoch, patience=3),
            EnhancedEvalCallback(tokenizer, test_prompts),
            StepLoggingCallback(),
            PlottingCallback()
        ]
    )
    
    # ========== TRAIN ==========
    print("="*70)
    print("🚀 Starting training (OFFLINE MODE)...")
    print("="*70)
    print()
    
    # Auto-resume from checkpoint
    latest_checkpoint = get_latest_checkpoint("./output")
    if latest_checkpoint:
        print(f"🔄 Auto-resuming from: {latest_checkpoint}\n")
    
    try:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
        
        # Save final model in multiple precisions
        print_training_summary(trainer)
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
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user (Ctrl+C)")
        print("   Partial checkpoints saved in: ./output/")
        print("   Run the script again to auto-resume!")
        
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        print("   Check logs in: ./logs/")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
