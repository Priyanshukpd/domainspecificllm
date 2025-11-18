"""
Configuration and Hardware Detection
"""
import torch
import os

# ============================================================================
# OFFLINE MODE - Disable all HuggingFace Hub connections
# ============================================================================
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================================================
# PATHS
# ============================================================================
MODEL_PATH = "/data02/WorkingSLIM/PHI_SLIM/phi-3-pytorch-phi-3.5-mini-instruct-v2"

# ============================================================================
# HARDWARE CONFIGURATION
# ============================================================================

def detect_hardware():
    """Auto-detect hardware and return optimal training configuration."""
    
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
    
    return {
        'device': device,
        'use_bf16': use_bf16,
        'batch_size': batch_size,
        'grad_accum': grad_accum,
        'gradient_checkpointing': gradient_checkpointing,
        'workers': workers
    }
