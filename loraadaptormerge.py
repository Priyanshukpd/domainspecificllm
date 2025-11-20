#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Merge LoRA Adapter with Base Model
Converts: Base Model + LoRA Adapter → Single Fine-tuned Model
"""

import torch
import os
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel



# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_MODEL_PATH = "./MODEL_DIR"              # Original base model (Phi-3.5)
LORA_ADAPTER_PATH = "./phi3.5-lora-sft"     # LoRA adapter from training
OUTPUT_PATH = "./phi3.5-lora-merged"        # Merged model output

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ============================================================================
# MERGE FUNCTION
# ============================================================================

def merge_adapter(base_model_path: str, adapter_path: str, output_path: str) -> None:
    """
    Merge LoRA adapter with base model.
    
    Args:
        base_model_path: Path to base model (Phi-3.5)
        adapter_path: Path to LoRA adapter (phi3.5-lora-sft)
        output_path: Path to save merged model
    """
    
    print("\n" + "=" * 70)
    print("🔀 MERGING LoRA ADAPTER WITH BASE MODEL")
    print("=" * 70)
    
    # Validate paths
    if not Path(base_model_path).exists():
        raise FileNotFoundError(f"Base model not found: {base_model_path}")
    if not Path(adapter_path).exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # ---- Load base model
    print(f"\n📦 Loading base model from: {base_model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else {"": "cpu"},
        trust_remote_code=False,
        local_files_only=True
    )
    print(f"   ✅ Base model loaded")
    
    # ---- Load LoRA adapter
    print(f"\n🔧 Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        device_map="auto" if DEVICE == "cuda" else {"": "cpu"}
    )
    print(f"   ✅ LoRA adapter loaded")
    
    # ---- Merge
    print(f"\n⚙️  Merging adapter into base model...")
    merged_model = model.merge_and_unload()
    print(f"   ✅ Merge completed")
    
    # ---- Load tokenizer
    print(f"\n📝 Loading tokenizer from: {adapter_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,
        trust_remote_code=False,
        local_files_only=True
    )
    print(f"   ✅ Tokenizer loaded (vocab size: {len(tokenizer)})")
    
    # ---- Save merged model
    print(f"\n💾 Saving merged model to: {output_path}")
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"   ✅ Merged model saved")
    
    # ---- Calculate size
    param_count = sum(p.numel() for p in merged_model.parameters())
    
    print("\n" + "=" * 70)
    print("📊 MERGE SUMMARY")
    print("=" * 70)
    print(f"✅ Base Model: {base_model_path}")
    print(f"✅ LoRA Adapter: {adapter_path}")
    print(f"✅ Merged Model: {output_path}")
    print(f"✅ Total Parameters: {param_count/1e9:.2f}B")
    print(f"✅ Data Type: {DTYPE}")
    print("\n📁 Output files:")
    print(f"   ├── pytorch_model.bin (merged weights)")
    print(f"   ├── config.json")
    print(f"   ├── tokenizer.json")
    print(f"   └── special_tokens_map.json")
    print("=" * 70)
    print("\n🎉 Merge completed successfully!")


# ============================================================================
# VERIFY MERGE
# ============================================================================

def verify_merge(merged_model_path: str) -> None:
    """Verify the merged model can be loaded and used."""
    
    print("\n" + "=" * 70)
    print("✅ VERIFYING MERGED MODEL")
    print("=" * 70)
    
    # Load merged model
    print(f"\n📦 Loading merged model from: {merged_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=DTYPE,
        device_map="auto" if DEVICE == "cuda" else {"": "cpu"},
        trust_remote_code=False,
        local_files_only=True
    )
    print(f"   ✅ Merged model loaded successfully")
    
    # Load tokenizer
    print(f"\n📝 Loading tokenizer from: {merged_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        merged_model_path,
        trust_remote_code=False,
        local_files_only=True
    )
    print(f"   ✅ Tokenizer loaded successfully")
    
    # Test generation
    print(f"\n🧪 Testing generation...")
    test_prompt = "<|user|>\nWhat is diabetes?\n<|assistant|>\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)
    
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   ✅ Generation test passed")
    print(f"\n   Sample output:\n   {response[-100:]}")
    
    print("\n" + "=" * 70)
    print("✅ VERIFICATION PASSED - Model is ready to use!")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main merge function."""
    
    try:
        # Merge adapter with base model
        merge_adapter(BASE_MODEL_PATH, LORA_ADAPTER_PATH, OUTPUT_PATH)
        
        # Verify the merge
        verify_merge(OUTPUT_PATH)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()