"""
Dataset Classes for Text Loading and Tokenization
"""
import torch
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
        
        import os
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


class SubsetDataset(TorchDataset):
    """Creates a subset of a dataset using indices."""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
