"""
Example: Using Modular Components Independently

This shows how you can reuse the modular components in other projects.
"""

# Example 1: Just hardware detection
# ====================================
from modules.config import detect_hardware

hw = detect_hardware()
print(f"Detected device: {hw['device']}")
print(f"Batch size: {hw['batch_size']}")
print(f"Gradient accumulation: {hw['grad_accum']}")


# Example 2: Load your own dataset
# ====================================
from modules.datasets import TextFileDataset, TokenizedDataset
from transformers import AutoTokenizer

# Load custom files with different weighting
dataset = TextFileDataset(
    ["domain.txt", "general.txt"],
    weights=[0.7, 0.3],  # 70/30 split instead of 80/20
    shuffle=True
)

# Tokenize with custom settings
# tokenizer = AutoTokenizer.from_pretrained("your-model")
# tokenized = TokenizedDataset(
#     dataset,
#     tokenizer,
#     max_length=512,  # Shorter sequences
#     stride=128
# )


# Example 3: Use callbacks in another training loop
# ===================================================
from modules.callbacks import PlottingCallback, SafeEarlyStoppingCallback
from transformers import Trainer

# Just add to your existing trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=eval_dataset,
#     callbacks=[
#         SafeEarlyStoppingCallback(min_steps=500, patience=5),
#         PlottingCallback(output_dir="./my_plots")
#     ]
# )


# Example 4: Resume from checkpoint
# ===================================
from modules.utils import get_latest_checkpoint

checkpoint = get_latest_checkpoint("./output")
if checkpoint:
    print(f"Found checkpoint: {checkpoint}")
    # trainer.train(resume_from_checkpoint=checkpoint)
else:
    print("No checkpoint found, starting fresh")


# Example 5: Custom weighted dataset
# ====================================
# Load 3 files with different weights
custom_dataset = TextFileDataset(
    ["file1.txt", "file2.txt", "file3.txt"],
    weights=[0.5, 0.3, 0.2],  # 50%, 30%, 20%
    shuffle=True,
    seed=123
)
print(f"Loaded {len(custom_dataset)} examples")


print("\n✅ All examples work! You can mix and match these components.")
