import os
import sys
import json
import gc
import signal
import traceback
from datasets import Dataset
from transformers import AutoTokenizer
from phi_lora_sft_final_plots import format_prompt, MODEL_PATH, DATA_PATH
from tqdm import tqdm

# ========== Config ==========
TOKENIZED_OUT_DIR = "tokenized_dataset"
CHUNK_SIZE = 5000  # adjust to your RAM
MAX_LENGTH_CAP = 4096  # safer than relying on tokenizer.model_max_length
PROGRESS_LOG = os.path.join(TOKENIZED_OUT_DIR, "progress.log")

# ========== Utilities ==========
def log(msg: str):
    print(msg)
    sys.stdout.flush()
    try:
        os.makedirs(TOKENIZED_OUT_DIR, exist_ok=True)
        with open(PROGRESS_LOG, "a", encoding="utf-8") as lf:
            lf.write(msg.rstrip() + "\n")
    except Exception:
        # avoid blocking on logging failure
        pass

def handle_signal(signum, frame):
    log(f"[SIGNAL {signum}] Received termination signal. Exiting safely.")
    sys.exit(1)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

def is_valid_finished_chunk_dir(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    # Must have Arrow files + dataset info + DONE marker
    data_arrow = os.path.join(path, "data.arrow")
    ds_info = os.path.join(path, "dataset_info.json")
    done_marker = os.path.join(path, "_DONE")
    return os.path.exists(data_arrow) and os.path.exists(ds_info) and os.path.exists(done_marker)

def mark_done(chunk_dir: str):
    with open(os.path.join(chunk_dir, "_DONE"), "w", encoding="utf-8") as f:
        f.write("ok")

# ========== Prepare ==========
os.makedirs(TOKENIZED_OUT_DIR, exist_ok=True)

log(f"Loading tokenizer from: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=False,
    local_files_only=True
)

# Ensure pad token exists (common for GPT-style)
if tokenizer.pad_token_id is None:
    tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    log(f"Pad token was missing; set pad_token_id = eos_token_id ({tokenizer.pad_token_id}).")

# Safe max length (avoid huge sentinel values)
context_len = min(
    MAX_LENGTH_CAP,
    getattr(tokenizer, "model_max_length", MAX_LENGTH_CAP) or MAX_LENGTH_CAP
)

log(f"Loading raw data from: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f if line.strip()]

total = len(data)
log(f"Tokenizing {total} samples in chunks of {CHUNK_SIZE}...")

chunks = [data[i:i+CHUNK_SIZE] for i in range(0, total, CHUNK_SIZE)]

# ========== Main Loop ==========
for idx, chunk in enumerate(tqdm(chunks, desc="Chunks", unit="chunk", dynamic_ncols=True)):
    chunk_dir = os.path.join(TOKENIZED_OUT_DIR, f"chunk_{idx:04d}")
    tmp_dir = os.path.join(TOKENIZED_OUT_DIR, f".chunk_{idx:04d}.tmp")

    # Resume-friendly: skip fully finished chunks
    if is_valid_finished_chunk_dir(chunk_dir):
        log(f"Skipping chunk {idx} (already valid & finished at {chunk_dir})")
        continue

    # If a leftover tmp dir exists (interrupted earlier), clean it up
    if os.path.isdir(tmp_dir):
        log(f"Cleaning up stale tmp dir: {tmp_dir}")
        try:
            import shutil
            shutil.rmtree(tmp_dir)
        except Exception as e:
            log(f"Failed to remove tmp dir {tmp_dir}: {e}")

    tokenized = []
    try:
        for ex in tqdm(chunk, desc=f"Tokenizing chunk {idx+1}/{len(chunks)}", unit="sample", dynamic_ncols=True):
            prompt = format_prompt(ex, tokenizer)
            enc = tokenizer(
                prompt,
                truncation=True,
                max_length=context_len,
                padding="max_length",
            )
            labels = enc["input_ids"].copy()
            pad_id = tokenizer.pad_token_id
            labels = [l if l != pad_id else -100 for l in labels]
            enc["labels"] = labels
            tokenized.append(enc)

        ds = Dataset.from_dict({
            "input_ids": [ex["input_ids"] for ex in tokenized],
            "attention_mask": [ex["attention_mask"] for ex in tokenized],
            "labels": [ex["labels"] for ex in tokenized],
        })

        os.makedirs(tmp_dir, exist_ok=True)
        log(f"Saving chunk {idx} atomically: {tmp_dir} -> {chunk_dir}")
        ds.save_to_disk(tmp_dir)

        # Rename tmp -> final atomically, then mark DONE
        os.replace(tmp_dir, chunk_dir)
        mark_done(chunk_dir)

        log(f"[OK] Chunk {idx} saved at {chunk_dir} (samples: {len(ds)})")

    except Exception as e:
        log(f"[ERROR] Chunk {idx} failed: {e}\n{traceback.format_exc()}")
        # Best-effort cleanup of tmp dir
        try:
            if os.path.isdir(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir)
                log(f"Removed tmp dir after failure: {tmp_dir}")
        except Exception as ce:
            log(f"Cleanup failed for tmp dir {tmp_dir}: {ce}")
        # Exit early to avoid silent partial state
        sys.exit(2)
    finally:
        # Free memory between chunks
        tokenized.clear()
        gc.collect()

log("All chunks processed and saved safely.")
