# 🔴 CRITICAL: Format Mismatch Between Training and Inference

## Problem Summary
Your **fine-tuned model is performing worse than the base model** because the **inference format does NOT match the training format**.

---

## Training Format (sft_new code.py)

```python
def format_prompt(example: Dict[str, str], tokenizer: Any) -> str:
    # ...
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
```

**Key characteristics:**
- ✅ Uses `<|user|>` and `<|assistant|>` **special tokens directly**
- ✅ Manual formatting with explicit newlines
- ✅ **NO system prompt** (trained without one)
- ✅ Ends with `tokenizer.eos_token`
- ✅ For instruction+input: includes `"Input:\n{input_text}\n"` section

**Example training prompt:**
```
<|user|>
What is the capital of France?
<|assistant|>
The capital of France is Paris.
```

or with input:
```
<|user|>
Translate the following text to French.

Input:
Hello, how are you?
<|assistant|>
Bonjour, comment allez-vous?
```

---

## Inference Format (app.py - CURRENT WRONG FORMAT)

```python
def build_messages_with_history(...):
    # Build messages list in proper JSON format
    messages_list = [{"role": "system", "content": system_prompt}]
    
    for exchange in history:
        messages_list.append({"role": "user", "content": exchange['user']})
        messages_list.append({"role": "assistant", "content": exchange['assistant']})
    
    messages_list.append({"role": "user", "content": user_input})
    
    # Convert to JSON string
    messages_json = json.dumps(messages_list)
    
    # Apply chat template
    prompt = tokenizer.apply_chat_template(messages=messages_json, add_generation_prompt=True)
```

**Problems:**
- ❌ Uses JSON format with `role`/`content` (standard chat format)
- ❌ **INCLUDES a system prompt** ("You are a helpful AI assistant.") that was NOT in training data
- ❌ Relies on `apply_chat_template()` to convert JSON → special tokens
- ❌ Format might not match Phi-3.5's expected structure

**What gets tokenized (approximate):**
```
<|system|>
You are a helpful AI assistant.
<|end|>
<|user|>
What is the capital of France?
<|end|>
<|assistant|>
```

---

## The Mismatch

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Format** | `<|user|>...<|assistant|>` | JSON → `apply_chat_template()` |
| **System Prompt** | ❌ None (not in training data) | ✅ "You are a helpful AI assistant." |
| **Special Tokens** | Direct: `<|user|>` | Via tokenizer template |
| **Input Handling** | Explicit "Input:" section | Not handled |
| **EOS Token** | Manual appending | Unknown in template |

### Why This Breaks Fine-Tuning

1. **Distribution Shift**: Model trained on `<|user|>...<|assistant|>` but sees `<|system|>...You are a helpful...<|user|>...` at inference
2. **Token Sequence Mismatch**: The exact byte sequence of tokens is different
3. **System Prompt Contradiction**: Model never saw "You are a helpful AI assistant." during training
4. **Input Section Loss**: If your training data used the "Input:" format, it's completely missing in inference

---

## Solution: Match Training Format Exactly

Replace `build_messages_with_history()` in `app.py` with:

```python
def build_messages_with_history(
    history: List[Dict],
    system_prompt: str,  # This parameter will be IGNORED
    user_input: str,
    tokenizer: og.Tokenizer,
    max_tokens: int
) -> str:
    """Build messages using EXACT training format (direct special tokens)"""
    
    # Start with user message (NO system prompt, NO JSON format)
    prompt_lines = []
    
    # Add historical exchanges
    for exchange in history:
        prompt_lines.append(f"<|user|>\n{exchange['user']}")
        prompt_lines.append(f"<|assistant|>\n{exchange['assistant']}")
    
    # Add current user input
    prompt_lines.append(f"<|user|>\n{user_input}")
    prompt_lines.append("<|assistant|>\n")
    
    # Join with newlines
    prompt = "\n".join(prompt_lines)
    
    # Check token count
    prompt_tokens = len(tokenizer.encode(prompt))
    
    if prompt_tokens < max_tokens - 100:
        return prompt
    
    # If too long, trim oldest exchanges
    max_history = len(history)
    while max_history > 0 and prompt_tokens > max_tokens - 100:
        max_history -= 1
        prompt_lines = []
        
        # Rebuild with fewer exchanges
        for exchange in history[-max_history:]:
            prompt_lines.append(f"<|user|>\n{exchange['user']}")
            prompt_lines.append(f"<|assistant|>\n{exchange['assistant']}")
        
        prompt_lines.append(f"<|user|>\n{user_input}")
        prompt_lines.append("<|assistant|>\n")
        
        prompt = "\n".join(prompt_lines)
        prompt_tokens = len(tokenizer.encode(prompt))
    
    return prompt
```

---

## Required Changes in app.py

### Change 1: Update sidebar (remove system prompt config if not used during training)

**Current:**
```python
system_prompt = st.sidebar.text_area(
    "System Prompt",
    value=DEFAULT_SYSTEM_PROMPT,
    height=100
)
```

**New (comment out or remove):**
```python
# System prompt is NOT used in the fine-tuned model
# (it was trained with direct <|user|> and <|assistant|> tokens)
system_prompt = ""  # Empty string, not used
```

### Change 2: Update the inference code section

Find where you call `build_messages_with_history()` in `generate_response_stream()`:

**Current:**
```python
prompt = build_messages_with_history(
    history=conversation_history,
    system_prompt=system_prompt,
    user_input=user_input,
    tokenizer=tokenizer,
    max_tokens=MAX_CONTEXT_TOKENS
)
```

**New (same call, but function behaves differently):**
```python
# Function now uses training format directly
prompt = build_messages_with_history(
    history=conversation_history,
    system_prompt="",  # Not used in fine-tuned model
    user_input=user_input,
    tokenizer=tokenizer,
    max_tokens=MAX_CONTEXT_TOKENS
)
```

---

## Testing the Fix

After making changes, test with a simple prompt:

```
User Input: "What is 2+2?"

Expected WRONG output (current):
<|system|>
You are a helpful AI assistant.
<|end|>
<|user|>
What is 2+2?
<|end|>
<|assistant|>

Expected CORRECT output (new):
<|user|>
What is 2+2?
<|assistant|>
```

---

## Verification Checklist

- [ ] No `<|system|>` tag in the prompt
- [ ] No "You are a helpful AI assistant." text
- [ ] Prompts start with `<|user|>\n`
- [ ] Responses follow `<|assistant|>\n`
- [ ] Format matches training data exactly
- [ ] Fine-tuned model now performs BETTER than base model

---

## Why This Was Missed

1. `apply_chat_template()` is the "standard" way to format Phi-3.5 messages
2. Adding a system prompt seems like best practice
3. The code worked (no crashes), but produced worse outputs (harder to debug)
4. The training code used a very specific manual format that's easy to overlook

