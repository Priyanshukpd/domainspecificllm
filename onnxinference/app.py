import streamlit as st
import onnxruntime_genai as og
import time
import os
from typing import Optional, List, Dict
import json

# ============================================================================
# Configuration
# ============================================================================
DEFAULT_SYSTEM_PROMPT = "You are a helpful AI assistant."
MAX_CONTEXT_TOKENS = 16000
MAX_CONVERSATION_HISTORY = 100  # Max number of exchanges to keep

# ============================================================================
# Helper Functions
# ============================================================================

@st.cache_resource
def get_execution_provider():
    """Detect best available execution provider"""
    try:
        # Check CUDA availability
        providers = og.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            return "cuda", "🚀 CUDA (GPU)"
        elif "DmlExecutionProvider" in providers:
            return "dml", "⚡ DML (GPU)"
    except:
        pass
    return "cpu", "💻 CPU"

@st.cache_resource
def load_model(model_path: str, provider: str):
    """Load ONNX model with specified provider"""
    try:
        config = og.Config(model_path)
        
        if provider != "follow_config":
            config.clear_providers()
            if provider == "cuda":
                config.append_provider("cuda")
            elif provider == "dml":
                config.append_provider("dml")
            # CPU is default
        
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None, None

def count_tokens(tokenizer: og.Tokenizer, text: str) -> int:
    """Count tokens in text"""
    try:
        tokens = tokenizer.encode(text)
        return len(tokens)
    except:
        return len(text.split()) * 1.3  # Rough estimate

def trim_conversation_history(
    history: List[Dict],
    tokenizer: og.Tokenizer,
    system_prompt_tokens: int,
    max_tokens: int
) -> List[Dict]:
    """Remove oldest messages if context exceeds max tokens"""
    current_tokens = system_prompt_tokens
    
    # Calculate tokens for all exchanges
    exchange_tokens = []
    for exchange in history:
        user_tokens = count_tokens(tokenizer, exchange['user'])
        assistant_tokens = count_tokens(tokenizer, exchange['assistant'])
        total = user_tokens + assistant_tokens
        exchange_tokens.append(total)
        current_tokens += total
    
    # Remove oldest exchanges if over limit
    trimmed_history = history.copy()
    idx = 0
    while current_tokens > max_tokens and idx < len(trimmed_history):
        current_tokens -= exchange_tokens[idx]
        trimmed_history.pop(0)
        exchange_tokens.pop(0)
        idx += 1
    
    # Also limit to max exchanges
    if len(trimmed_history) > MAX_CONVERSATION_HISTORY:
        removed = len(trimmed_history) - MAX_CONVERSATION_HISTORY
        trimmed_history = trimmed_history[-MAX_CONVERSATION_HISTORY:]
        return trimmed_history, removed
    
    return trimmed_history, idx

def build_messages_with_history(
    history: List[Dict],
    system_prompt: str,
    user_input: str,
    tokenizer: og.Tokenizer,
    max_tokens: int
) -> str:
    """Build messages in JSON format like model-chat.py - lets tokenizer handle formatting"""
    
    # Build messages list in proper JSON format
    messages_list = [{"role": "system", "content": system_prompt}]
    
    # Add all history
    for exchange in history:
        messages_list.append({"role": "user", "content": exchange['user']})
        messages_list.append({"role": "assistant", "content": exchange['assistant']})
    
    # Add current user input
    messages_list.append({"role": "user", "content": user_input})
    
    # Convert to JSON string
    messages_json = json.dumps(messages_list)
    
    # Apply chat template (like model-chat.py does)
    try:
        prompt = tokenizer.apply_chat_template(messages=messages_json, add_generation_prompt=True)
    except:
        # Fallback to manual format if apply_chat_template fails
        prompt = f"System: {system_prompt}\n\n"
        for exchange in history:
            prompt += f"User: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
        prompt += f"User: {user_input}\nAssistant: "
    
    # Check token count
    prompt_tokens = len(tokenizer.encode(prompt))
    
    if prompt_tokens < max_tokens - 100:
        return prompt  # Fits fine
    
    # Too long - trim oldest exchanges
    while len(messages_list) > 2 and prompt_tokens > max_tokens - 100:  # Keep system + current user
        # Remove oldest exchange (after system prompt)
        if len(messages_list) > 2:
            messages_list.pop(1)  # Remove oldest user message
        if len(messages_list) > 2:
            messages_list.pop(1)  # Remove oldest assistant message
        
        messages_json = json.dumps(messages_list)
        try:
            prompt = tokenizer.apply_chat_template(messages=messages_json, add_generation_prompt=True)
        except:
            # Fallback
            prompt = f"System: {system_prompt}\n\n"
            for msg in messages_list[1:-1]:
                if msg['role'] == 'user':
                    prompt += f"User: {msg['content']}\n"
                else:
                    prompt += f"Assistant: {msg['content']}\n\n"
            prompt += f"User: {messages_list[-1]['content']}\nAssistant: "
        
        prompt_tokens = len(tokenizer.encode(prompt))
    
    return prompt

def generate_response_stream(
    model: og.Model,
    tokenizer: og.Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    placeholder
):
    """Generate response with streaming (like model-chat.py) - yields tokens in real-time"""
    try:
        # Check prompt size first
        prompt_tokens = len(tokenizer.encode(prompt))
        available_tokens = max_tokens - prompt_tokens
        
        if available_tokens < 50:
            error_msg = f"ERROR: Prompt too long ({prompt_tokens} tokens). Increase 'Max Output Tokens' in settings (need at least {prompt_tokens + 100} tokens)."
            placeholder.markdown(error_msg)
            return error_msg, 0
        
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            batch_size=1
        )
        
        generator = og.Generator(model, params)
        input_tokens = tokenizer.encode(prompt)
        generator.append_tokens(input_tokens)
        
        # Stream tokens like model-chat.py
        tokenizer_stream = tokenizer.create_stream()
        output_text = ""
        token_count = 0
        
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            token_text = tokenizer_stream.decode(new_token)
            output_text += token_text
            token_count += 1
            
            # Update display in real-time
            placeholder.markdown(output_text + "▌")  # Add cursor
        
        # Final update without cursor
        placeholder.markdown(output_text.strip())
        
        return output_text.strip(), token_count
        
    except Exception as e:
        error_msg = str(e)
        if "exceeds max length" in error_msg:
            error_msg = "ERROR: Token limit exceeded. Increase 'Max Output Tokens' in the sidebar settings."
        else:
            error_msg = f"Error: {error_msg}"
        placeholder.markdown(error_msg)
        return error_msg, 0

# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="Phi-3.5 Chat",
        page_icon="🤖",
        # layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Phi-3.5 Chat Interface")
    st.markdown("---")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model Path
        model_path = st.text_input(
            "Model Path",
            value="/Users/munishm/Documents/phi-3.5-mini-instruct/onnx",
            help="Path to ONNX model directory"
        )
        
        # Auto-detect provider
        provider, provider_name = get_execution_provider()
        st.info(f"🔧 **Detected Provider**: {provider_name}")
        
        # Allow override
        provider_override = st.selectbox(
            "Override Provider",
            options=["auto", "cuda", "dml", "cpu"],
            index=0,
            help="Auto uses detected provider"
        )
        
        if provider_override != "auto":
            provider = provider_override
        
        # Generation Parameters
        st.subheader("Generation Settings")
        max_output_tokens = st.slider(
            "Max Output Tokens",
            min_value=500,
            max_value=5000,
            value=1000,
            step=500,
            help="Total tokens including prompt. Needs to be larger than prompt length"
        )
        
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.1,
            help="Lower = more deterministic, Higher = more creative"
        )
        
        top_p = st.slider(
            "Top P",
            min_value=0.0,
            max_value=1.0,
            value=0.9,
            step=0.05,
            help="Nucleus sampling parameter"
        )
        
        # System Prompt
        st.subheader("System Prompt")
        system_prompt = st.text_area(
            "System Prompt",
            value=DEFAULT_SYSTEM_PROMPT,
            height=100,
            help="Instructions for the AI"
        )
        
        # Context Info
        st.subheader("📊 Context Info")
        st.metric("Max Context Tokens", f"{MAX_CONTEXT_TOKENS:,}")
        st.metric("Max History Exchanges", MAX_CONVERSATION_HISTORY)
        
        # Clear History Button
        if st.button("🗑️ Clear Conversation History", use_container_width=True):
            st.session_state.conversation_history = []
            st.session_state.token_count = 0
            st.success("Conversation cleared!")
    
    # Initialize Session State
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            model, tokenizer = load_model(model_path, provider)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.provider = provider
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    
    # Check if model loaded successfully
    if st.session_state.model is None or st.session_state.tokenizer is None:
        st.error("❌ Failed to load model. Check the model path.")
        return
    
    # # Display Provider Info
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.metric("Provider", st.session_state.provider.upper())
    # with col2:
    #     st.metric("Conversations", len(st.session_state.conversation_history))
    # with col3:
    #     token_usage = (st.session_state.token_count / MAX_CONTEXT_TOKENS) * 100
    #     st.metric(
    #         "Context Usage",
    #         f"{token_usage:.1f}%",
    #         f"{st.session_state.token_count}/{MAX_CONTEXT_TOKENS}"
    #     )
    
    # st.markdown("---")
    
    # Display Chat History using st.chat_message (cleaner UI)
    for exchange in st.session_state.conversation_history:
        # User message
        with st.chat_message("user"):
            st.markdown(exchange['user'])
        
        # Assistant message with timing
        with st.chat_message("assistant"):
            st.markdown(exchange['assistant'])
            if 'timing' in exchange:
                timing = exchange['timing']
                st.caption(
                    f"⏱️ {timing['tokens']} tokens • "
                    f"{timing['time']:.2f}s • "
                    f"{timing['tokens_per_sec']:.1f} tok/sec"
                )
    
    # Chat Input at the bottom (cleaner than text_area + button)
    user_input = st.chat_input("Ask me anything...")
    
    # Process User Input
    if user_input:
        # Add user message to display immediately
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate and display assistant response with streaming
        with st.chat_message("assistant"):
            start_time = time.time()
            
            # Build messages with FULL history (like model-chat.py)
            prompt = build_messages_with_history(
                st.session_state.conversation_history,
                system_prompt,
                user_input,
                st.session_state.tokenizer,
                max_output_tokens
            )
            
            # Create placeholder for streaming text
            message_placeholder = st.empty()
            
            # Generate response with streaming (like model-chat.py)
            response, response_tokens = generate_response_stream(
                st.session_state.model,
                st.session_state.tokenizer,
                prompt,
                max_output_tokens,
                temperature,
                top_p,
                message_placeholder
            )
            
            generation_time = time.time() - start_time
            tokens_per_sec = response_tokens / generation_time if generation_time > 0 else 0
            
            # Show timing info
            st.caption(
                f"⏱️ {response_tokens} tokens • "
                f"{generation_time:.2f}s • "
                f"{tokens_per_sec:.1f} tok/sec"
            )
        
        # Add to history with timing metadata
        st.session_state.conversation_history.append({
            "user": user_input,
            "assistant": response,
            "timing": {
                "tokens": response_tokens,
                "time": generation_time,
                "tokens_per_sec": tokens_per_sec
            }
        })
        
        # Update token count
        user_tokens = count_tokens(st.session_state.tokenizer, user_input)
        st.session_state.token_count += user_tokens + response_tokens
        
        # Trim history if needed
        trimmed_history, removed = trim_conversation_history(
            st.session_state.conversation_history,
            st.session_state.tokenizer,
            count_tokens(st.session_state.tokenizer, system_prompt),
            MAX_CONTEXT_TOKENS
        )
        
        if removed > 0:
            st.session_state.conversation_history = trimmed_history
            st.session_state.token_count = sum(
                count_tokens(st.session_state.tokenizer, ex['user']) +
                count_tokens(st.session_state.tokenizer, ex['assistant'])
                for ex in trimmed_history
            )
            st.toast(f"⚠️ Removed {removed} oldest exchange(s) to fit token limit", icon="⚠️")
        
        st.rerun()

if __name__ == "__main__":
    main()