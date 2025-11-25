"""
Simple Phi-3.5 Chat Interface (No Conversation History)
Based on model-qa.py reference implementation
"""

import streamlit as st
import onnxruntime_genai as og
import json
import time
from typing import Tuple

# Constants
MAX_OUTPUT_TOKENS = 8000


def get_execution_provider():
    """Auto-detect best execution provider"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda", "CUDA (GPU)"
    except ImportError:
        pass
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            return "dml", "DirectML (GPU)"
        elif "CUDAExecutionProvider" in providers:
            return "cuda", "CUDA (GPU)"
    except ImportError:
        pass
    
    return "cpu", "CPU"


@st.cache_resource
def load_model(model_path: str, provider: str):
    """Load ONNX model and tokenizer"""
    try:
        config = og.Config(model_path)
        
        if provider != "follow_config":
            config.clear_providers()
            if provider != "cpu":
                config.append_provider(provider)
        
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)
        
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def generate_response_stream(
    model: og.Model,
    tokenizer: og.Tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    length_penalty: float,
    placeholder
) -> Tuple[str, int]:
    """Generate response with streaming (model-qa.py style)"""
    
    # Set search options
    search_options = {
        "max_length": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "length_penalty": length_penalty,
        "do_sample": True if temperature > 0 else False,
        "batch_size": 1
    }
    
    # Create generator params
    params = og.GeneratorParams(model)
    params.set_search_options(**search_options)
    
    # Create generator
    generator = og.Generator(model, params)
    
    # Encode prompt and append tokens
    input_tokens = tokenizer.encode(prompt)
    generator.append_tokens(input_tokens)
    
    # Stream tokens
    tokenizer_stream = tokenizer.create_stream()
    full_response = ""
    token_count = 0
    
    try:
        while not generator.is_done():
            generator.generate_next_token()
            
            if not generator.is_done():
                new_token = generator.get_next_tokens()[0]
                token_text = tokenizer_stream.decode(new_token)
                full_response += token_text
                token_count += 1
                
                # Update placeholder with streaming text + cursor
                placeholder.markdown(full_response + "▌")
        
        # Final update without cursor
        placeholder.markdown(full_response)
        
    except Exception as e:
        placeholder.error(f"Generation error: {e}")
        full_response = f"Error: {e}"
    
    finally:
        # Clean up generator
        del generator
    
    return full_response, token_count


def build_prompt(system_prompt: str, user_input: str, tokenizer: og.Tokenizer) -> str:
    """Build prompt using apply_chat_template (model-qa.py style)"""
    
    # Create messages list
    messages_list = []
    
    if system_prompt:
        messages_list.append({"role": "system", "content": system_prompt})
    
    messages_list.append({"role": "user", "content": user_input})
    
    # Convert to JSON string
    messages = json.dumps(messages_list)
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(messages=messages, add_generation_prompt=True)
    except Exception as e:
        st.error(f"Error applying chat template: {e}")
        # Fallback
        prompt = user_input
    
    return prompt


def main():
    st.set_page_config(
        page_title="Phi-3.5 Chat (Simple)",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 Phi-3.5 Chat Interface (No History)")
    st.markdown("*Single-turn conversations - Each message is independent*")
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
            min_value=1000,
            max_value=8000,
            value=3000,
            step=500,
            help="Maximum tokens for response"
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
        
        length_penalty = st.slider(
            "Length Penalty",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1,
            help="Penalty for sequence length. >1.0 encourages longer sequences, <1.0 encourages shorter ones"
        )
        
        # System Prompt
        st.subheader("System Prompt")
        use_system_prompt = st.checkbox("Use System Prompt", value=True)
        
        if use_system_prompt:
            system_prompt = st.text_area(
                "System Prompt",
                value="You are a helpful AI assistant.",
                height=100,
                help="Instructions for the model"
            )
        else:
            system_prompt = ""
        
        # Clear History Button
        if st.button("🗑️ Clear Chat Display", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat cleared!")
    
    # Initialize Session State
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            model, tokenizer = load_model(model_path, provider)
            st.session_state.model = model
            st.session_state.tokenizer = tokenizer
            st.session_state.provider = provider
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Check if model loaded successfully
    if st.session_state.model is None or st.session_state.tokenizer is None:
        st.error("❌ Failed to load model. Check the model path.")
        return
    
    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "timing" in message and message["role"] == "assistant":
                timing = message["timing"]
                st.caption(
                    f"⏱️ {timing['tokens']} tokens • "
                    f"{timing['time']:.2f}s • "
                    f"{timing['tokens_per_sec']:.1f} tok/sec"
                )
    
    # Chat Input
    user_input = st.chat_input("Ask me anything...")
    
    # Process User Input
    if user_input:
        # Add user message to display
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Add to messages
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            start_time = time.time()
            
            # Build prompt (NO HISTORY - just current message)
            prompt = build_prompt(
                system_prompt,
                user_input,
                st.session_state.tokenizer
            )
            
            # Debug: Show prompt format
            with st.expander("🔍 Debug: View Prompt Format"):
                st.code(prompt, language="text")
            
            # Create placeholder for streaming text
            message_placeholder = st.empty()
            
            # Generate response with streaming
            response, response_tokens = generate_response_stream(
                st.session_state.model,
                st.session_state.tokenizer,
                prompt,
                max_output_tokens,
                temperature,
                top_p,
                length_penalty,
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
        
        # Add to messages with timing
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "timing": {
                "tokens": response_tokens,
                "time": generation_time,
                "tokens_per_sec": tokens_per_sec
            }
        })
        
        st.rerun()


if __name__ == "__main__":
    main()
