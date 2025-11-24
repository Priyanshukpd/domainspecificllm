"""
FastAPI server for ONNX Phi-3.5 Model
Exposes model inference as REST API endpoints
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import onnxruntime_genai as og
import uvicorn
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phi-3.5 ONNX Model API",
    description="REST API for Phi-3.5 mini-instruct ONNX model",
    version="1.0.0"
)

# Global model and tokenizer (loaded once at startup)
model = None
tokenizer = None
CONFIG = {
    "model_path": "/Users/munishm/Documents/phi-3.5-mini-instruct/onnx",
    "execution_provider": "cpu",
}

# Request/Response models
class GenerateRequest(BaseModel):
    """Request model for text generation"""
    prompt: str
    max_length: int = 200
    min_length: int = 10
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    do_sample: bool = True
    system_prompt: Optional[str] = "You are a helpful AI assistant."

class GenerateResponse(BaseModel):
    """Response model for text generation"""
    prompt: str
    generated_text: str
    tokens_generated: int

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model, tokenizer
    try:
        logger.info(f"Loading model from {CONFIG['model_path']}...")
        config = og.Config(CONFIG['model_path'])
        
        if CONFIG['execution_provider'] != "follow_config":
            config.clear_providers()
            if CONFIG['execution_provider'] != "cpu":
                config.append_provider(CONFIG['execution_provider'])
        
        model = og.Model(config)
        tokenizer = og.Tokenizer(model)
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {str(e)}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global model, tokenizer
    model = None
    tokenizer = None
    logger.info("Model unloaded")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text based on input prompt
    
    Example:
    {
        "prompt": "What is artificial intelligence?",
        "max_length": 200,
        "temperature": 0.7,
        "system_prompt": "You are a helpful AI assistant."
    }
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Create messages with system prompt
        messages_list = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.prompt}
        ]
        
        import json
        messages = json.dumps(messages_list)
        
        # Apply chat template
        prompt_formatted = tokenizer.apply_chat_template(
            messages=messages, 
            add_generation_prompt=True
        )
        
        # Encode and generate
        input_tokens = tokenizer.encode(prompt_formatted)
        
        params = og.GeneratorParams(model)
        params.set_search_options(
            max_length=request.max_length,
            min_length=request.min_length,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            do_sample=request.do_sample,
            batch_size=1
        )
        
        generator = og.Generator(model, params)
        generator.append_tokens(input_tokens)
        
        # Generate tokens
        tokenizer_stream = tokenizer.create_stream()
        generated_parts = []
        
        while not generator.is_done():
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            generated_parts.append(tokenizer_stream.decode(new_token))
        
        # Join all parts to get complete text
        generated_text = "".join(generated_parts).strip()
        
        # Get output for token count
        output_tokens = generator.get_sequence(0)
        
        # Clean up
        del generator
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated_text,
            tokens_generated=len(output_tokens) - len(input_tokens)
        )
        
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate-batch")
async def generate_batch(requests: list[GenerateRequest]):
    """
    Generate text for multiple prompts
    
    Example:
    [
        {"prompt": "What is AI?"},
        {"prompt": "What is ML?"}
    ]
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for req in requests:
        try:
            result = await generate(req)
            results.append(result)
        except Exception as e:
            logger.error(f"Batch generation error: {str(e)}")
            results.append({"error": str(e)})
    
    return {"results": results}

@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "Phi-3.5 ONNX Model API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health (GET)",
            "generate": "/generate (POST)",
            "generate_batch": "/generate-batch (POST)",
            "docs": "/docs (Swagger UI)",
            "redoc": "/redoc (ReDoc)"
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
