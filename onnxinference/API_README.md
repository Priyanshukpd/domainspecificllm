# Phi-3.5 ONNX Model API

REST API server for Phi-3.5 mini-instruct ONNX model

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_api.txt
```

### 2. Start the API Server

```bash
python api_server.py
```

Output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Test the API

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Generate Text:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What is artificial intelligence?",
    "max_length": 200,
    "temperature": 0.7
  }'
```

**Interactive Swagger UI:**
Open browser: http://localhost:8000/docs

---

## API Endpoints

### GET `/health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

### POST `/generate`
Generate text from a single prompt

**Request:**
```json
{
  "prompt": "What is machine learning?",
  "max_length": 200,
  "min_length": 10,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 40,
  "do_sample": true,
  "system_prompt": "You are a helpful AI assistant."
}
```

**Response:**
```json
{
  "prompt": "What is machine learning?",
  "generated_text": "Machine learning is a subset of artificial intelligence...",
  "tokens_generated": 45
}
```

---

### POST `/generate-batch`
Generate text for multiple prompts

**Request:**
```json
[
  {"prompt": "What is AI?"},
  {"prompt": "What is ML?"},
  {"prompt": "What is DL?"}
]
```

**Response:**
```json
{
  "results": [
    {
      "prompt": "What is AI?",
      "generated_text": "...",
      "tokens_generated": 42
    },
    ...
  ]
}
```

---

## Python Client Example

```python
from api_client import PhiAPIClient

client = PhiAPIClient()

# Single generation
result = client.generate(
    prompt="Explain quantum computing",
    max_length=300,
    temperature=0.8
)
print(result['generated_text'])

# Batch generation
results = client.generate_batch([
    "What is Python?",
    "What is JavaScript?",
    "What is Rust?"
])
```

---

## Advanced Configuration

### Change Model Path
Edit `api_server.py`:
```python
CONFIG = {
    "model_path": "/your/path/to/onnx/model",
    "execution_provider": "cpu",  # or "cuda", "dml"
}
```

### Enable CUDA
```python
CONFIG = {
    "model_path": "/path/to/model",
    "execution_provider": "cuda",
}
```

### Change Server Port
```bash
python api_server.py --port 5000
```

Or modify in code:
```python
uvicorn.run(app, host="0.0.0.0", port=5000)
```

---

## Docker Deployment

### Build Docker Image

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements_api.txt .
RUN pip install -r requirements_api.txt

COPY api_server.py .
COPY /path/to/model ./model

CMD ["python", "api_server.py"]
```

Build and run:
```bash
docker build -t phi-api .
docker run -p 8000:8000 phi-api
```

---

## Performance Tips

1. **Use CUDA if available** (10x faster)
   ```python
   "execution_provider": "cuda"
   ```

2. **Batch requests** for higher throughput
   ```bash
   curl -X POST http://localhost:8000/generate-batch
   ```

3. **Adjust max_length** - Lower = Faster
   ```json
   {"max_length": 100}
   ```

4. **Use lower temperature** for faster deterministic outputs
   ```json
   {"temperature": 0.5}
   ```

---

## Monitoring

View API logs:
```bash
curl -X GET http://localhost:8000/health
# Check logs from server console
```

View Swagger UI:
```
http://localhost:8000/docs
```

View ReDoc:
```
http://localhost:8000/redoc
```

---

## Troubleshooting

**Model not loading:**
- Check model path exists
- Verify `genai_config.json` is present
- Check memory availability

**Out of memory:**
- Reduce `max_length`
- Use smaller batch size
- Use CPU instead of GPU if needed

**Slow responses:**
- Enable CUDA GPU
- Reduce temperature
- Lower max_length

---

## License

MIT
