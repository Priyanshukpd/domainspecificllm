"""
Client for Phi-3.5 ONNX Model API
Examples of how to call the API
"""

import requests
import json
from typing import Optional

BASE_URL = "http://localhost:8000"

class PhiAPIClient:
    """Client for Phi Model API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> dict:
        """Check if API is healthy"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def generate(
        self,
        prompt: str,
        max_length: int = 200,
        temperature: float = 0.7,
        system_prompt: str = "You are a helpful AI assistant."
    ) -> dict:
        """Generate text from prompt"""
        payload = {
            "prompt": prompt,
            "max_length": max_length,
            "temperature": temperature,
            "system_prompt": system_prompt
        }
        
        response = requests.post(
            f"{self.base_url}/generate",
            json=payload
        )
        return response.json()
    
    def generate_batch(self, prompts: list[str]) -> dict:
        """Generate text for multiple prompts"""
        payload = [{"prompt": p} for p in prompts]
        response = requests.post(
            f"{self.base_url}/generate-batch",
            json=payload
        )
        return response.json()

# Example usage
if __name__ == "__main__":
    client = PhiAPIClient()
    
    # Health check
    print("1️⃣  Health Check:")
    print(json.dumps(client.health_check(), indent=2))
    print()
    
    # Single generation
    print("2️⃣  Single Generation:")
    result = client.generate(
        prompt="What is artificial intelligence?",
        max_length=150,
        temperature=0.7
    )
    print(json.dumps(result, indent=2))
    print()
    
    # Batch generation
    print("3️⃣  Batch Generation:")
    results = client.generate_batch([
        "What is Python?",
        "What is machine learning?",
        "Explain neural networks"
    ])
    print(json.dumps(results, indent=2))
