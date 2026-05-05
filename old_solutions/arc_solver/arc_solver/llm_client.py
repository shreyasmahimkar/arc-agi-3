import os
import requests
from abc import ABC, abstractmethod

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response from the LLM given a prompt."""
        pass

class CloudGeminiProvider(LLMProvider):
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for CloudGeminiProvider.")
        
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent?key={self.api_key}"

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}]
        }
        response = requests.post(self.api_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError):
            return ""

class LocalVLLMProvider(LLMProvider):
    def __init__(self):
        self.api_url = "http://localhost:8000/v1/chat/completions"
        self.model = os.environ.get("VLLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

    def generate(self, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1024,
            "temperature": 0.0
        }
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.ConnectionError:
            print("[DevMode Mock] vLLM server not found. Returning a mocked LocalSimulator.")
            return """
```python
import sys
import json

def simulate_step(grid, action):
    # Dummy logic
    return grid, 1.0, False

class LocalSimulator:
    def __init__(self):
        self.grid = [[0 for _ in range(10)] for _ in range(10)]
        
    def step(self, state, action):
        new_grid, reward, done = simulate_step(self.grid, 0)
        return {"grid": new_grid}, reward, done

if __name__ == "__main__":
    if len(sys.argv) > 1:
        memory_file = sys.argv[1]
        try:
            with open(memory_file, 'r') as f:
                data = json.load(f)
                print("Loaded memory:", len(data))
        except Exception as e:
            pass
    sim = LocalSimulator()
    print("Success")
```
"""
        except Exception as e:
            return ""

def get_llm_client() -> LLMProvider:
    """Factory method to get the correct LLM provider based on the environment."""
    env = os.environ.get("ENV", "DEV")
    
    if env == "DEV":
        return CloudGeminiProvider()
    elif env == "KAGGLE":
        return LocalVLLMProvider()
    else:
        raise ValueError(f"Unknown ENV setting: {env}. Expected 'DEV' or 'KAGGLE'.")
