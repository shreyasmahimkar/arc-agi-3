import os
import re
import requests
from typing import Optional
from multimodel_solver.multimodal_prompter import build_gemini_payload

class MultimodalWorldModeler:
    def __init__(self, memory_filepath: str = "multimodel_solver/memory_buffer/episodic_memory.json", output_path: str = "LocalSimulator.py"):
        self.memory_filepath = memory_filepath
        self.output_path = output_path
        self.api_key = os.environ.get("GEMINI_API_KEY")
        
        # Use a high-capacity reasoning model capable of multimodal inputs
        self.model_name = "gemini-2.5-pro"
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"

    def generate_simulator(self) -> Optional[str]:
        """
        Uses the multimodal Gemini API to generate a LocalSimulator.py
        based on the visual transitions in episodic memory.
        """
        if not self.api_key:
            print("ERROR: GEMINI_API_KEY environment variable is not set.")
            print("Falling back to mocked simulator for safety.")
            return self._write_mock_simulator()

        print(f"Building multimodal payload from {self.memory_filepath}...")
        try:
            payload = build_gemini_payload(self.memory_filepath)
        except Exception as e:
            print(f"Failed to build Gemini payload: {e}")
            return None

        headers = {
            "Content-Type": "application/json"
        }
        
        url_with_key = f"{self.api_url}?key={self.api_key}"

        print("Sending visual payload to Gemini API...")
        try:
            response = requests.post(url_with_key, headers=headers, json=payload)
            response.raise_for_status()
            response_json = response.json()
            
            # Parse the response text
            candidates = response_json.get("candidates", [])
            if not candidates:
                print("No candidates returned from Gemini.")
                return None
                
            generated_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            
            # Clean markdown code blocks if the LLM wrapped it in ```python ... ```
            cleaned_code = self._extract_python_code(generated_text)
            
            with open(self.output_path, "w") as f:
                f.write(cleaned_code)
                
            print(f"Successfully synthesized and wrote simulator to {self.output_path}")
            return cleaned_code
            
        except requests.exceptions.RequestException as e:
            print(f"API Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response data: {e.response.text}")
            return None

    def correct_simulator(self, original_code: str, error_log: str) -> Optional[str]:
        """Requests Gemini to fix the code based on the execution error log."""
        if not self.api_key:
            print("ERROR: GEMINI_API_KEY environment variable is not set.")
            return None
            
        prompt = f"""You previously wrote a simulator for an ARC-AGI environment. 
However, it failed validation against the episodic memory.

Here is the code you wrote:
```python
{original_code}
```

Here is the error log or state diff from the validation:
{error_log}

Here is the standardized action interface for ARC-AGI-3 games to help you understand what the actions mean:
- `RESET`: Initialize or restart the game/level state
- `ACTION1`: Simple action - semantically mapped to UP
- `ACTION2`: Simple action - semantically mapped to DOWN
- `ACTION3`: Simple action - semantically mapped to LEFT
- `ACTION4`: Simple action - semantically mapped to RIGHT
- `ACTION5`: Simple action - interact, select, rotate, attach/detach, execute, etc.
- `ACTION6`: Complex action requiring x, y coordinates (0-63 range)
- `ACTION7`: Simple action - Undo

CRITICAL RULES FOR CORRECTION:
1. ENFORCE TYPING: The `grid` input is a standard Python `List[List[int]]` (a 2D list of integers). It is NOT a NumPy array, and it is NOT an RGB image. Do not attempt NumPy broadcasting or tuple-based list indexing.
2. SYNTAX REMINDER: To access or modify a cell, you MUST use standard nested list indexing like `grid[y][x]`, NOT `grid[y, x]`.

Please fix the code to resolve this issue and return the corrected Python function `def predict_next_state(grid, action):`.
"""
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2}
        }
        
        url_with_key = f"{self.api_url}?key={self.api_key}"
        try:
            response = requests.post(url_with_key, headers={"Content-Type": "application/json"}, json=payload)
            response.raise_for_status()
            response_json = response.json()
            candidates = response_json.get("candidates", [])
            if not candidates:
                return None
            generated_text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
            cleaned_code = self._extract_python_code(generated_text)
            
            with open(self.output_path, "w") as f:
                f.write(cleaned_code)
                
            return cleaned_code
        except Exception as e:
            print(f"Correction API Request failed: {e}")
            return None

    def _extract_python_code(self, raw_text: str) -> str:
        """Extracts python code from markdown if present."""
        pattern = re.compile(r"```(?:python)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
        match = pattern.search(raw_text)
        if match:
            return match.group(1)
        return raw_text

    def _write_mock_simulator(self) -> str:
        mock_code = '''
def predict_next_state(grid, action):
    # Mocked simulator fallback
    return grid
'''
        with open(self.output_path, "w") as f:
            f.write(mock_code)
        return mock_code
