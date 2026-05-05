import base64
import json
import os
from typing import List, Dict, Any

def encode_image(image_path: str) -> str:
    """Encodes an image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def build_gemini_payload(memory_filepath: str = "multimodel_solver/memory_buffer/episodic_memory.json") -> Dict[str, Any]:
    """
    Constructs a payload for the Gemini API using images from the episodic memory.
    The payload includes the sequence of (image_state_t, action, image_state_t+1) 
    as visual tokens, accompanied by a text prompt.
    """
    if not os.path.exists(memory_filepath):
        raise FileNotFoundError(f"Memory file not found at {memory_filepath}")
        
    with open(memory_filepath, 'r') as f:
        memory = json.load(f)
        
    if not memory:
        raise ValueError("Episodic memory is empty.")
        
    contents = []
    
    for transition in memory:
        state_img_path = transition.get("state_image")
        next_state_img_path = transition.get("next_state_image")
        action = transition.get("action")
        
        if not state_img_path or not os.path.exists(state_img_path):
            continue
        if not next_state_img_path or not os.path.exists(next_state_img_path):
            continue
            
        state_b64 = encode_image(state_img_path)
        next_state_b64 = encode_image(next_state_img_path)
        
        contents.append({
            "role": "user",
            "parts": [
                {"text": "State t:"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": state_b64
                    }
                },
                {"text": f"Action: {json.dumps(action)}"},
                {"text": "State t+1:"},
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": next_state_b64
                    }
                }
            ]
        })
        
    text_prompt = """Analyze these visual transitions. Write a Python function `def predict_next_state(grid, action):` that models the latent physics and rules of this environment.

Here is the standardized action interface for ARC-AGI-3 games to help you understand what the actions mean:
- `RESET`: Initialize or restart the game/level state
- `ACTION1`: Simple action - semantically mapped to UP
- `ACTION2`: Simple action - semantically mapped to DOWN
- `ACTION3`: Simple action - semantically mapped to LEFT
- `ACTION4`: Simple action - semantically mapped to RIGHT
- `ACTION5`: Simple action - interact, select, rotate, attach/detach, execute, etc.
- `ACTION6`: Complex action requiring x, y coordinates (0-63 range)
- `ACTION7`: Simple action - Undo

CRITICAL RULES:
1. ENFORCE TYPING: The `grid` input is a standard Python `List[List[int]]` (a 2D list of integers). It is NOT a NumPy array, and it is NOT an RGB image. Do not attempt NumPy broadcasting or tuple-based list indexing.
2. SYNTAX REMINDER: To access or modify a cell, you MUST use standard nested list indexing like `grid[y][x]`, NOT `grid[y, x]`.
3. ACTION PARAMETER: The `action` parameter is a simple string containing the action name (e.g., "ACTION1", "ACTION2"). It is NOT a dictionary.
"""
    
    # Append the final prompt
    contents.append({
        "role": "user",
        "parts": [
            {"text": text_prompt}
        ]
    })
    
    payload = {
        "contents": contents,
        "generationConfig": {
            "temperature": 0.2
        }
    }
    
    return payload

if __name__ == "__main__":
    # Example usage
    try:
        payload = build_gemini_payload()
        print("Successfully built multimodal payload.")
        print("Total tokens in payload (approx):", len(json.dumps(payload)) // 4)
    except Exception as e:
        print(f"Error: {e}")
