import json
import os
from typing import Any, Dict, List

class EpisodicMemory:
    def __init__(self, filepath: str = "episodic_memory.json"):
        self.filepath = filepath
        self.memory: List[Dict[str, Any]] = []
        self._load_if_exists()
        
    def _load_if_exists(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.memory = json.load(f)
            except json.JSONDecodeError:
                self.memory = []
                
    def log(self, state: Any, action: Any, next_state: Any):
        """Logs a transition to memory."""
        self.memory.append({
            "state": state,
            "action": action,
            "next_state": next_state
        })
        
    def save(self):
        """Saves memory to disk."""
        with open(self.filepath, 'w') as f:
            json.dump(self.memory, f, indent=4)
            
    def clear(self):
        """Clears current memory and deletes the file."""
        self.memory = []
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
