import json
import os
import random
from typing import Any, Dict, List, TypedDict, Optional
from arc_solver.perception import parse_state, get_active_coordinates

# Strict Action Schema Ruleset
# ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right, 
# ACTION5=Interact/Rotate, ACTION6=Click (requires x,y coords), 
# ACTION7=Undo, RESET=Restart. Raw integers are strictly forbidden.
class ActionDict(TypedDict, total=False):
    action: str
    x: Optional[int]
    y: Optional[int]

class Explorer:
    def __init__(self, memory: 'EpisodicMemory'):
        self.memory = memory
        
    def sample_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reads available_actions from state and strictly samples from them.
        Implements Coordinate Pruning for ACTION6.
        """
        grid, available_actions = parse_state(state)
        
        if not available_actions:
            available_actions = ["ACTION1"]
            
        if "ACTION5" in available_actions and random.random() < 0.8:
            # Prioritize ACTION5 to observe rotation mechanics (80% chance if available)
            chosen_action_str = "ACTION5"
        else:
            chosen_action_str = random.choice(available_actions)
            
        action_dict = {"action": chosen_action_str}
        
        if chosen_action_str == "ACTION6":
            # Coordinate Pruning: Only sample active (non-zero) pixels
            active_coords = get_active_coordinates(grid)
            if active_coords:
                x, y = random.choice(active_coords)
                action_dict["x"] = x
                action_dict["y"] = y
            else:
                action_dict["x"] = 0
                action_dict["y"] = 0
                
        return action_dict

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump(mode="json")
        elif hasattr(obj, 'value'):
            return obj.value
        return super().default(obj)

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
            json.dump(self.memory, f, indent=4, cls=CustomJSONEncoder)
            
    def clear(self):
        """Clears current memory and deletes the file."""
        self.memory = []
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
