import json
import os
import random
import uuid
import numpy as np
from typing import Any, Dict, List, TypedDict, Optional
from PIL import Image

# Import perception from original arc_solver
from arc_solver.perception import parse_state, get_active_coordinates

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
            d = obj.model_dump(mode="json")
            # Ensure the grid is serialized if present in frame
            if hasattr(obj, 'frame') and isinstance(obj.frame, list) and len(obj.frame) > 0:
                frame = obj.frame[0]
                if isinstance(frame, np.ndarray):
                    d['grid'] = frame.tolist()
                elif isinstance(frame, list):
                    d['grid'] = frame
            return d
        elif hasattr(obj, 'value'):
            return obj.value
        # Attempt to handle generic numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

ARC_COLORS = {
    0: (0, 0, 0),       # Black
    1: (0, 116, 217),   # Blue
    2: (255, 65, 54),   # Red
    3: (46, 204, 64),   # Green
    4: (255, 220, 0),   # Yellow
    5: (170, 170, 170), # Grey
    6: (240, 18, 190),  # Fuchsia
    7: (255, 133, 27),  # Orange
    8: (127, 219, 255), # Teal
    9: (133, 20, 75),   # Maroon
}

def render_grid_to_image(grid_data: Any, filepath: str, cell_size: int = 20):
    grid = []
    if isinstance(grid_data, dict):
        grid = grid_data.get("grid", [])
    elif isinstance(grid_data, (list, np.ndarray)):
        grid = grid_data
    elif hasattr(grid_data, "grid"):
        grid = getattr(grid_data, "grid")
    elif hasattr(grid_data, "frame") and isinstance(getattr(grid_data, "frame"), list) and len(getattr(grid_data, "frame")) > 0:
        grid = getattr(grid_data, "frame")[0]

    if grid is None or len(grid) == 0:
        img = Image.new('RGB', (cell_size, cell_size), color=(0,0,0))
        img.save(filepath)
        return
        
    try:
        height = len(grid)
        width = len(grid[0])
    except (TypeError, IndexError):
        img = Image.new('RGB', (cell_size, cell_size), color=(0,0,0))
        img.save(filepath)
        return

    img = Image.new('RGB', (width * cell_size, height * cell_size), color=(0,0,0))
    pixels = img.load()
    
    for y in range(height):
        for x in range(width):
            try:
                val = int(grid[y][x])
            except (ValueError, TypeError):
                val = 0
            color = ARC_COLORS.get(val, (0,0,0))
            for py in range(cell_size):
                for px in range(cell_size):
                    pixels[x * cell_size + px, y * cell_size + py] = color
                    
    img.save(filepath)

class EpisodicMemory:
    def __init__(self, filepath: str = "multimodel_solver/memory_buffer/episodic_memory.json", image_dir: str = "multimodel_solver/memory_buffer"):
        self.filepath = filepath
        self.image_dir = image_dir
        self.memory: List[Dict[str, Any]] = []
        os.makedirs(self.image_dir, exist_ok=True)
        self._load_if_exists()
        
    def _load_if_exists(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, 'r') as f:
                    self.memory = json.load(f)
            except json.JSONDecodeError:
                self.memory = []
                
    def log(self, state: Any, action: Any, next_state: Any):
        """Logs a transition to memory, generating multimodal visual logging."""
        state_id = str(uuid.uuid4())
        next_state_id = str(uuid.uuid4())
        
        state_img_path = os.path.join(self.image_dir, f"{state_id}.png")
        next_state_img_path = os.path.join(self.image_dir, f"{next_state_id}.png")
        
        # Render states to images
        render_grid_to_image(state, state_img_path)
        render_grid_to_image(next_state, next_state_img_path)

        self.memory.append({
            "state_image": state_img_path,
            "action": action,
            "next_state_image": next_state_img_path,
            "state_raw": state,
            "next_state_raw": next_state
        })
        
    def save(self):
        """Saves memory to disk."""
        with open(self.filepath, 'w') as f:
            json.dump(self.memory, f, indent=4, cls=CustomJSONEncoder)
            
    def clear(self):
        """Clears current memory and deletes the JSON file and images."""
        for m in self.memory:
            if "state_image" in m and os.path.exists(m["state_image"]):
                os.remove(m["state_image"])
            if "next_state_image" in m and os.path.exists(m["next_state_image"]):
                os.remove(m["next_state_image"])
        
        self.memory = []
        if os.path.exists(self.filepath):
            os.remove(self.filepath)
