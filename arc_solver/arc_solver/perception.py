import numpy as np
from scipy.ndimage import label
from typing import List, Dict, Any, Tuple

def parse_state(state: Any) -> Tuple[np.ndarray, List[str]]:
    """
    Parses the ARC environment state dictionary.
    Isolates the grid as a numpy array and extracts available_actions.
    Prevents MCTS float conversion errors by ensuring proper dictionary parsing.
    """
    if isinstance(state, dict):
        grid_data = state.get("grid", [])
        grid = np.array(grid_data) if grid_data else np.array([])
        available_actions = state.get("available_actions", [])
    else:
        grid = np.array(state) if state is not None else np.array([])
        available_actions = []
        
    return grid, available_actions

def get_active_coordinates(grid: np.ndarray, background_color: int = 0) -> List[Tuple[int, int]]:
    """
    Returns a list of (x, y) coordinates for all non-background pixels.
    Used for Coordinate Pruning.
    """
    if grid.size == 0:
        return []
    coords = np.argwhere(grid != background_color)
    return [(int(c), int(r)) for r, c in coords]  # Returning (x, y)

def find_objects(grid: List[List[int]], background_color: int = 0) -> List[Dict[str, Any]]:
    """
    Finds contiguous objects of the same color in a 2D grid using scipy.ndimage.label.
    Objects are defined as orthogonally or diagonally connected pixels of the SAME non-background color.
    """
    grid_np = np.array(grid)
    objects = []
    
    unique_colors = np.unique(grid_np)
    for color in unique_colors:
        if color == background_color:
            continue
            
        mask = (grid_np == color).astype(int)
        structure = np.ones((3, 3), dtype=int)
        labeled_array, num_features = label(mask, structure=structure)
        
        for feature_id in range(1, num_features + 1):
            coords = np.argwhere(labeled_array == feature_id)
            objects.append({
                'color': int(color),
                'coords': [(int(r), int(c)) for r, c in coords]
            })
            
    return objects
