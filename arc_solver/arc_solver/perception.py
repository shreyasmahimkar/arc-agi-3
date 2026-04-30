import numpy as np
from scipy.ndimage import label
from typing import List, Dict, Any

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
