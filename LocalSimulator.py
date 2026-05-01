import numpy as np

def predict_next_state(grid, action):
    """
    Predicts the next state of the grid based on the observed rules.

    The environment operates in two phases:
    1. Charging Phase: A progress bar fills up. When full, a blocker object is removed.
    2. Action Phase: An elevator object moves upwards.

    Args:
        grid (np.ndarray): A 3D numpy array representing the current state (H, W, C).
        action (dict): The action taken (unused in this deterministic environment).

    Returns:
        np.ndarray: The grid representing the next state.
    """
    # Define colors using their RGB values from the image
    YELLOW = np.array([255, 217, 0], dtype=np.uint8)
    GREEN = np.array([58, 168, 58], dtype=np.uint8)
    BLACK = np.array([0, 0, 0], dtype=np.uint8)
    PURPLE = np.array([135, 38, 87], dtype=np.uint8)

    # Define key coordinates and dimensions based on the grid
    BLOCKER_Y, BLOCKER_X = 26, 24
    BLOCKER_H, BLOCKER_W = 2, 4
    
    BAR_Y = 44
    BAR_X_START = 12
    BAR_X_END = 43 # inclusive
    SEGMENT_WIDTH = 4

    ELEVATOR_W = 4
    ELEVATOR_H = 2
    ELEVATOR_SHAFT_X = 26

    # Create a copy of the grid to modify
    next_grid = np.copy(grid)

    # --- Determine the current phase by checking for the blocker ---
    # We check the color of the top-left pixel where the blocker should be.
    is_blocker_present = np.array_equal(grid[BLOCKER_Y, BLOCKER_X], BLACK)

    if is_blocker_present:
        # --- Phase 1: Charging ---
        # The elevator does not move in this phase.

        # Count the number of green pixels in the progress bar
        num_green_pixels = np.sum(np.all(grid[BAR_Y, BAR_X_START:BAR_X_END+1] == GREEN, axis=1))
        num_segments = num_green_pixels // SEGMENT_WIDTH

        if num_segments < 2:
            # Add a new green segment to the bar (0->1 or 1->2)
            start_x = BAR_X_START + num_segments * SEGMENT_WIDTH
            end_x = start_x + SEGMENT_WIDTH
            next_grid[BAR_Y, start_x:end_x] = GREEN
        else: # num_segments is 2
            # Transition to Phase 2: Reset bar and remove blocker
            # 1. Reset the second segment of the bar to black
            start_x = BAR_X_START + SEGMENT_WIDTH
            end_x = start_x + SEGMENT_WIDTH
            next_grid[BAR_Y, start_x:end_x] = BLACK
            
            # 2. Remove the blocker by filling its area with the yellow background
            next_grid[BLOCKER_Y:BLOCKER_Y + BLOCKER_H, BLOCKER_X:BLOCKER_X + BLOCKER_W] = YELLOW
    
    else:
        # --- Phase 2: Action ---
        # The progress bar does not change. The elevator moves up.

        # 1. Find the current position of the elevator
        elevator_y = -1
        # Scan the known vertical shaft for the elevator's black top
        for y in range(grid.shape[0] - ELEVATOR_H + 1):
            pixel_slice = grid[y, ELEVATOR_SHAFT_X : ELEVATOR_SHAFT_X + ELEVATOR_W]
            if np.all(np.all(pixel_slice == BLACK, axis=1)):
                elevator_y = y
                break
        
        if elevator_y != -1:
            # 2. Erase the elevator from its current position
            # The background it moves over is green
            next_grid[elevator_y : elevator_y + ELEVATOR_H, 
                      ELEVATOR_SHAFT_X : ELEVATOR_SHAFT_X + ELEVATOR_W] = GREEN

            # 3. Draw the elevator at its new position (one pixel up)
            new_y = elevator_y - 1
            # Draw top black part
            next_grid[new_y, ELEVATOR_SHAFT_X : ELEVATOR_SHAFT_X + ELEVATOR_W] = BLACK
            # Draw bottom purple part
            next_grid[new_y + 1, ELEVATOR_SHAFT_X : ELEVATOR_SHAFT_X + ELEVATOR_W] = PURPLE

    return next_grid