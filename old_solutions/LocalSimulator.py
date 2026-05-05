import copy

def predict_next_state(grid: list[list[int]], action: str) -> list[list[int]]:
    """
    Predicts the next state of the game grid based on the given action.

    The game's state changes are driven by ACTION1, which affects two elements:
    1. A progress bar at the bottom fills with green, one pixel per action.
    2. A "mouth" on the central creature animates in a 3-state cycle (UP, UP, DOWN)
       that is synchronized with the progress bar's advancement.

    Args:
        grid: A 2D list of integers representing the current game state.
        action: A string representing the action taken by the agent.

    Returns:
        A 2D list of integers representing the predicted next game state.
    """
    if action != "ACTION1":
        # No other actions are observed to have an effect, so return the grid unchanged.
        return grid

    next_grid = copy.deepcopy(grid)

    # Define constants based on visual analysis
    GREEN = 8
    BLACK = 0
    PURPLE = 6
    
    PROGRESS_BAR_Y = 60
    PROGRESS_BAR_START_X = 12
    
    MOUTH_X = 31
    MOUTH_TOP_Y = 47
    MOUTH_BOTTOM_Y = 48

    # --- Part 1: Determine current progress ---
    # Count the number of green pixels on the progress bar to find the current state.
    current_progress = 0
    # The bar appears to run from x=12 to x=59.
    for x in range(PROGRESS_BAR_START_X, 60):
        if next_grid[PROGRESS_BAR_Y][x] == GREEN:
            current_progress += 1
        else:
            # Stop counting at the first non-green pixel.
            break
    
    # The new state count for the animation is the progress after this action.
    new_state_count = current_progress + 1

    # --- Part 2: Update the progress bar ---
    # Add one more green pixel to the bar.
    next_pixel_x = PROGRESS_BAR_START_X + current_progress
    if next_pixel_x < 60:  # Ensure we stay within the bar's bounds.
        next_grid[PROGRESS_BAR_Y][next_pixel_x] = GREEN
        
    # --- Part 3: Update the mouth animation ---
    # The animation cycle is UP, UP, DOWN. The DOWN state occurs when the
    # new_state_count is a multiple of 3.
    if new_state_count % 3 == 0:
        # Set mouth to DOWN state (black pixel above purple).
        next_grid[MOUTH_TOP_Y][MOUTH_X] = BLACK
        next_grid[MOUTH_BOTTOM_Y][MOUTH_X] = PURPLE
    else:  # This covers when new_state_count % 3 is 1 or 2.
        # Set mouth to UP state (purple pixel moves up, green fills below).
        next_grid[MOUTH_TOP_Y][MOUTH_X] = PURPLE
        next_grid[MOUTH_BOTTOM_Y][MOUTH_X] = GREEN

    return next_grid