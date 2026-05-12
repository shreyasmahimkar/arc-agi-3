import time
import sys
import os
import glob
from pathlib import Path
import logging

# Load .env file automatically
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                k, v = line.strip().split('=', 1)
                os.environ[k] = v.strip('"\'')

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

# Add arc_solver to path for arc_agi
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.insert(0, os.path.join(repo_root, 'arc_solver'))
sys.path.insert(0, os.path.join(repo_root, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))

import arc_agi
from arcengine import GameAction, GameState, ActionInput

from my_agent import MyAgent

# Setup logging
log_file = os.path.join(os.path.dirname(__file__), "v9_run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

ARC_COLORS = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
cmap = ListedColormap(ARC_COLORS)

def save_frame_as_image(frame_data, current_level, step_count, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    if not hasattr(frame_data, 'frame') or not frame_data.frame:
        return
    
    grid = frame_data.frame[0]
    plt.figure(figsize=(4,4))
    plt.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    # v7 Grid-Overlay Coordinate Calibration
    plt.grid(color='white', linestyle='-', linewidth=0.5, alpha=0.5)
    plt.xticks(np.arange(-0.5, 64, 5), np.arange(0, 65, 5), rotation=90, fontsize=6)
    plt.yticks(np.arange(-0.5, 64, 5), np.arange(0, 65, 5), fontsize=6)
    filename = os.path.join(output_dir, f"level_{current_level:02d}_step_{step_count:04d}.png")
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Play ARC game with Swarm Solver")
    parser.add_argument("--game", type=str, default="ls20", help="Name of the game to play (e.g. ls20)")
    args = parser.parse_args()

    logger.info("==============================================")
    logger.info("=== Swarm Solver (v4): Play Game using MyAgent ===")
    logger.info("==============================================")
    
    # 4. Initializer to change the game. Start with ls20.
    GAME_NAME = args.game
    
    # Ensure it only gets games from local dir
    env_dir = os.path.join(repo_root, 'arc-prize-2026-arc-agi-3', 'environment_files')
    
    logger.info(f"\n[1] Instantiating arc_agi environment for game: {GAME_NAME} (terminal render mode)...")
    arc = arc_agi.Arcade()
    
    # Make the environment
    env = arc.make(GAME_NAME, render_mode='terminal')
    
    # Force the local_dir to point to the correct arc-prize-2026-arc-agi-3/environment_files folder
    # This fulfills requirement #3
    game_file_pattern = os.path.join(env_dir, GAME_NAME, "**", f"{GAME_NAME}.py")
    matches = glob.glob(game_file_pattern, recursive=True)
    if matches:
        game_local_dir = os.path.dirname(matches[0])
    else:
        game_local_dir = os.path.join(env_dir, GAME_NAME)

    class DummyEnvInfo:
        def __init__(self, d):
            self.local_dir = d
            
    if not hasattr(env, 'environment_info') or not env.environment_info:
        env.environment_info = DummyEnvInfo(game_local_dir)
    else:
        env.environment_info.local_dir = game_local_dir

    logger.info(f"\n[2] Instantiating MyAgent... (Loading from {game_local_dir})")
    # agent will use find_game_source_and_class using arc_env.environment_info.local_dir
    agent = MyAgent(
        card_id="",
        game_id=GAME_NAME,
        agent_name="hybrid_solver",
        ROOT_URL="",
        record=False,
        arc_env=env
    )
    
    # Fix PicklingError in my_agent._fast_deepcopy:
    # my_agent dynamically loads the game module while arc_agi also dynamically loads it.
    # This creates TWO different Ls20 classes in memory, completely breaking python's `pickle`.
    # Since we cannot edit my_agent.py, we monkey-patch its _fast_deepcopy to use `copy.deepcopy`!
    import copy
    import my_agent
    
    def stable_deepcopy(game):
        camera = getattr(game, '_camera', None)
        if camera is not None:
            game._camera = None
        g = copy.deepcopy(game)
        if camera is not None:
            game._camera = camera
            g._camera = camera
        return g
        
    my_agent._fast_deepcopy = stable_deepcopy

    
    logger.info("\n[3] Resetting environment and starting play loop...")
    images_dir = os.path.join(os.path.dirname(__file__), 'images', GAME_NAME)
    try:
        reset_out = env.reset()
        lf = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    except Exception as e:
        logger.info(f"Error resetting environment: {e}")
        return

    frames = [lf]
    agent.append_frame(lf)
    save_frame_as_image(lf, getattr(lf, "levels_completed", 0), 0, images_dir)

    step_count = 0
    level_step_count = 0
    max_steps_per_level = 200
    current_level = getattr(lf, "levels_completed", 0)
    
    while True:
        step_count += 1
        level_step_count += 1
        
        if level_step_count > max_steps_per_level:
            logger.info(f"Max steps ({max_steps_per_level}) reached for Level {current_level}. Terminating.")
            break
        logger.info(f"\n----------------------------------------------")
        logger.info(f"--- STEP {step_count} ---")
        
        # Check if game is over
        if agent.is_done(frames, lf) or (hasattr(lf, 'state') and lf.state in [GameState.GAME_OVER, GameState.WIN]):
            state_val = getattr(lf, 'state', 'UNKNOWN')
            logger.info(f"Game finished! Final state: {state_val}")
            break
            
        logger.info("Agent is analyzing the state and thinking...")
        t0 = time.time()
        
        try:
            action = agent.choose_action(frames, lf)
        except Exception as e:
            logger.info(f"Agent failed to choose action: {e}")
            break
            
        t1 = time.time()
        
        # Extract action details
        act_id = None
        if hasattr(action, 'value'):
            act_id = action.value
        elif hasattr(action, 'id') and hasattr(action.id, 'value'):
            act_id = action.id.value
        else:
            try:
                act_id = int(action.id) if hasattr(action, 'id') else int(action)
            except:
                act_id = -1
                
        act_name = getattr(action, 'name', f"ACTION_{act_id}")
        if hasattr(action, 'id') and hasattr(action.id, 'name'):
            act_name = action.id.name
            
        reasoning = getattr(action, 'reasoning', "No reasoning provided")
        
        logger.info(f">> Agent decided on Action: {act_name} (ID: {act_id}) in {t1-t0:.2f}s")
        logger.info(f">> Reasoning: {reasoning}")
        
        try:
            if act_id == 6:
                # Need to extract x and y for ACTION6 (click)
                x, y = 0, 0
                if hasattr(action, 'data') and isinstance(action.data, dict):
                    x = action.data.get('x', 0)
                    y = action.data.get('y', 0)
                elif hasattr(action, 'get_data') and callable(action.get_data):
                    data = action.get_data()
                    if isinstance(data, dict):
                        x = data.get('x', 0)
                        y = data.get('y', 0)
                else:
                    # In my_agent.py, it parses reasoning: "cnn:c({x},{y})"
                    import re
                    match = re.search(r'c\((\d+),(\d+)\)', reasoning)
                    if match:
                        x, y = int(match.group(1)), int(match.group(2))
                
                logger.info(f"Applying click at X: {x}, Y: {y}")
                step_result = env.step(GameAction.ACTION6, data={'x': int(x), 'y': int(y)})
            else:
                logger.info(f"Applying action: {act_name}")
                if isinstance(action, GameAction):
                    step_result = env.step(action)
                else:
                    step_result = env.step(GameAction.from_id(act_id))
                
        except Exception as e:
            logger.info(f"Error applying step: {e}")
            # Try to recover by passing action.id if it's an object instead of enum
            try:
                if act_id == 6:
                    step_result = env.step(GameAction.ACTION6, data={'x': int(x), 'y': int(y)})
                else:
                    step_result = env.step(GameAction.from_id(act_id))
            except Exception as e2:
                logger.info(f"Recovery failed: {e2}")
                break
            
        # Parse step result back into latest frame
        if isinstance(step_result, tuple):
            if len(step_result) == 5:
                lf, reward, terminated, truncated, step_info = step_result
            elif len(step_result) == 4:
                lf, reward, done, step_info = step_result
            else:
                lf = step_result[0]
        else:
            lf = step_result
            
        frames.append(lf)
        agent.append_frame(lf)
        
        new_level = getattr(lf, "levels_completed", 0)
        if new_level != current_level:
            logger.info(f"Advanced to Level {new_level}!")
            current_level = new_level
            level_step_count = 0
            
        state_val = getattr(lf, 'state', 'UNKNOWN')
        logger.info(f"Current Level: {current_level}, State: {state_val}, Level Steps: {level_step_count}")
        
        save_frame_as_image(lf, current_level, step_count, images_dir)
        
        time.sleep(0.5)
        
    logger.info("\n==============================================")
    logger.info("=== Execution Complete ===")
    try:
        logger.info("\nScorecard Efficiency:")
        logger.info(arc.get_scorecard())
    except Exception:
        pass

if __name__ == "__main__":
    main()
