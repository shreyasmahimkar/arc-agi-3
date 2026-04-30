import gymnasium as gym
from gymnasium import spaces
import numpy as np
import arc_agi
from arcengine import GameAction, GameState

class LS20DreamerEnv(gym.Env):
    """
    A Gymnasium wrapper for the ARC-AGI-3 ls20 environment.
    Includes Neural Map memory logic and reward shaping.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        self.render_mode = render_mode
        self.arc = arc_agi.Arcade()
        
    # Use the following for offline mode in kaggle notebook while extracting
#     import os
# self.arc = arc_agi.Arcade(
#     environments_dir="/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files",
#     operation_mode=arc_agi.OperationMode.OFFLINE
# )
        
        # We start with the arc environment
        # we don't pass render_mode to arc.make because we will handle rendering of memory if needed
        self.env = self.arc.make("ls20")
        
        self.grid_size = 64
        self.num_colors = 16
        
        # Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=ACTION1 (or ACTION5 depending on mapping)
        # In arcengine, actions are ACTION1, ACTION2, ACTION3, ACTION4, ACTION5, ACTION6
        # Typically 1-4 are movements, 5 is action, 6 is coordinate. 
        # We'll map discrete 0-4 to ACTION1-ACTION5.
        self.action_space = spaces.Discrete(5)
        self.action_mapping = [
            GameAction.ACTION1,
            GameAction.ACTION2,
            GameAction.ACTION3,
            GameAction.ACTION4,
            GameAction.ACTION5
        ]

        # Observation Space Dictionary
        # visible_frame: The raw frame (0 is assumed black/unseen)
        # memory_map: The aggregated memory
        # available_actions: One-hot vector of available actions
        self.observation_space = spaces.Dict({
            "visible_frame": spaces.Box(low=0, high=15, shape=(self.grid_size, self.grid_size), dtype=np.int8),
            "memory_map": spaces.Box(low=0, high=15, shape=(self.grid_size, self.grid_size), dtype=np.int8),
            "available_actions": spaces.MultiBinary(5)
        })

        # Internal state
        self.neural_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.lives = 3
        self.current_level = 0
        self.last_frame_state = None
        self.last_visible_frame = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)

    def _get_obs(self, frame_data):
        # Extract 64x64 frame
        frame = np.array(frame_data.frame[-1], dtype=np.int8)
        
        # Neural Map Aggregation
        # Assume 0 is the "dark/unseen" color. If 0 is meaningful, we might need a different heuristic,
        # but typically in ARC 0 is black background.
        visible_mask = (frame != 0)
        self.neural_map[visible_mask] = frame[visible_mask]

        # Available actions mask
        avail_mask = np.zeros(5, dtype=np.int8)
        for a in frame_data.available_actions:
            # a is 1 to 6
            idx = a - 1
            if 0 <= idx < 5:
                avail_mask[idx] = 1

        return {
            "visible_frame": frame,
            "memory_map": self.neural_map.copy(),
            "available_actions": avail_mask
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the underlying env
        self.env = self.arc.make("ls20")
        frame_data = self.env.reset()
        
        # Reset internal tracking
        self.neural_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        self.lives = 3
        self.current_level = frame_data.levels_completed
        self.last_frame_state = frame_data.state
        
        obs = self._get_obs(frame_data)
        self.last_visible_frame = obs["visible_frame"].copy()
        
        info = {"lives": self.lives, "level": self.current_level}
        
        return obs, info

    def step(self, action_idx):
        action = self.action_mapping[action_idx]
        frame_data = self.env.step(action)
        
        if frame_data is None:
            # End of everything
            return self._get_empty_obs(), 0, True, False, {}

        # 1. Base Step Penalty
        reward = -0.01
        terminated = False
        
        # 2. Level Check
        new_level = frame_data.levels_completed
        if new_level > self.current_level:
            reward += 10.0
            self.current_level = new_level
            # Also clear the memory map for the new level
            self.neural_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            
        # 3. Game Over / Full Reset Check (Fuel ran out -> lost a life)
        if frame_data.full_reset or frame_data.state == GameState.GAME_OVER:
            # The agent died (ran out of fuel or hit a trap)
            self.lives -= 1
            reward -= 10.0
            # Reset memory map because it respawns or resets level
            self.neural_map = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
            
            if self.lives <= 0:
                terminated = True
            elif frame_data.state == GameState.GAME_OVER:
                # Intercept the native GAME_OVER and use one of our lives
                frame_data = self.env.reset()
                self.current_level = frame_data.levels_completed
                
        # 4. Win Check
        if frame_data.state == GameState.WIN:
            reward += 10.0
            terminated = True
            
        self.last_frame_state = frame_data.state
        
        obs = self._get_obs(frame_data)
        
        # 5. Intrinsic Motivation: Reward for changing the frame (moving instead of hitting walls)
        if not np.array_equal(self.last_visible_frame, obs["visible_frame"]):
            reward += 0.1
        else:
            reward -= 0.1  # Extra penalty for standing still / hitting a wall
            
        self.last_visible_frame = obs["visible_frame"].copy()
        
        info = {"lives": self.lives, "level": self.current_level}
        
        return obs, reward, terminated, False, info

    def _get_empty_obs(self):
        return {
            "visible_frame": np.zeros((self.grid_size, self.grid_size), dtype=np.int8),
            "memory_map": np.zeros((self.grid_size, self.grid_size), dtype=np.int8),
            "available_actions": np.zeros(5, dtype=np.int8)
        }

    def render(self):
        if self.render_mode == "human":
            pass # We will handle rendering externally in play_and_learn or implement PyGame here

    def close(self):
        pass

