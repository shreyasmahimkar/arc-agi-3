import numpy as np
from typing import Any, Tuple, Dict, List

class ARCGymWrapper:
    """
    Gym-style RL environment wrapper for ARC-AGI-3.
    Supports sequential state observations, continuous representations, and temporal feedback.
    """
    def __init__(self, env: Any):
        self.env = env
        self.current_state = None
        self.history = []
        
    def reset(self) -> Any:
        try:
            self.current_state = self.env.reset()
        except Exception:
            # Fallback if raw env doesn't support reset perfectly
            self.current_state = {"grid": [], "available_actions": ["ACTION1"]}
        self.history = [self.current_state]
        return self.current_state
        
    def step(self, action: Any) -> Tuple[Any, float, bool, Dict]:
        """
        Executes action and handles ARC's temporal feedback loops.
        Returns standard Gym tuple: (observation, reward, done, info)
        """
        try:
            if isinstance(action, dict) and "x" in action and "y" in action:
                step_result = self.env.step(action["action"], x=action["x"], y=action["y"])
            else:
                step_result = self.env.step(action)
                
            # Unpack dynamically based on engine version
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    obs, reward, term, trunc, info = step_result
                    done = term or trunc
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, done, info = step_result[0], 0.0, False, {}
            else:
                obs = step_result
                reward = 0.0
                done = False
                if hasattr(obs, 'state') and getattr(obs, 'state') == "FINISHED":
                    done = True
                info = {}
                
            self.current_state = obs
            self.history.append(obs)
            return obs, float(reward), done, info
            
        except Exception as e:
            # Return defensive failure state
            return self.current_state, -1.0, True, {"error": str(e)}

class ActiveInferenceAgent:
    """
    Active Inference Agent utilizing a generative world model.
    Updates internal state via Hamiltonian Monte Carlo and selects actions
    that minimize Expected Free Energy (Surprise).
    """
    def __init__(self, state_dim: int, action_dim: int):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.internal_mu = np.zeros(state_dim)
        self.internal_sigma = np.eye(state_dim)
        
    def _hamiltonian_monte_carlo_update(self, observation: np.ndarray, steps: int = 5, step_size: float = 0.01):
        """
        Approximates posterior belief updating using a simplified HMC approach.
        """
        q = self.internal_mu.copy()
        p = np.random.randn(*q.shape)
        
        for _ in range(steps):
            # Half step for momentum
            p = p - (step_size / 2.0) * (q - observation)  # Simple gradient of Gaussian prior/likelihood
            # Full step for position
            q = q + step_size * p
            # Half step for momentum
            p = p - (step_size / 2.0) * (q - observation)
            
        # Update internal belief state
        self.internal_mu = q
        
    def _calculate_expected_free_energy(self, state_belief: np.ndarray, candidate_action: np.ndarray) -> float:
        """
        Calculates Expected Free Energy (G) = Epistemic Value (Ambiguity) + Pragmatic Value (Risk/Surprise).
        For simplicity in this foundational class, we use a basic divergence metric.
        """
        # Predicted next state based on a dummy linear transition model
        predicted_state = state_belief + candidate_action[:len(state_belief)] * 0.1
        
        # Pragmatic value: distance to a "preferred" zero-surprise state
        surprise = np.sum(predicted_state**2)
        
        # Epistemic value: uncertainty reduction (approximated)
        ambiguity = np.trace(self.internal_sigma)
        
        return surprise + ambiguity
        
    def select_action(self, observation: Any, available_actions: List[Any]) -> Any:
        """
        Updates internal world model via HMC, then evaluates candidate actions 
        to minimize Expected Free Energy.
        """
        # 1. Perceive and Update (Inference)
        # Flatten observation into a simplistic vector for foundational logic
        if isinstance(observation, dict) and 'grid' in observation:
            obs_vec = np.array(observation['grid']).flatten()
            if len(obs_vec) > self.state_dim:
                obs_vec = obs_vec[:self.state_dim]
            elif len(obs_vec) < self.state_dim:
                obs_vec = np.pad(obs_vec, (0, self.state_dim - len(obs_vec)))
        else:
            obs_vec = np.zeros(self.state_dim)
            
        self._hamiltonian_monte_carlo_update(obs_vec)
        
        # 2. Plan (Active Inference)
        best_action = None
        min_efe = float('inf')
        
        for action in available_actions:
            # Create a dummy continuous representation of the action
            action_vec = np.random.randn(self.action_dim) 
            
            efe = self._calculate_expected_free_energy(self.internal_mu, action_vec)
            if efe < min_efe:
                min_efe = efe
                best_action = action
                
        return best_action
