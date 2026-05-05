import torch
import torch.nn as nn
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, state_dim=256+32, action_dim=5, hidden_dim=256):
        super().__init__()
        
        # Actor network: takes state (h_t, z_t) -> policy logits
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic network: takes state (h_t, z_t) -> state value (symlog)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def get_action_dist(self, state):
        """Returns a Categorical distribution over actions."""
        logits = self.actor(state)
        return Categorical(logits=logits)
        
    def get_value(self, state):
        """Returns the predicted value (in symlog space)."""
        return self.critic(state)
