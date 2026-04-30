import torch
import torch.nn.functional as F
from models.dreamer.symlog import symexp

class LatentPlanner:
    def __init__(self, horizon=5, num_samples=100, action_dim=5, cem_iters=3, elite_frac=0.1):
        self.horizon = horizon
        self.num_samples = num_samples
        self.action_dim = action_dim
        self.cem_iters = cem_iters
        self.num_elites = max(1, int(num_samples * elite_frac))

    def plan(self, world_model, actor_critic, state, device):
        deter, stoch = state
        
        # We maintain a probability distribution over actions for each step in the horizon
        # Initial distribution is uniform over the 5 actions
        action_probs = torch.ones((self.horizon, self.action_dim), device=device) / self.action_dim
        
        best_action_ever = None
        best_return_ever = -float('inf')
        
        for i in range(self.cem_iters):
            # Sample action sequences for the whole horizon
            actions_dist = torch.distributions.Categorical(probs=action_probs)
            actions = actions_dist.sample((self.num_samples,)).T # Shape: (horizon, num_samples)
            
            # Mix in Actor policy for guidance (first 20 samples on first iteration)
            if i == 0:
                curr_deter = deter.repeat(20, 1)
                curr_stoch = stoch.repeat(20, 1)
                curr_state = (curr_deter, curr_stoch)
                for t in range(self.horizon):
                    full_state = torch.cat([curr_state[0], curr_state[1]], dim=-1)
                    a = actor_critic.get_action_dist(full_state).sample()
                    actions[t, :20] = a
                    curr_state = world_model.rssm.imagine_step(curr_state, F.one_hot(a, num_classes=self.action_dim).float())

            # Parallel Rollout
            curr_deter = deter.repeat(self.num_samples, 1)
            curr_stoch = stoch.repeat(self.num_samples, 1)
            curr_state = (curr_deter, curr_stoch)
            
            cumulative_rewards = torch.zeros(self.num_samples, device=device)
            first_actions = actions[0]
            
            for t in range(self.horizon):
                action_one_hot = F.one_hot(actions[t].long(), num_classes=self.action_dim).float()
                curr_state = world_model.rssm.imagine_step(curr_state, action_one_hot)
                
                next_full_state = torch.cat([curr_state[0], curr_state[1]], dim=-1)
                reward_pred = world_model.rssm.reward_net(next_full_state).squeeze(-1)
                cumulative_rewards += symexp(reward_pred)
                
            # Critic evaluation at end of horizon
            final_full_state = torch.cat([curr_state[0], curr_state[1]], dim=-1)
            value_pred = actor_critic.get_value(final_full_state).squeeze(-1)
            total_returns = cumulative_rewards + (0.99 ** self.horizon) * symexp(value_pred)
            
            # Find elites (top performers)
            elite_returns, elite_indices = torch.topk(total_returns, self.num_elites)
            elite_actions = actions[:, elite_indices] # shape (horizon, num_elites)
            
            # Update overall best
            if elite_returns[0].item() > best_return_ever:
                best_return_ever = elite_returns[0].item()
                best_action_ever = elite_actions[0, 0]
                
            # Refit distribution based on elites
            new_action_probs = torch.zeros_like(action_probs)
            for t in range(self.horizon):
                counts = torch.bincount(elite_actions[t].long(), minlength=self.action_dim).float()
                new_action_probs[t] = counts / self.num_elites
                
            # Smooth update
            action_probs = 0.8 * new_action_probs + 0.2 * action_probs
            
        return best_action_ever, best_return_ever
