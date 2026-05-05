import os
# --- CRITICAL: Set these BEFORE any other imports ---
# This ensures the arc_agi library sees the path during initialization
os.environ["ARC_GAMES_DIR"] = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files"
os.environ["OPERATION_MODE"] = "offline"
os.environ["ARC_API_KEY"] = "test-key-123"

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import random
import matplotlib.pyplot as plt

from dreamerv3.ls20_dreamer_env import LS20DreamerEnv
from models.dreamer.rssm import WorldModel
from models.dreamer.actor_critic import ActorCritic
from models.dreamer.symlog import symlog, symexp
from models.dreamer.planner import LatentPlanner

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, obs, action, reward, done):
        self.buffer.append((obs, action, reward, done))
        
    def sample(self, batch_size, sequence_length):
        valid_indices = [i for i in range(len(self.buffer) - sequence_length) 
                         if not any(self.buffer[i+j][3] for j in range(sequence_length-1))]
        if len(valid_indices) < batch_size:
            indices = np.random.choice(len(self.buffer) - sequence_length, batch_size)
        else:
            indices = np.random.choice(valid_indices, batch_size)
        
        obs_seq, action_seq, reward_seq = [], [], []
        for i in indices:
            o_seq, a_seq, r_seq = [], [], []
            for j in range(sequence_length):
                o, a, r, d = self.buffer[i+j]
                o_seq.append(o)
                a_seq.append(a)
                r_seq.append(r)
            obs_seq.append(o_seq)
            action_seq.append(a_seq)
            reward_seq.append(r_seq)
            
        return obs_seq, action_seq, reward_seq
        
    def __len__(self):
        return len(self.buffer)

def preprocess_obs(obs, device):
    """Stack visible_frame and memory_map into a 2-channel tensor."""
    vf = torch.tensor(obs["visible_frame"], dtype=torch.float32)
    mm = torch.tensor(obs["memory_map"], dtype=torch.float32)
    # Scale 0-15 to 0-1
    vf = vf / 15.0
    mm = mm / 15.0
    stacked = torch.stack([vf, mm], dim=0).unsqueeze(0).to(device) # (1, 2, 64, 64)
    return stacked

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render_memory", action="store_true", help="Visualize the Neural Map memory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    env = LS20DreamerEnv()
    
    world_model = WorldModel(action_dim=5).to(device)
    actor_critic = ActorCritic(state_dim=256+32, action_dim=5, hidden_dim=256).to(device)
    
    # Auto-resume from latest checkpoint if available
    import glob
    wm_checkpoints = glob.glob("checkpoints/world_model_step_*.pth")
    if wm_checkpoints:
        wm_checkpoints.sort(key=lambda x: int(x.split('_step_')[1].split('.pth')[0]))
        latest_wm = wm_checkpoints[-1]
        latest_ac = latest_wm.replace("world_model", "actor_critic")
        if os.path.exists(latest_ac):
            print(f"🔄 Resuming from checkpoint: {latest_wm}")
            world_model.load_state_dict(torch.load(latest_wm, map_location=device))
            actor_critic.load_state_dict(torch.load(latest_ac, map_location=device))
            
    target_actor_critic = ActorCritic(state_dim=256+32, action_dim=5, hidden_dim=256).to(device)
    target_actor_critic.load_state_dict(actor_critic.state_dict())
    target_actor_critic.eval()
    
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-4)
    ac_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-5)
    
    buffer = ReplayBuffer(capacity=50000)
    
    obs, info = env.reset()
    
    # Initialize RSSM state
    state = world_model.rssm.initial(1, device)
    
    if args.render_memory:
        plt.ion()
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        img_vis = axes[0].imshow(obs["visible_frame"], vmin=0, vmax=15, cmap="tab20")
        axes[0].set_title("Visible Frame (Arc-AGI)")
        img_mem = axes[1].imshow(obs["memory_map"], vmin=0, vmax=15, cmap="tab20")
        axes[1].set_title("Neural Map (Memory)")
        plt.show()

    steps = 0
    episode_reward = 0
    current_lives = info.get("lives", 3)
    current_level = info.get("level", 0)
    
    # Initialize Test-Time Planner
    planner = LatentPlanner(horizon=5, num_samples=100)
    
    print("Starting online Play & Learn loop...")
    
    while True:
        steps += 1
        
        # 1. Play: Choose action using Actor
        with torch.no_grad():
            obs_tensor = preprocess_obs(obs, device)
            embed = world_model.encoder(obs_tensor)
            
            # Since we are online, we don't know the previous action without tracking, 
            # we'll pass a dummy zero action for the first step, or keep track.
            prev_action_tensor = torch.zeros(1, 5, device=device) # simplified for prototype
            
            # Update state with posterior (using actual observation)
            state, prior, post = world_model.rssm.observe_step(state, prev_action_tensor, embed)
            
            # Concatenate deter and stoch to get full state
            deter, stoch = state
            full_state = torch.cat([deter, stoch], dim=-1)
            
            # Get action distribution for logging
            action_dist = actor_critic.get_action_dist(full_state)
            
            # Thinking Step: Use LatentPlanner to find the best action
            action, expected_return = planner.plan(world_model, actor_critic, state, device)
            
            # Print action debugging occasionally
            if steps % 50 == 0:
                probs = F.softmax(action_dist.logits, dim=-1).cpu().numpy()[0]
                print(f"[DEBUG] Step {steps} | Level: {current_level} | Expected Return: {expected_return:.2f} | Selected: {action.item()}")
                
        action_val = action.item()
        
        # 2. Step in environment
        next_obs, reward, terminated, truncated, info = env.step(action_val)
        episode_reward += reward
        
        # Check for level up
        if info["level"] > current_level:
            print(f"[{steps}] 🌟 LEVEL UP! Advanced to Level {info['level']}")
            current_level = info["level"]
            
        # Check for life loss based on info
        if info["lives"] < current_lives:
            current_lives = info["lives"]
            if current_lives > 0:
                print(f"[{steps}] ⚠️  FUEL DEPLETED! Agent died. Lives remaining: {current_lives}/3")
            else:
                print(f"[{steps}] 💀  GAME OVER! All fuel/lives exhausted.")
        
        # 3. Render Memory if requested
        if args.render_memory and steps % 2 == 0:
            img_vis.set_data(next_obs["visible_frame"])
            img_mem.set_data(next_obs["memory_map"])
            plt.pause(0.01)
            
        # 4. Add to buffer
        buffer.add(obs, action_val, reward, terminated)
        obs = next_obs
        
        if terminated:
            print(f"Step {steps} | Environment Reset | Episode Reward: {episode_reward:.2f} | Level: {info['level']}")
            obs, info = env.reset()
            current_lives = info.get("lives", 3)
            current_level = info.get("level", 0)
            state = world_model.rssm.initial(1, device)
            episode_reward = 0
            
        # 5. Train (Online Updates)
        if len(buffer) > 64 and steps % 5 == 0:
            seq_len = 16
            batch_size = 32
            if len(buffer) < seq_len + 1:
                continue
                
            wm_optimizer.zero_grad()
            o_seq, a_seq, r_seq = buffer.sample(batch_size, seq_len)
            
            b_state = world_model.rssm.initial(batch_size, device)
            prev_action = torch.zeros(batch_size, 5, device=device)
            
            post_states = []
            reward_loss = 0
            kl_loss = 0
            
            # Backpropagation Through Time (BPTT)
            for t in range(seq_len):
                o_t = torch.cat([preprocess_obs(o_seq[b][t], device) for b in range(batch_size)], dim=0)
                a_t = F.one_hot(torch.tensor([a_seq[b][t] for b in range(batch_size)], device=device).long(), num_classes=5).float()
                r_t = torch.tensor([r_seq[b][t] for b in range(batch_size)], dtype=torch.float32, device=device).unsqueeze(1)
                
                embeds = world_model.encoder(o_t)
                b_state, b_prior, b_post = world_model.rssm.observe_step(b_state, prev_action, embeds)
                prev_action = a_t
                
                b_full_state = torch.cat([b_state[0], b_state[1]], dim=-1)
                post_states.append(b_full_state)
                
                pred_reward = world_model.rssm.reward_net(b_full_state)
                reward_loss = reward_loss + F.mse_loss(pred_reward, symlog(r_t))
                
                prior_mean, prior_logvar = b_prior
                post_mean, post_logvar = b_post
                kl_step = -0.5 * torch.sum(1 + post_logvar - prior_logvar - ((post_mean - prior_mean).pow(2) + post_logvar.exp()) / prior_logvar.exp(), dim=1).mean()
                kl_loss = kl_loss + kl_step
                
            wm_loss = (reward_loss + 0.1 * kl_loss) / seq_len
            wm_loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
            wm_optimizer.step()
            
            # --- Actor Critic Update (Imagination) ---
            ac_optimizer.zero_grad()
            
            # Flatten post_states to imagine from all visited states
            flattened_states = torch.cat(post_states, dim=0).detach()
            deter_dim = 256
            curr_state = (flattened_states[:, :deter_dim], flattened_states[:, deter_dim:])
            
            H = 15
            img_rewards = []
            img_values = []
            img_log_probs = []
            img_entropies = []
            
            # Imagine trajectories
            for t in range(H):
                full_s = torch.cat([curr_state[0], curr_state[1]], dim=-1)
                dist = actor_critic.get_action_dist(full_s)
                action = dist.sample()
                
                img_log_probs.append(dist.log_prob(action))
                img_entropies.append(dist.entropy())
                
                a_tensor = F.one_hot(action.long(), num_classes=5).float()
                curr_state = world_model.rssm.imagine_step(curr_state, a_tensor)
                next_full_s = torch.cat([curr_state[0], curr_state[1]], dim=-1)
                
                img_rewards.append(symexp(world_model.rssm.reward_net(next_full_s)))
                img_values.append(symexp(target_actor_critic.get_value(next_full_s)))
                
            # Compute Lambda Returns backward
            returns = img_values[-1].detach()
            actor_loss = 0
            
            for t in reversed(range(H)):
                returns = img_rewards[t].detach() + 0.99 * returns
                actor_loss = actor_loss - (img_log_probs[t] * symlog(returns).squeeze(-1)).mean() - 0.05 * img_entropies[t].mean()
                
            actor_loss = actor_loss / H
            
            # Critic Loss fits the very first state to the imagined return
            ret0 = img_rewards[0].detach() + 0.99 * img_values[0].detach()
            val_pred = actor_critic.get_value(flattened_states)
            critic_loss = F.mse_loss(val_pred, symlog(ret0))
            
            ac_loss = actor_loss + critic_loss
            ac_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor_critic.parameters(), 100.0)
            ac_optimizer.step()
            
            # Polyak Averaging for Target Critic
            with torch.no_grad():
                for param, target_param in zip(actor_critic.parameters(), target_actor_critic.parameters()):
                    target_param.data.copy_(0.01 * param.data + 0.99 * target_param.data)
            
            if steps % 100 == 0:
                print(f"[DEBUG] Loss | WM: {wm_loss.item():.4f} (Reward: {(reward_loss/seq_len).item():.4f}, KL: {(kl_loss/seq_len).item():.4f}) | AC: {ac_loss.item():.4f} (Actor: {actor_loss.item():.4f}, Critic: {critic_loss.item():.4f})")
                
            # Save Checkpoints
            if steps % 5000 == 0:
                os.makedirs("checkpoints", exist_ok=True)
                torch.save(world_model.state_dict(), f"checkpoints/world_model_step_{steps}.pth")
                torch.save(actor_critic.state_dict(), f"checkpoints/actor_critic_step_{steps}.pth")
                print(f"[{steps}] 💾 Model checkpoints saved to disk!")

if __name__ == "__main__":
    main()

# !mkdir /kaggle/working/logs # do not delete this line
# !python play_and_learn.py 2>&1 | tee /kaggle/working/logs/training.log # keep this here. 