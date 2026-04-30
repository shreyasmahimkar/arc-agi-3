import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    """Encodes the 2-channel 64x64 grid (visible frame + memory map) into a dense representation."""
    def __init__(self, embed_dim=256):
        super().__init__()
        # Input: (B, 2, 64, 64)
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1), # (16, 32, 32)
            nn.SiLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # (32, 16, 16)
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # (64, 8, 8)
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # (128, 4, 4)
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, obs):
        # obs is a dictionary of tensors, or we expect a tensor
        # Here we expect a tensor of shape (B, 2, 64, 64)
        return self.net(obs)

class Decoder(nn.Module):
    """Reconstructs the 2-channel 64x64 grid from the latent state."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.fc = nn.Linear(embed_dim, 128 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # (64, 8, 8)
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1), # (32, 16, 16)
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1), # (16, 32, 32)
            nn.SiLU(),
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1), # (2, 64, 64)
        )
        
    def forward(self, latent):
        x = self.fc(latent)
        x = x.view(-1, 128, 4, 4)
        return self.net(x)

class RSSM(nn.Module):
    """
    Recurrent State-Space Model.
    For simplicity in this lightweight version, we use a continuous latent state
    instead of DreamerV3's discrete categorical latents, combined with a GRU.
    """
    def __init__(self, action_dim=5, deter_dim=256, stoch_dim=32, hidden_dim=256):
        super().__init__()
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim
        
        # RNN for deterministic state
        self.cell = nn.GRUCell(hidden_dim, deter_dim)
        
        # Prior predictor: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim) # mean and logvar
        )
        
        # Posterior predictor: q(z_t | h_t, x_t)
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + hidden_dim, hidden_dim), # deter_dim + embed_dim
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * stoch_dim)
        )
        
        # Input to RNN: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.rnn_input_net = nn.Sequential(
            nn.Linear(stoch_dim + action_dim, hidden_dim),
            nn.SiLU()
        )
        
        # Predictors from state (h_t, z_t)
        state_dim = deter_dim + stoch_dim
        
        self.reward_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # predict symlog reward
        )
        
        self.continue_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1) # predict logit for termination
        )
        
    def initial(self, batch_size, device):
        return (
            torch.zeros(batch_size, self.deter_dim, device=device),
            torch.zeros(batch_size, self.stoch_dim, device=device)
        )
        
    def observe_step(self, prev_state, prev_action, embed):
        """One step of posterior (training)."""
        prev_deter, prev_stoch = prev_state
        
        # 1. Update deterministic state
        rnn_in = self.rnn_input_net(torch.cat([prev_stoch, prev_action], dim=-1))
        deter = self.cell(rnn_in, prev_deter)
        
        # 2. Prior
        prior_stats = self.prior_net(deter)
        prior_mean, prior_logvar = torch.chunk(prior_stats, 2, dim=-1)
        
        # 3. Posterior (uses observation embed)
        post_stats = self.post_net(torch.cat([deter, embed], dim=-1))
        post_mean, post_logvar = torch.chunk(post_stats, 2, dim=-1)
        
        # Sample stochastic state
        std = torch.exp(0.5 * post_logvar)
        stoch = post_mean + std * torch.randn_like(std)
        
        return (deter, stoch), (prior_mean, prior_logvar), (post_mean, post_logvar)
        
    def imagine_step(self, prev_state, prev_action):
        """One step of prior (imagination/rollout)."""
        prev_deter, prev_stoch = prev_state
        
        rnn_in = self.rnn_input_net(torch.cat([prev_stoch, prev_action], dim=-1))
        deter = self.cell(rnn_in, prev_deter)
        
        prior_stats = self.prior_net(deter)
        prior_mean, prior_logvar = torch.chunk(prior_stats, 2, dim=-1)
        
        std = torch.exp(0.5 * prior_logvar)
        stoch = prior_mean + std * torch.randn_like(std)
        
        return (deter, stoch)

class WorldModel(nn.Module):
    def __init__(self, action_dim=5):
        super().__init__()
        self.encoder = Encoder(embed_dim=256)
        self.decoder = Decoder(embed_dim=256+32) # decodes from state (deter + stoch)
        self.rssm = RSSM(action_dim=action_dim, deter_dim=256, stoch_dim=32, hidden_dim=256)
