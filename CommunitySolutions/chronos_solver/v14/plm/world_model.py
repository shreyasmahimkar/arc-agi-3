"""v14 PLM — block-causal simulator: (belief, action) -> next frame tokens.

Predicts ALL 64 next-frame tokens in one forward pass (block prediction,
Dreamer-v4 style) plus reward and change heads. UNTESTED SKELETON.
"""
import torch
import torch.nn as nn

from .trm import ActionEmbed

REWARD_NEUTRAL, REWARD_WIN, REWARD_RESET = 0, 1, 2


class BlockCausalSimulator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        T = cfg.tokens_per_frame
        d = cfg.wm_dim
        self.belief_in = nn.Linear(cfg.belief_dim, d)
        self.act = ActionEmbed(cfg, d)
        self.query = nn.Parameter(torch.randn(T, d) * 0.02)   # one per cell
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.wm_heads, dim_feedforward=d * 4,
            batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, cfg.wm_layers)
        self.tok_head = nn.Linear(d, cfg.codebook)
        self.reward_head = nn.Linear(d, 3)
        self.change_head = nn.Linear(d, 1)

    def forward(self, h, aid, ax, ay):
        """h: (B, belief)  a*: (B,) -> (B,64,K) token logits,
        (B,3) reward logits, (B,) change logit."""
        B = h.shape[0]
        ctx = torch.stack([self.belief_in(h), self.act(aid, ax, ay)], 1)  # (B,2,d)
        q = self.query.unsqueeze(0).expand(B, -1, -1)                     # (B,T,d)
        seq = torch.cat([ctx, q], 1)                                      # (B,2+T,d)
        out = self.tf(seq)
        cells = out[:, 2:, :]                                             # (B,T,d)
        pooled = out[:, 0, :]
        return (self.tok_head(cells),
                self.reward_head(pooled),
                self.change_head(pooled).squeeze(-1))
