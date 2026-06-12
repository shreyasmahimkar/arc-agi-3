"""v15 PLM — token-conditioned simulator: (belief, CURRENT TOKENS, action)
-> next frame tokens. THE v15 change.

v14's failure, measured: train_tok_acc 0.997 but 0.366 on fresh episodes
of the SAME game — pure memorization. Cause: the simulator predicted all
64 next tokens from [pooled belief, action] + 64 learned queries. The
current frame only reached the output through the 512-d belief
bottleneck, so the near-identity map (most patches don't change between
frames) was not cheaply representable, and SGD chose the cheaper
solution available to it: encode "which training trajectory is this" in
the belief and decode memorized continuations.

v15: the input sequence IS the current frame — 64 current-token
embeddings (+ position), with belief and action as two context tokens.
Each output position reads its own input token through residual
self-attention, so "copy" is the default and dynamics are learned as
deltas. The belief is freed to carry rules/history only.
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
        # v15: embed the CURRENT frame's tokens — this replaces v14's
        # blind learned queries and makes the identity map residual-cheap
        self.tok_embed = nn.Embedding(cfg.codebook, d)
        self.pos = nn.Parameter(torch.randn(T, d) * 0.02)     # one per cell
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.wm_heads, dim_feedforward=d * 4,
            batch_first=True, norm_first=True, activation="gelu")
        self.tf = nn.TransformerEncoder(layer, cfg.wm_layers)
        self.tok_head = nn.Linear(d, cfg.codebook)
        self.reward_head = nn.Linear(d, 3)
        self.change_head = nn.Linear(d, 1)

    def forward(self, h, cur_tokens, aid, ax, ay):
        """h: (B, belief)  cur_tokens: (B, 64) or (B, 8, 8) int ids
        a*: (B,) -> (B, 64, K) next-token logits, (B, 3) reward logits,
        (B,) change logit."""
        B = h.shape[0]
        cur = cur_tokens.reshape(B, -1)                           # (B, T)
        ctx = torch.stack([self.belief_in(h), self.act(aid, ax, ay)], 1)
        cells = self.tok_embed(cur) + self.pos.unsqueeze(0)       # (B, T, d)
        seq = torch.cat([ctx, cells], 1)                          # (B, 2+T, d)
        out = self.tf(seq)
        return (self.tok_head(out[:, 2:, :]),                    # next tokens
                self.reward_head(out[:, 0, :]),                  # from belief slot
                self.change_head(out[:, 1, :]).squeeze(-1))      # from action slot
