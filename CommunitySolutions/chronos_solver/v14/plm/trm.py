"""v14 PLM — TRM belief core: O(1) recursive memory over the episode.

A GRU gives the fixed-size recursive belief state the architecture calls
for (Mamba-class efficiency at this scale, zero exotic deps). UNTESTED
SKELETON — first task on a live machine: `python -m plm.smoke`.
"""
import torch
import torch.nn as nn


class ActionEmbed(nn.Module):
    """(id, x, y) -> vector. x=y=0 for simple actions."""

    def __init__(self, cfg, dim):
        super().__init__()
        self.e_id = nn.Embedding(cfg.n_action_ids, dim)
        self.e_x = nn.Embedding(cfg.grid, dim)
        self.e_y = nn.Embedding(cfg.grid, dim)

    def forward(self, aid, ax, ay):
        return self.e_id(aid) + self.e_x(ax) + self.e_y(ay)


class BeliefCore(nn.Module):
    """H_t = GRU(H_{t-1}, [pool(Z_t), embed(A_{t-1})])

    The belief vector is the model's running hypothesis about the hidden
    rules of the current game. It is the ONLY memory across the episode.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_embed = nn.Embedding(cfg.codebook, cfg.code_dim)
        self.pool = nn.Sequential(
            nn.Linear(cfg.code_dim * cfg.tokens_per_frame, cfg.belief_dim),
            nn.GELU(),
        )
        self.act = ActionEmbed(cfg, cfg.belief_dim)
        self.gru = nn.GRUCell(cfg.belief_dim * 2, cfg.belief_dim)

    def initial(self, batch=1, device="cpu"):
        return torch.zeros(batch, self.cfg.belief_dim, device=device)

    def step(self, h, token_ids, aid, ax, ay):
        """h: (B, belief)  token_ids: (B, 8, 8) ints  a*: (B,) ints"""
        B = token_ids.shape[0]
        z = self.tok_embed(token_ids).reshape(B, -1)     # (B, 64*code_dim)
        zin = self.pool(z)
        ain = self.act(aid, ax, ay)
        return self.gru(torch.cat([zin, ain], -1), h)
