"""v14 PLM configuration — every dimension lives here."""
from dataclasses import dataclass


@dataclass
class PLMConfig:
    # grid
    grid: int = 64
    n_colors: int = 16
    patch: int = 8                 # 64/8 -> 8x8 = 64 tokens per frame
    # tokenizer
    codebook: int = 1024
    code_dim: int = 64
    enc_ch: int = 96               # CNN width
    # belief core (TRM)
    belief_dim: int = 512
    # world model
    wm_dim: int = 384
    wm_layers: int = 6
    wm_heads: int = 6
    # actions
    n_action_ids: int = 8          # RESET..ACTION7 (0..7)
    # planner
    plan_depth: int = 5
    plan_beam: int = 64            # surviving states per depth
    plan_topk_clicks: int = 12     # candidate clicks from object centroids
    # curiosity (goose)
    surprise_ema: float = 0.9
    explore_threshold: float = 0.12   # token error rate below which goose retires
    min_explore_steps: int = 8

    @property
    def tokens_per_frame(self) -> int:
        return (self.grid // self.patch) ** 2
