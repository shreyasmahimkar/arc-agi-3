"""v14 PLM — Puzzle Language Model package.

Modules:
    config       — PLMConfig dataclass (all dimensions in one place)
    encoder      — BFS object channels + CNN + VQ tokenizer
    trm          — recursive belief core (the O(1) episode memory)
    world_model  — block-causal simulator (belief, action) -> next frame
    planner      — latent BFS inside the simulator's imagination
    curiosity    — the Stochastic Goose (explore while the model is wrong)
    agent_plm    — runtime loop tying all of the above together
"""
from .config import PLMConfig  # noqa: F401
