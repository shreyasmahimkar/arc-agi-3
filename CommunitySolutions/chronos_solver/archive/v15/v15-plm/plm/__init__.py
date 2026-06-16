"""v15 PLM — Puzzle Language Model package.

v15 = v14 + the memorization fix. v14's simulator predicted all 64
next-frame tokens from a POOLED belief vector alone — the current frame
had no direct path to the output, so the identity map ("most patches
don't change") was unrepresentable except through a lossy bottleneck and
SGD memorized training episodes instead (train 0.997, fresh episodes of
the SAME game 0.366). v15's simulator cross-attends over the current
frame's 64 token embeddings: copying is now the cheap solution and
dynamics are learned as deltas on top of it.

Modules:
    config       — PLMConfig dataclass (all dimensions in one place)
    encoder      — BFS object channels + CNN + VQ tokenizer (= v14)
    trm          — recursive belief core (= v14)
    world_model  — token-conditioned simulator (THE v15 change)
    planner      — latent BFS, frontier now carries (belief, tokens)
    curiosity    — Stochastic Goose (pick() now takes current tokens)
    agent_plm    — runtime loop tying all of the above together
"""
from .config import PLMConfig  # noqa: F401
