"""v15 — TEST-TIME TRAINING: pass 2, executed INSIDE the agent.

The agent goes in blind (hidden eval: unknown game, no engine source).
Pass 1 (scout) collects real transitions; this module finetunes the
belief core + simulator + value head ON THOSE TRANSITIONS under a
wall-clock budget, in-place, mid-game. The pretrained weights are the
prior; this is the posterior for the game actually being played.

Integrity: trains exclusively on the agent's OWN interactions from the
current run — nothing offline is consulted.
"""
import logging
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from .encoder import frame_to_tensor

logger = logging.getLogger(__name__)


def _value_targets(rewards, gamma):
    """gamma^(steps to next WIN); 0 when no win lies ahead."""
    v = np.zeros(len(rewards), np.float32)
    d = None
    for t in range(len(rewards) - 1, -1, -1):
        if rewards[t] == 1:
            d = 0
        elif d is not None:
            d += 1
        if d is not None:
            v[t] = gamma ** d
    return v


@torch.enable_grad()
def finetune(tok, core, sim, episodes, device, cfg,
             seconds=240.0, lr=1e-4, bsz=32):
    """episodes: list of (frames (T+1,64,64) uint8, actions (T,3) int,
    rewards (T,) int) gathered live. Tokenizes once (tokenizer FROZEN),
    then K-step BPTT on belief+sim until the budget expires.
    Returns stats dict; models are updated in-place and left in eval()."""
    K = 8
    t0 = time.monotonic()
    gamma = cfg.value_gamma

    # ---- tokenize the buffer once (frozen tokenizer) ----
    data = []
    tok.eval()
    with torch.no_grad():
        for g, a, r in episodes:
            if len(a) <= K:
                continue
            xs = torch.stack([frame_to_tensor(f, cfg.n_colors) for f in g])
            ids = []
            for i in range(0, len(xs), 256):
                _, idx = tok.encode(xs[i:i + 256].to(device))
                ids.append(idx.cpu())
            data.append((torch.cat(ids).numpy().astype(np.int16),
                         np.asarray(a, np.int16),
                         np.asarray(r, np.uint8),
                         _value_targets(np.asarray(r), gamma)))
    if not data:
        return {"steps": 0, "note": "no episodes longer than K"}
    n_wins = int(sum((d[2] == 1).sum() for d in data))

    core.train(); sim.train()
    opt = torch.optim.AdamW(list(core.parameters()) + list(sim.parameters()),
                            lr=lr)
    steps, last_loss = 0, float("nan")
    while time.monotonic() - t0 < seconds:
        toks, acts, rews, vals = [], [], [], []
        for _ in range(bsz):
            t_, a_, r_, v_ = random.choice(data)
            i0 = random.randrange(len(a_) - K)
            toks.append(torch.from_numpy(t_[i0:i0 + K + 1].astype(np.int64)))
            acts.append(torch.from_numpy(a_[i0:i0 + K].astype(np.int64)))
            rews.append(torch.from_numpy(r_[i0:i0 + K].astype(np.int64)))
            vals.append(torch.from_numpy(v_[i0:i0 + K]))
        toks = torch.stack(toks).to(device)
        acts = torch.stack(acts).to(device)
        rews = torch.stack(rews).to(device)
        vals = torch.stack(vals).to(device)

        B = toks.shape[0]
        h = core.initial(B, device)
        prev = torch.zeros(B, 3, dtype=torch.long, device=device)
        loss = torch.zeros((), device=device)
        for t in range(K):
            h = core.step(h, toks[:, t], prev[:, 0], prev[:, 1], prev[:, 2])
            a = acts[:, t]
            cur = toks[:, t].reshape(B, -1)
            tl, rl, ch, val = sim(h, cur, a[:, 0], a[:, 1], a[:, 2])
            tgt = toks[:, t + 1].reshape(B, -1)
            loss = loss + F.cross_entropy(tl.reshape(-1, cfg.codebook),
                                          tgt.reshape(-1))
            loss = loss + 0.5 * F.cross_entropy(rl, rews[:, t])
            changed = (tgt != cur).any(-1).float()
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(ch, changed)
            vt = vals[:, t]
            w = 1.0 + 2.0 * (vt > 0).float()
            loss = loss + 0.5 * (w * (val - vt) ** 2).mean()
            prev = a
        loss = loss / K
        if not torch.isfinite(loss):          # NaN fuse (the 06-12 lesson)
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(core.parameters()) + list(sim.parameters()), 1.0)
        opt.step()
        steps += 1
        last_loss = float(loss)
    core.eval(); sim.eval()
    return {"steps": steps, "loss": last_loss, "episodes": len(data),
            "wins_in_buffer": n_wins,
            "seconds": round(time.monotonic() - t0, 1)}
