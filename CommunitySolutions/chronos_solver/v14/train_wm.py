#!/usr/bin/env python3
"""
v14 Phases 1+2 — train the tokenizer, then the world model.

    # Phase 1: VQ tokenizer (pixel-exact reconstruction is the gate)
    python train_wm.py --phase tok --shards /tmp/v14_shards --epochs 10

    # Phase 2: belief core + simulator, K-step BPTT, tokenizer frozen
    python train_wm.py --phase wm --shards /tmp/v14_shards --epochs 20 \
        --holdout ls20,vc33,tu93,ft09,sp80

Outputs plm_weights.pt = {"tokenizer","belief","world_model"} state dicts
(the exact format PLMAgent._load_weights expects).

Hardware: reuses the spirit of the v13 HW profile — bf16 autocast on cuda,
fp32 on mps/cpu, OOM batch backoff. UNTESTED SKELETON.
"""
import argparse
import glob
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plm.config import PLMConfig                      # noqa: E402
from plm.encoder import Tokenizer, frame_to_tensor    # noqa: E402
from plm.trm import BeliefCore                        # noqa: E402
from plm.world_model import BlockCausalSimulator      # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def device_setup():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True        # fixed 64x64 inputs
        return torch.device("cuda"), True             # amp on
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps"), False             # fp32 for stability
    return torch.device("cpu"), False


def load_shards(shard_dir, holdout):
    """Returns {game_id: episode list}; episode = (grids, actions, rewards)."""
    train, held = {}, {}
    for p in sorted(glob.glob(os.path.join(shard_dir, "*.npz"))):
        gid = os.path.basename(p)[:-4]
        z = np.load(p)
        eps, g0, a0 = [], 0, 0
        for ln in z["lengths"]:
            eps.append((z["grids"][g0:g0 + ln + 1],
                        z["actions"][a0:a0 + ln],
                        z["rewards"][a0:a0 + ln]))
            g0 += ln + 1
            a0 += ln
        (held if gid in holdout else train)[gid] = eps
    logger.info(f"shards: {len(train)} train games, {len(held)} held-out")
    return train, held


def batch_frames(data, n, device, cfg):
    """Random frames -> (x, target) for tokenizer training."""
    frames = []
    for _ in range(n):
        eps = random.choice(random.choice(list(data.values())))
        frames.append(eps[0][random.randrange(len(eps[0]))])
    x = torch.stack([frame_to_tensor(f.astype(np.int64), cfg.n_colors)
                     for f in frames]).to(device)
    y = torch.from_numpy(np.stack(frames).astype(np.int64)).to(device)
    return x, y


def train_tokenizer(cfg, data, device, amp, args, state):
    tok = Tokenizer(cfg).to(device)
    opt = torch.optim.AdamW(tok.parameters(), lr=3e-4)
    bsz = args.bsz
    for ep in range(args.epochs):
        tot = n = 0
        for step in range(args.steps_per_epoch):
            x, y = batch_frames(data, bsz, device, cfg)
            opt.zero_grad()
            try:
                if amp:
                    with torch.autocast("cuda", torch.bfloat16):
                        loss, recon, _ = tok(x, y)
                else:
                    loss, recon, _ = tok(x, y)
                loss.backward(); opt.step()
            except (RuntimeError, MemoryError) as e:   # OOM backoff (v13 habit)
                if 'out of memory' not in str(e).lower():
                    raise
                bsz = max(8, bsz // 2)
                torch.cuda.empty_cache()
                logger.warning(f"OOM -> bsz {bsz}")
                continue
            tot += recon.item(); n += 1
        # GATE: pixel-exact reconstruction rate on a fresh batch
        with torch.no_grad():
            x, y = batch_frames(data, 64, device, cfg)
            q, _ = tok.encode(x)
            acc = (tok.dec(q).argmax(1) == y).float().mean().item()
        logger.info(f"tok epoch {ep}: recon_loss {tot/max(n,1):.4f} "
                    f"pixel_acc {acc:.4f} (gate: 0.995)")
    state["tokenizer"] = tok.state_dict()
    return tok


def train_world_model(cfg, data, held, device, amp, args, state, tok):
    tok.eval()
    core = BeliefCore(cfg).to(device)
    sim = BlockCausalSimulator(cfg).to(device)
    opt = torch.optim.AdamW(list(core.parameters()) + list(sim.parameters()),
                            lr=3e-4)
    K = args.bptt

    def episode_batch(src, n):
        """n random K+1-frame windows -> tokens(T+1), actions(T), rewards(T).
        Retries until the batch is non-empty: episodes shorter than the BPTT
        window are skipped, and torch.stack on an empty list would crash."""
        toks, acts, rews = [], [], []
        attempts = 0
        while len(toks) < n and attempts < n * 20:
            attempts += 1
            eps = random.choice(random.choice(list(src.values())))
            g, a, r = eps
            if len(a) <= K:
                continue
            t0 = random.randrange(len(a) - K)
            with torch.no_grad():
                x = torch.stack([frame_to_tensor(f.astype(np.int64), cfg.n_colors)
                                 for f in g[t0:t0 + K + 1]]).to(device)
                _, ids = tok.encode(x)               # (K+1, 8, 8)
            toks.append(ids)
            acts.append(torch.from_numpy(a[t0:t0 + K].astype(np.int64)))
            rews.append(torch.from_numpy(r[t0:t0 + K].astype(np.int64)))
        if not toks:
            raise RuntimeError(
                f"no episodes longer than bptt={K} in shards — regenerate "
                f"data with --max-steps > {K} or lower --bptt")
        return (torch.stack(toks), torch.stack(acts).to(device),
                torch.stack(rews).to(device))

    def rollout_loss(toks, acts, rews):
        """Teacher-forced K-step BPTT through the GRU belief."""
        B = toks.shape[0]
        h = core.initial(B, device)
        prev_a = torch.zeros(B, 3, dtype=torch.long, device=device)
        loss = torch.zeros((), device=device)
        tok_correct = tok_total = 0
        for t in range(K):
            h = core.step(h, toks[:, t], prev_a[:, 0], prev_a[:, 1], prev_a[:, 2])
            a = acts[:, t]
            tl, rl, ch = sim(h, a[:, 0], a[:, 1], a[:, 2])
            tgt = toks[:, t + 1].reshape(B, -1)              # (B, 64)
            loss = loss + F.cross_entropy(tl.reshape(-1, cfg.codebook),
                                          tgt.reshape(-1))
            loss = loss + 0.5 * F.cross_entropy(rl, rews[:, t])
            changed = (tgt != toks[:, t].reshape(B, -1)).any(-1).float()
            loss = loss + 0.2 * F.binary_cross_entropy_with_logits(ch, changed)
            tok_correct += (tl.argmax(-1) == tgt).sum().item()
            tok_total += tgt.numel()
            prev_a = a
        return loss / K, tok_correct / max(tok_total, 1)

    for ep in range(args.epochs):
        for step in range(args.steps_per_epoch):
            toks, acts, rews = episode_batch(data, args.bsz)
            opt.zero_grad()
            if amp:
                with torch.autocast("cuda", torch.bfloat16):
                    loss, acc = rollout_loss(toks, acts, rews)
            else:
                loss, acc = rollout_loss(toks, acts, rews)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(core.parameters()) + list(sim.parameters()), 1.0)
            opt.step()
        # GATE: accuracy on HELD-OUT GAMES = the generalization number
        with torch.no_grad():
            ht, ha, hr = episode_batch(held or data, 32)
            _, hacc = rollout_loss(ht, ha, hr)
        logger.info(f"wm epoch {ep}: train_tok_acc {acc:.3f} "
                    f"HELDOUT_tok_acc {hacc:.3f} (gate: 0.90)")
    state["belief"] = core.state_dict()
    state["world_model"] = sim.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tok", "wm", "all"], default="all")
    ap.add_argument("--shards", default="/tmp/v14_shards")
    ap.add_argument("--holdout", default="ls20,vc33,tu93,ft09,sp80")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--bptt", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(HERE, "plm_weights.pt"))
    args = ap.parse_args()

    cfg = PLMConfig()
    device, amp = device_setup()
    logger.info(f"device={device} amp={amp}")
    train, held = load_shards(args.shards, set(args.holdout.split(",")))

    state = {}
    if os.path.exists(args.out):                       # resume-friendly
        state = dict(torch.load(args.out, map_location=device,
                                weights_only=True))
        logger.info(f"resuming from {args.out}: {list(state)}")

    if args.phase in ("tok", "all"):
        tok = train_tokenizer(cfg, train, device, amp, args, state)
    else:
        tok = Tokenizer(cfg).to(device)
        tok.load_state_dict(state["tokenizer"])
    if args.phase in ("wm", "all"):
        train_world_model(cfg, train, held, device, amp, args, state, tok)

    torch.save(state, args.out)
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
