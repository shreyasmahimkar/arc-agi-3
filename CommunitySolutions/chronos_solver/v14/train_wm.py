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
import multiprocessing as mp
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from plm.config import PLMConfig                      # noqa: E402
from plm.encoder import (Tokenizer, frame_to_tensor,  # noqa: E402
                         object_channels)
from plm.trm import BeliefCore                        # noqa: E402
from plm.world_model import BlockCausalSimulator      # noqa: E402


# ==================== [perf patch 2] PARALLEL FEATURIZATION ====================
# `ps` showed training pinned at 101% CPU = ONE core of 28. The hot spot
# is per-frame featurization. Split it: the one-hot half vectorizes across
# the whole batch in a single numpy op (no parallelism needed); the
# connected-components half (the actually-slow part) fans out to a worker
# pool. Workers return only 2x64x64 floats each, so IPC stays cheap.

_POOL = None


def init_pool():
    global _POOL
    if _POOL is None and mp.cpu_count() > 2:
        try:
            ctx = mp.get_context("fork") if sys.platform != "darwin" \
                else mp.get_context("spawn")
            _POOL = ctx.Pool(min(16, mp.cpu_count() - 2))
            logger.info(f"featurize pool: {_POOL._processes} workers")
        except Exception as e:
            logger.warning(f"featurize pool unavailable ({e}); single-core")


def featurize_frames(frames_u8, n_colors=16):
    """(B,64,64) uint8 -> (B, n_colors+2, 64, 64) float32 torch (CPU).
    Batched one-hot + pooled object channels."""
    f = frames_u8.astype(np.int64).clip(0, n_colors - 1)
    oh = np.eye(n_colors, dtype=np.float32)[f]            # (B,H,W,C)
    oh = oh.transpose(0, 3, 1, 2)                         # (B,C,H,W)
    if _POOL is not None:
        objs = _POOL.map(object_channels, list(frames_u8), chunksize=16)
    else:
        objs = [object_channels(fr) for fr in frames_u8]
    x = np.concatenate([oh, np.stack(objs)], axis=1)      # (B,C+2,H,W)
    return torch.from_numpy(x)

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
    """Random frames -> (x, target) for tokenizer training.
    Featurization is batched + pooled (see featurize_frames)."""
    frames = []
    for _ in range(n):
        eps = random.choice(random.choice(list(data.values())))
        frames.append(eps[0][random.randrange(len(eps[0]))])
    frames = np.stack(frames)
    x = featurize_frames(frames, cfg.n_colors).to(device, non_blocking=True)
    y = torch.from_numpy(frames.astype(np.int64)).to(device, non_blocking=True)
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


# ==================== [perf patch] PRE-TOKENIZATION ====================
# Phase 2 used to re-featurize + re-encode ~2300 raw frames PER STEP in
# single-threaded Python — the GPU (H200 or RTX PRO 6000 alike) idled at
# 5-10% while the CPU crawled. The tokenizer is FROZEN in Phase 2, so we
# encode every frame exactly once, cache the token ids (64 uint16/frame,
# the whole dataset shrinks to ~200MB), and train the world model purely
# from token arrays. Device-agnostic: cuda / mps / cpu, with OOM backoff.

def _tok_fingerprint(tok):
    """Cheap checksum of the codebook — a retrained tokenizer must
    invalidate any stale token cache."""
    return float(tok.vq.embed.abs().sum().item())


def pretokenize_shards(cfg, data, tok, device, token_dir):
    """Encode all raw episodes once. Writes one npz per game into token_dir
    with the same episode layout as the raw shards (tokens/lengths/actions/
    rewards) + the tokenizer fingerprint."""
    os.makedirs(token_dir, exist_ok=True)
    fp = _tok_fingerprint(tok)
    tok.eval()
    enc_bsz = 2048                       # halved automatically on OOM
    for gid, eps in data.items():
        out = os.path.join(token_dir, f"{gid}.npz")
        if os.path.exists(out):
            try:
                if abs(float(np.load(out)["fingerprint"]) - fp) < 1e-3:
                    continue             # cache valid for this tokenizer
            except Exception:
                pass
        all_frames = np.concatenate([e[0] for e in eps])      # (N,64,64) u8
        ids_chunks = []
        i = 0
        while i < len(all_frames):
            chunk = all_frames[i:i + enc_bsz]
            try:
                with torch.no_grad():
                    x = featurize_frames(chunk, cfg.n_colors).to(
                        device, non_blocking=True)
                    _, ids = tok.encode(x)                    # (B,8,8)
                ids_chunks.append(ids.to("cpu", torch.int16).numpy())
                i += len(chunk)
            except (RuntimeError, MemoryError) as e:
                if 'out of memory' not in str(e).lower() or enc_bsz <= 64:
                    raise
                enc_bsz //= 2
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                logger.warning(f"pretok: OOM -> enc_bsz {enc_bsz}")
        np.savez_compressed(
            out,
            tokens=np.concatenate(ids_chunks),
            lengths=np.array([len(e[1]) for e in eps]),
            actions=np.concatenate([e[1] for e in eps]),
            rewards=np.concatenate([e[2] for e in eps]),
            fingerprint=np.float64(fp))
        logger.info(f"pretok {gid}: {len(all_frames)} frames -> {out}")


def load_token_shards(token_dir, games):
    """{gid: [(tokens(T+1,8,8) i16, actions(T,3) i16, rewards(T) u8)]}"""
    out = {}
    for gid in games:
        p = os.path.join(token_dir, f"{gid}.npz")
        if not os.path.exists(p):
            continue
        z = np.load(p)
        eps, t0, a0 = [], 0, 0
        for ln in z["lengths"]:
            eps.append((z["tokens"][t0:t0 + ln + 1],
                        z["actions"][a0:a0 + ln],
                        z["rewards"][a0:a0 + ln]))
            t0 += ln + 1
            a0 += ln
        out[gid] = eps
    return out


def train_world_model(cfg, tok_train, tok_held, device, amp, args, state):
    """Phase 2 on PRE-TOKENIZED data: the training loop never touches raw
    pixels or the tokenizer — pure tensor slicing feeding the GPU."""
    core = BeliefCore(cfg).to(device)
    sim = BlockCausalSimulator(cfg).to(device)
    sim_run = sim
    if args.compile and device.type == 'cuda':
        try:
            sim_run = torch.compile(sim)   # weights stay in `sim` (save that)
            logger.info("wm: torch.compile enabled for the simulator")
        except Exception as e:
            logger.warning(f"wm: compile unavailable ({e})")
    opt = torch.optim.AdamW(list(core.parameters()) + list(sim.parameters()),
                            lr=args.lr)
    K = args.bptt

    def episode_batch(src, n):
        """n random K+1 windows straight from token arrays — microseconds."""
        toks, acts, rews = [], [], []
        attempts = 0
        eps_pool = list(src.values())
        while len(toks) < n and attempts < n * 20:
            attempts += 1
            t, a, r = random.choice(random.choice(eps_pool))
            if len(a) <= K:
                continue
            t0 = random.randrange(len(a) - K)
            toks.append(torch.from_numpy(t[t0:t0 + K + 1].astype(np.int64)))
            acts.append(torch.from_numpy(a[t0:t0 + K].astype(np.int64)))
            rews.append(torch.from_numpy(r[t0:t0 + K].astype(np.int64)))
        if not toks:
            raise RuntimeError(
                f"no episodes longer than bptt={K} in token shards — "
                f"regenerate data with --max-steps > {K} or lower --bptt")
        return (torch.stack(toks).to(device, non_blocking=True),
                torch.stack(acts).to(device, non_blocking=True),
                torch.stack(rews).to(device, non_blocking=True))

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
            tl, rl, ch = sim_run(h, a[:, 0], a[:, 1], a[:, 2])
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

    import time as _time
    for ep in range(args.epochs):
        t_ep = _time.time()
        for step in range(args.steps_per_epoch):
            toks, acts, rews = episode_batch(tok_train, args.bsz)
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
            ht, ha, hr = episode_batch(tok_held or tok_train, 32)
            _, hacc = rollout_loss(ht, ha, hr)
        logger.info(f"wm epoch {ep}: train_tok_acc {acc:.3f} "
                    f"HELDOUT_tok_acc {hacc:.3f} (gate: 0.90) "
                    f"[{_time.time()-t_ep:.0f}s]")
    # NOTE: save the UNcompiled module's weights (compile wraps state_dict
    # keys with _orig_mod. — the v13 lesson, applied in reverse)
    state["belief"] = core.state_dict()
    state["world_model"] = sim.state_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tok", "pretok", "wm", "all"],
                    default="all")
    ap.add_argument("--shards", default="/tmp/v14_shards")
    ap.add_argument("--holdout", default="ls20,vc33,tu93,ft09,sp80")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps-per-epoch", type=int, default=500)
    ap.add_argument("--bsz", type=int, default=64)
    ap.add_argument("--bptt", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the simulator (cuda only; H200/RTX)")
    ap.add_argument("--out", default=os.path.join(HERE, "plm_weights.pt"))
    args = ap.parse_args()

    cfg = PLMConfig()
    device, amp = device_setup()
    logger.info(f"device={device} amp={amp}")
    init_pool()      # parallel featurization across CPU cores
    holdout = set(args.holdout.split(","))
    train, held = load_shards(args.shards, holdout)

    state = {}
    if os.path.exists(args.out):                       # resume-friendly
        state = dict(torch.load(args.out, map_location=device,
                                weights_only=True))
        logger.info(f"resuming from {args.out}: {list(state)}")

    # ---- Phase 1: tokenizer (raw pixels; CPU featurization tolerable) ----
    if args.phase in ("tok", "all"):
        tok = train_tokenizer(cfg, train, device, amp, args, state)
        torch.save(state, args.out)                    # checkpoint per phase
        logger.info(f"tokenizer checkpointed -> {args.out}")
    else:
        tok = Tokenizer(cfg).to(device)
        tok.load_state_dict(state["tokenizer"])

    # ---- Phase 1.5: pre-tokenize EVERYTHING once (the H200/RTX speed fix:
    # Phase 2's loop then never touches raw pixels or the tokenizer) ----
    token_dir = args.shards.rstrip("/") + "_tokens"
    if args.phase in ("pretok", "wm", "all"):
        pretokenize_shards(cfg, {**train, **held}, tok, device, token_dir)

    # ---- Phase 2: world model on token arrays (GPU-bound at last) ----
    if args.phase in ("wm", "all"):
        tok_train = load_token_shards(token_dir, list(train))
        tok_held = load_token_shards(token_dir, list(held))
        train_world_model(cfg, tok_train, tok_held, device, amp, args, state)

    torch.save(state, args.out)
    logger.info(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
