#!/usr/bin/env python3
"""v14 diagnostic — why does a 97%-train-acc model mispredict 70% live?

Three checks, most likely culprit first:

  A. FRAME PARITY: are the frames gen_data trained on (raw engine,
     perform_action(raw=True)) pixel-identical to the frames play_game
     serves (camera-rendered env.step)? The v12 ar25 lesson says: maybe not.
  B. TOKEN PARITY: do those two frames tokenize to the same ids?
  C. MODEL SANITY: replay a stored training episode through belief+sim on
     THIS machine and measure token accuracy. High (~0.95) = model and
     this machine's pipeline are fine, the live inputs are the problem.
     Low = the problem is in weights/precision/featurization here.

Run from v14 dir:  python diag_mismatch.py --game ar25 \
    [--shards /tmp/v14_shards]
"""
import argparse
import glob
import importlib.util
import os
import re
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))
sys.path.insert(0, HERE)

from arcengine import GameAction, ActionInput  # noqa: E402
for _m in GameAction:
    GameAction._value2member_map_.setdefault(_m.value, _m)

from plm.config import PLMConfig                       # noqa: E402
from plm.encoder import Tokenizer, frame_to_tensor     # noqa: E402
from plm.trm import BeliefCore                         # noqa: E402
from plm.world_model import BlockCausalSimulator       # noqa: E402


def load_engine(game_id):
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    src = glob.glob(os.path.join(env_dir, game_id, "**", f"{game_id}.py"),
                    recursive=True)[0]
    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', open(src).read())
    spec = importlib.util.spec_from_file_location('game_mod', src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['game_mod'] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, m.group(1))


def raw_reset_frame(game_cls):
    """The gen_data way: raw engine, double RESET."""
    g = game_cls()
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    return np.array(r.frame[-1], dtype=np.uint8)


def env_reset_frame(game_id):
    """The play_game way: arc_agi env, camera-rendered."""
    import arc_agi
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    arc = arc_agi.Arcade(environments_dir=env_dir,
                         operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make(game_id, render_mode=None)
    out = env.reset()
    lf = out[0] if isinstance(out, tuple) else out
    return np.array(lf.frame, dtype=np.uint8)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ar25")
    ap.add_argument("--shards", default="/tmp/v14_shards")
    ap.add_argument("--weights", default=os.path.join(HERE, "plm_weights.pt"))
    args = ap.parse_args()

    cfg = PLMConfig()
    device = torch.device("cpu")        # determinism > speed for diagnosis
    state = torch.load(args.weights, map_location=device, weights_only=True)
    print(f"weights keys: {list(state)}")
    tok = Tokenizer(cfg).to(device).eval()
    tok.load_state_dict(state["tokenizer"])

    # ---------- A: frame parity ----------
    fr_raw = raw_reset_frame(load_engine(args.game))
    fr_env = env_reset_frame(args.game)
    print(f"\n[A] raw engine frame: {fr_raw.shape}  env frame: {fr_env.shape}")
    if fr_raw.shape != fr_env.shape:
        print("[A] *** SHAPE MISMATCH — training saw a different canvas than "
              "play_game serves. This alone explains the 70% error. ***")
    else:
        d = int((fr_raw != fr_env).sum())
        print(f"[A] differing pixels: {d}/{fr_raw.size} "
              f"({100*d/fr_raw.size:.1f}%)"
              + ("  <- MISMATCH" if d else "  (identical — A is clean)"))

    # ---------- B: token parity ----------
    with torch.no_grad():
        ids_raw = tok.encode(frame_to_tensor(fr_raw, cfg.n_colors)
                             .unsqueeze(0))[1].flatten()
        if fr_raw.shape == fr_env.shape:
            ids_env = tok.encode(frame_to_tensor(fr_env, cfg.n_colors)
                                 .unsqueeze(0))[1].flatten()
            dt = int((ids_raw != ids_env).sum())
            print(f"[B] differing tokens: {dt}/{len(ids_raw)}"
                  + ("  <- MISMATCH" if dt else "  (identical — B is clean)"))

    # ---------- C: model sanity on stored training data ----------
    shard = os.path.join(args.shards, f"{args.game}.npz")
    if not os.path.exists(shard):
        print(f"[C] skipped — no shard at {shard} (regenerate with gen_data)")
        return
    if "belief" not in state:
        print("[C] skipped — weights have no belief/world_model keys")
        return
    core = BeliefCore(cfg).to(device).eval()
    core.load_state_dict(state["belief"])
    sim = BlockCausalSimulator(cfg).to(device).eval()
    sim.load_state_dict(state["world_model"])

    with np.load(shard) as z:
        grids, lengths = z["grids"], z["lengths"]
        actions = z["actions"]
    # first episode of the shard
    T = int(lengths[0])
    g, a = grids[:T + 1], actions[:T]
    with torch.no_grad():
        ids = tok.encode(torch.stack([frame_to_tensor(f, cfg.n_colors)
                                      for f in g]))[1]          # (T+1, 8, 8)

        def replay(reset_every=None):
            """Continuous belief (live behavior) vs windowed (training
            behavior: BPTT K=8, h=0 at every window start)."""
            h = core.initial(1, device)
            prev = torch.zeros(1, 3, dtype=torch.long)
            accs = []
            for t in range(min(T, 60)):
                if reset_every and t % reset_every == 0:
                    h = core.initial(1, device)
                    prev = torch.zeros(1, 3, dtype=torch.long)
                h = core.step(h, ids[t:t+1], prev[:, 0], prev[:, 1], prev[:, 2])
                at = torch.tensor(a[t:t+1], dtype=torch.long)
                tl, _, _ = sim(h, at[:, 0], at[:, 1], at[:, 2])
                accs.append(float((tl.argmax(-1).flatten()
                                   == ids[t+1].flatten()).float().mean()))
                prev = at
            return accs

        cont = replay()
        wind = replay(reset_every=8)
        print(f"[C] continuous belief (live behavior):  "
              f"mean {sum(cont)/len(cont):.3f}")
        print(f"    per-step t0..t11: "
              + " ".join(f"{x:.2f}" for x in cont[:12]))
        print(f"    per-step t12+  : mean {sum(cont[12:])/max(len(cont[12:]),1):.3f}")
        print(f"[C2] windowed K=8 (training behavior): "
              f"mean {sum(wind)/len(wind):.3f}")
        print("    C low everywhere + C2 high  -> GRU drift past the BPTT "
              "window: fix = sliding-window belief in the agent (no retrain)")
        print("    C and C2 both low           -> token/teacher mismatch "
              "deeper in the pipeline")

    # ---------- D: codebook degeneracy / platform-noise ids ----------
    diag_codebook(tok, cfg, g)


def diag_codebook(tok, cfg, grids):
    """[D] Is the VQ codebook degenerate? Near-duplicate codes mean argmin
    assignments sit on numerical ties -> token ids flip between platforms
    (instance CUDA encoded the training tokens; this machine re-encodes).
    The wm would then have learned a mapping over platform-specific noise."""
    with torch.no_grad():
        E = tok.vq.embed                                   # (K, D)
        d = torch.cdist(E, E) + torch.eye(len(E)) * 1e9
        nn_dist = d.min(1).values
        scale = E.pow(2).sum(1).sqrt().mean()
        dup = int((nn_dist < 1e-3 * scale).sum())
        print(f"\n[D] codebook: {len(E)} codes, mean norm {scale:.3f}")
        print(f"    near-duplicate codes (nn dist < 0.1% of norm): {dup}")

        x = torch.stack([frame_to_tensor(f, cfg.n_colors)
                         for f in grids[:64]])
        z = tok.enc(x)
        _, ids0, _ = tok.vq(z)
        used = len(set(ids0.flatten().tolist()))
        # perturbation test: does a numerically-tiny nudge flip the ids?
        flips = []
        for eps in (1e-5, 1e-4, 1e-3):
            _, ids_eps, _ = tok.vq(z + eps * torch.randn_like(z))
            flips.append(float((ids_eps != ids0).float().mean()))
        print(f"    codes actually used on 64 frames: {used}/{len(E)}")
        print(f"    id flip rate under z-noise 1e-5/1e-4/1e-3: "
              + " ".join(f"{f:.3f}" for f in flips))
        print("    high flip rate at 1e-5/1e-4 -> ids are platform noise; "
              "the wm trained on the instance's ids -> THIS is the bug")


if __name__ == "__main__":
    main()
