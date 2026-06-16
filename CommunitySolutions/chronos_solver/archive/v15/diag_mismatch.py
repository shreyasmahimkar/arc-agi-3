#!/usr/bin/env python3
"""v15 diagnostic — the script that found v14's memorization bug, updated
for the token-conditioned simulator.

THE NUMBER THAT MATTERS: [C] fresh-episode token accuracy on a trained
game. v14 scored 0.366 (vs 0.997 train) = memorization. The v15 gate is
>= 0.90 — if the architecture fix worked, even a small Mac-trained wm
should clear it, because "copy the static patches" alone scores ~0.9.

Run from v15 dir:  python diag_mismatch.py --game ar25 \
    [--shards /tmp/v15_shards]
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
    ap.add_argument("--shards", default="/tmp/v15_shards")
    ap.add_argument("--weights", default=os.path.join(HERE, "plm_weights.pt"))
    args = ap.parse_args()

    cfg = PLMConfig()
    device = torch.device("cpu")        # determinism > speed for diagnosis
    state = torch.load(args.weights, map_location=device, weights_only=True)
    print(f"weights keys: {list(state)}")
    tok = Tokenizer(cfg).to(device).eval()
    tok.load_state_dict(state["tokenizer"])

    # ---------- A: frame parity (raw engine vs camera env) ----------
    fr_raw = raw_reset_frame(load_engine(args.game))
    fr_env = env_reset_frame(args.game)
    print(f"\n[A] raw engine frame: {fr_raw.shape}  env frame: {fr_env.shape}")
    if fr_raw.shape != fr_env.shape:
        print("[A] *** SHAPE MISMATCH — training saw a different canvas than "
              "play_game serves ***")
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

    # ---------- C: fresh-episode accuracy (THE memorization detector) ----------
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
    T = int(lengths[0])
    g, a = grids[:T + 1], actions[:T]
    with torch.no_grad():
        ids = tok.encode(torch.stack([frame_to_tensor(f, cfg.n_colors)
                                      for f in g]))[1]          # (T+1, 8, 8)

        # copy baseline: what does "predict no change" score on this episode?
        copy_acc = float((ids[:-1].reshape(len(ids) - 1, -1)
                          == ids[1:].reshape(len(ids) - 1, -1))
                         .float().mean())

        def replay(reset_every=None):
            h = core.initial(1, device)
            prev = torch.zeros(1, 3, dtype=torch.long)
            accs = []
            for t in range(min(T, 60)):
                if reset_every and t % reset_every == 0:
                    h = core.initial(1, device)
                    prev = torch.zeros(1, 3, dtype=torch.long)
                h = core.step(h, ids[t:t+1], prev[:, 0], prev[:, 1], prev[:, 2])
                at = torch.tensor(a[t:t+1], dtype=torch.long)
                cur = ids[t:t+1].reshape(1, -1)          # v15: current tokens
                tl, _, _, _ = sim(h, cur, at[:, 0], at[:, 1], at[:, 2])
                accs.append(float((tl.argmax(-1).flatten()
                                   == ids[t+1].flatten()).float().mean()))
                prev = at
            return accs

        cont = replay()
        wind = replay(reset_every=8)
        print(f"[C] copy baseline (predict no change): {copy_acc:.3f}")
        print(f"[C] continuous belief (live behavior):  "
              f"mean {sum(cont)/len(cont):.3f}   (v14 scored 0.366; "
              f"gate >= 0.90 and > copy baseline)")
        print(f"    per-step t0..t11: "
              + " ".join(f"{x:.2f}" for x in cont[:12]))
        print(f"    per-step t12+  : mean {sum(cont[12:])/max(len(cont[12:]),1):.3f}")
        print(f"[C2] windowed K=8 (training behavior): "
              f"mean {sum(wind)/len(wind):.3f}")

    # ---------- E: VALUE COMPASS along the expert corridor ----------
    # Replay the v13 cached solution through the real engine and ask the
    # value head at every step: "how close to a win is this?" A working
    # compass rises ~gamma^d -> 1.0 toward the win. Flat ~0 = the planner
    # is flying blind (exactly what p=0.00 in the play log means).
    import json
    if 'core' not in dir() or 'sim' not in dir():
        print("\n[E] skipped — belief/world_model not loaded (see [C])")
        return
    sol = None
    for vdir in ('v13', 'v12'):
        cp = os.path.join(HERE, '..', vdir, f'{vdir}_bfs_cache_{args.game}.json')
        if os.path.exists(cp):
            sols = json.load(open(cp))
            if "0" in sols:
                sol = sols["0"]
                break
    if sol is None:
        print("\n[E] skipped — no cached L0 solution for this game")
    else:
        from arcengine import ActionInput as _AI
        g = load_engine(args.game)()
        g.perform_action(_AI(id=GameAction.RESET), raw=True)
        r = g.perform_action(_AI(id=GameAction.RESET), raw=True)
        gamma = cfg.value_gamma
        h = core.initial(1, device)
        prev = torch.zeros(1, 3, dtype=torch.long)
        print(f"\n[E] value compass along the {len(sol)}-action expert path "
              f"(want: rising ~{gamma:.2f}^d -> 1.0):")
        rows = []
        frame = np.array(r.frame[-1], dtype=np.uint8)
        for t, (aid_, data) in enumerate(sol):
            x_ = (data or {}).get('x', 0)
            y_ = (data or {}).get('y', 0)
            ids_t = tok.encode(frame_to_tensor(frame, cfg.n_colors)
                               .unsqueeze(0))[1]
            h = core.step(h, ids_t, prev[:, 0], prev[:, 1], prev[:, 2])
            at = torch.tensor([[aid_, x_, y_]], dtype=torch.long)
            _, _, _, vpred = sim(h, ids_t.reshape(1, -1),
                                 at[:, 0], at[:, 1], at[:, 2])
            tgt = gamma ** (len(sol) - 1 - t)
            rows.append((t, float(vpred), tgt))
            prev = at
            ai = _AI(id=GameAction.from_id(aid_),
                     data={'x': x_, 'y': y_, 'game_id': 'diag'}) \
                if aid_ == 6 else _AI(id=GameAction.from_id(aid_))
            r = g.perform_action(ai, raw=True)
            if r.frame:
                frame = np.array(r.frame[-1], dtype=np.uint8)
        for t, vp, tgt in rows:
            bar = '#' * int(vp * 40)
            print(f"    t={t:2d} pred={vp:.3f} target={tgt:.3f} {bar}")
        corr = np.corrcoef([r_[1] for r_ in rows],
                           [r_[2] for r_ in rows])[0, 1]
        print(f"[E] pred/target correlation: {corr:.3f} "
              "(>0.7 = compass works; ~0 = value head failed to learn)")

    # ---------- D: codebook health ----------
    with torch.no_grad():
        E = tok.vq.embed
        d = torch.cdist(E, E) + torch.eye(len(E)) * 1e9
        nn_dist = d.min(1).values
        scale = E.pow(2).sum(1).sqrt().mean()
        dup = int((nn_dist < 1e-3 * scale).sum())
        x = torch.stack([frame_to_tensor(f, cfg.n_colors) for f in g[:64]])
        z = tok.enc(x)
        _, ids0, _ = tok.vq(z)
        used = len(set(ids0.flatten().tolist()))
        flips = []
        for eps in (1e-5, 1e-4, 1e-3):
            _, ids_eps, _ = tok.vq(z + eps * torch.randn_like(z))
            flips.append(float((ids_eps != ids0).float().mean()))
        print(f"\n[D] codebook: {dup} near-duplicate codes, "
              f"{used}/{len(E)} in use, flip rates "
              + "/".join(f"{f:.3f}" for f in flips)
              + "  (known-degenerate but stable; revival queued)")


if __name__ == "__main__":
    main()
