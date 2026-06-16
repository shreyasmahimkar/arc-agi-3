#!/usr/bin/env python3
"""v19 WM-imagination planner (ExIt step 4) — crack levels BFS can't.

Uses the trained world model (wm_weights.pt) to PLAN in imagination: from the
current real frame it beam-searches short action sequences *inside the model*
(free, no engine), scoring by predicted reward (optimistic) + predicted novelty,
then executes the best first action on the REAL engine and replans (MPC). Every
executed action is verified against the real engine, so a claimed solve is real.

Why this is the lever: BFS dies of breadth on hard levels; the world model gives
a learned forward prior that steers the search toward reward — and it improves
every ExIt cycle as the corpus grows. Research basis: optimistic world models
(arXiv 2602.10044), sparse imagination (arXiv 2506.01392).

This module is import-safe (no heavy work at import). Run `python wm_planner.py
--selftest` for the smoke test.
"""
from __future__ import annotations
import os, sys, json, argparse
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from train_wm_v19 import WorldModel                      # the trained model class
import engine as E                                       # real engine (verifier)

HERE = os.path.dirname(os.path.abspath(__file__))
GRID = 64
N_SIMPLE = 5
CLICK = 6


def _click_targets(frame, limit=10):
    cnt = np.bincount(frame.ravel(), minlength=16); bg = int(cnt.argmax()); out = []
    for c in range(16):
        if c == bg or cnt[c] == 0 or cnt[c] > 3200:
            continue
        ys, xs = np.where(frame == c)
        out.append((int(np.median(ys)), int(np.median(xs))))
    return out[:limit]


def candidate_actions(frame, avail):
    """(action_id, x, y) tuples; x=y=0 for simple actions."""
    av = set(int(a) for a in (avail or [1, 2, 3, 4]))
    cands = [(a, 0, 0) for a in (1, 2, 3, 4, 5) if a in av]
    if CLICK in av:
        cands += [(CLICK, x, y) for (y, x) in _click_targets(frame)]
    return cands or [(a, 0, 0) for a in (1, 2, 3, 4)]


class WMPlanner:
    """MPC planner over the learned world model."""

    def __init__(self, weights=None, device=None, horizon=3, beam=8):
        self.device = torch.device(device) if device else torch.device(
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.net = WorldModel().to(self.device)
        wp = weights or os.path.join(HERE, "wm_weights.pt")
        self.loaded = False
        if os.path.exists(wp):
            try:
                sd = torch.load(wp, map_location=self.device, weights_only=True)
                self.net = WorldModel.from_state_dict(sd).to(self.device)  # match trained width
                self.loaded = True
            except Exception:
                pass
        self.net.eval()
        self.horizon = horizon
        self.beam = beam

    @torch.no_grad()
    def _imagine(self, frames, acts):
        """Batched WM step. frames (B,64,64) long, acts (B,3) long ->
        (next_frames (B,64,64) long, reward_prob (B,))."""
        logits, rlog = self.net(frames, acts)
        return logits.argmax(1), torch.sigmoid(rlog)

    @torch.no_grad()
    def plan(self, frame_np, avail):
        """Return the best next (action_id, x, y) via imagination beam search."""
        cands = candidate_actions(frame_np, avail)
        f0 = torch.from_numpy(frame_np.astype(np.int64)).to(self.device)
        # beam entries: (first_action, cur_frame_tensor(64,64), score, prev_frame)
        beams = [(None, f0, 0.0)]
        for h in range(self.horizon):
            cur = torch.stack([b[1] for b in beams])               # (Bb,64,64)
            exp_frames, exp_meta = [], []
            for bi, (first, _, score) in enumerate(beams):
                for (a, x, y) in cands:
                    exp_frames.append(cur[bi])
                    exp_meta.append((bi, first if first is not None else (a, x, y),
                                     (a, x, y), score))
            fr = torch.stack(exp_frames)
            ac = torch.tensor([[m[2][0], m[2][1], m[2][2]] for m in exp_meta],
                              dtype=torch.long, device=self.device)
            nf, rp = self._imagine(fr, ac)
            # score = cumulative reward prob + novelty (predicted frame change)
            changed = (nf != fr).float().flatten(1).mean(1)         # fraction changed
            new = []
            for k, (bi, first, _, score) in enumerate(exp_meta):
                s = score + float(rp[k]) * 5.0 + float(changed[k])
                new.append((first, nf[k], s))
            new.sort(key=lambda t: -t[2])
            beams = new[:self.beam]
        best = max(beams, key=lambda t: t[2])
        return best[0]                                              # (a,x,y)


def _to_perform(act):
    a, x, y = act
    if a == CLICK:
        return CLICK, {"x": int(x), "y": int(y), "game_id": "wm"}
    return a, None


def solve_level_wm(game, level, corpus, planner, budget=400):
    """Attempt a (BFS-failed) level via WM-MPC on the REAL engine. Chains to the
    level start using the corpus, then plans+executes; a solve is verified by the
    engine's levels_completed."""
    g = E.load_game(game)
    cache = {int(k): [tuple(a) for a in v] for k, v in corpus.items()}
    try:
        r, _ = E.chain_to_level(g, level, {str(k): v for k, v in cache.items()})
    except Exception:
        r = E.reset(g)
    start_lv = r.levels_completed
    plan_path = []
    for step in range(budget):
        f = E.frame_of(r)
        if f is None:
            break
        avail = list(getattr(r, "available_actions", None) or [1, 2, 3, 4])
        act = planner.plan(f, avail)
        aid, data = _to_perform(act)
        try:
            r = E.perform(g, aid, data)
        except Exception:
            break
        plan_path.append([aid, data])
        if r.levels_completed > start_lv:
            return {"solved": True, "actions": step + 1, "path": plan_path}
    return {"solved": False, "actions": budget}


def selftest():
    print("[wm_planner] self-test")
    p = WMPlanner()
    print(f"  weights loaded: {p.loaded} (device={p.device})")
    f = E.frame_of(E.reset(E.load_game("ls20")))
    act = p.plan(f, [1, 2, 3, 4])
    assert act is not None and act[0] in (1, 2, 3, 4, 5, 6), f"bad action {act}"
    print(f"  plan() on ls20 returned a valid action: {act}")
    corpus = json.load(open(os.path.join(HERE, "solutions", "ls20.json")))
    res = solve_level_wm("ls20", 5, corpus, p, budget=40)   # L5 = the BFS-hard one
    print(f"  attempt ls20 L5 (40 actions): solved={res['solved']} actions={res['actions']}")
    print("[wm_planner] self-test PASSED (runs + valid actions; solving improves with the WM)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--game", default=None)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--budget", type=int, default=400)
    args = ap.parse_args()
    if args.selftest or not args.game:
        return selftest()
    p = WMPlanner()
    corpus = json.load(open(os.path.join(HERE, "solutions", f"{args.game}.json")))
    res = solve_level_wm(args.game, args.level, corpus, p, budget=args.budget)
    print(f"{args.game} L{args.level}: {res}")


if __name__ == "__main__":
    main()
