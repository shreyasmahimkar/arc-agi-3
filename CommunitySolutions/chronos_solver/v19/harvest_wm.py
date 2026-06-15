#!/usr/bin/env python3
"""v19 world-model data harvest (ExIt step 2).

Replays the corpus solutions (solutions/<game>.json) through the REAL engine and
records transitions (frame_t, action_t, frame_t+1, reward_t) for world-model
training. Adds a short novelty walk per game for negative/coverage transitions.
reward = 1 on a level-completion step, else 0 (the optimistic-WM signal).

OFFLINE on the public games (we hold the sources at train time); the world model
then carries this knowledge to hidden games inside its WEIGHTS — not as stored
answers. Output: wm_data.npz {grids,(N+1,64,64) actions,(N,3) rewards,(N,) game_ids}.
"""
from __future__ import annotations
import os, sys, json, glob, random
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v17")))
import engine as E   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "solutions")


def _act_xy(data):
    if isinstance(data, dict) and "x" in data:
        return int(data["x"]), int(data["y"])
    return 0, 0


def harvest_game(game, sol, explore_steps=120):
    """Replay all corpus levels in order, then a short novelty walk."""
    try:
        g = E.load_game(game)
    except Exception:
        return None
    r = E.reset(g)
    f0 = E.frame_of(r)
    if f0 is None:
        return None
    grids = [f0.astype(np.uint8)]
    actions, rewards = [], []
    prev_lv = r.levels_completed
    for lvl in sorted(sol, key=lambda k: int(k)):
        for act in sol[lvl]:
            aid = act[0]
            data = act[1] if len(act) > 1 and act[1] else None
            try:
                r = E.perform(g, aid, data)
            except Exception:
                break
            f = E.frame_of(r)
            if f is None:
                break
            x, y = _act_xy(data)
            grids.append(f.astype(np.uint8)); actions.append([aid, x, y])
            rewards.append(1 if r.levels_completed > prev_lv else 0)
            prev_lv = r.levels_completed
    # short novelty walk from the post-solution state for coverage / negatives
    rng = random.Random(0); seen = set()
    for _ in range(explore_steps):
        avail = list(getattr(r, "available_actions", None) or [1, 2, 3, 4])
        avail = [a for a in avail if a not in (0,)]
        if not avail:
            break
        aid = rng.choice(avail)
        data = None
        if aid == 6:
            f = E.frame_of(r)
            ys, xs = np.where(f != np.bincount(f.flatten(), minlength=16).argmax())
            if len(xs):
                j = rng.randrange(len(xs)); data = {"x": int(xs[j]), "y": int(ys[j])}
        try:
            r2 = E.perform(g, aid, data)
        except Exception:
            break
        f2 = E.frame_of(r2)
        if f2 is None:
            break
        x, y = _act_xy(data)
        grids.append(f2.astype(np.uint8)); actions.append([aid, x, y])
        rewards.append(1 if r2.levels_completed > prev_lv else 0)
        prev_lv = r2.levels_completed; r = r2
    if not actions:
        return None
    return (np.stack(grids), np.array(actions, dtype=np.int16),
            np.array(rewards, dtype=np.int8))


def main():
    games = sorted(os.path.basename(f)[:-5] for f in glob.glob(os.path.join(SOL, "*.json")))
    print(f"[harvest_wm] {len(games)} games with solutions: {games}")
    F, A, NF, R, G = [], [], [], [], []
    for gi, game in enumerate(games):
        sol = json.load(open(os.path.join(SOL, f"{game}.json")))
        out = harvest_game(game, sol)
        if out is None:
            print(f"  {game}: skip"); continue
        grids, actions, rewards = out
        F.append(grids[:-1]); NF.append(grids[1:]); A.append(actions)
        R.append(rewards); G.append(np.full(len(actions), gi, dtype=np.int16))
        print(f"  {game}: {len(actions)} transitions ({int(rewards.sum())} reward)")
    if not A:
        print("no data"); return
    F = np.concatenate(F); NF = np.concatenate(NF); A = np.concatenate(A)
    R = np.concatenate(R); G = np.concatenate(G)
    out_p = os.path.join(HERE, "wm_data.npz")
    np.savez_compressed(out_p, frames=F, next_frames=NF, actions=A, rewards=R,
                        game_ids=G, games=np.array(games, dtype=object))
    print(f"[harvest_wm] {len(A)} transitions, {int(R.sum())} reward-steps, "
          f"{len(games)} games -> {os.path.basename(out_p)} ({os.path.getsize(out_p)//1024} KB)")


if __name__ == "__main__":
    main()
