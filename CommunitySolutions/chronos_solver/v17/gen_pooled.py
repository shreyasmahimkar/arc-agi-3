"""Puzzle-LM data — pooled cross-game transitions from all 25 official games.

This is the v15 move: instead of one game's tapes, learn dynamics from EVERY
game so the prior is cross-game. For each game we collect transitions
(s_t, a_t, s_{t+1}, progressed, win) in the shared object-feature "language"
(engine.object_features, 76-d), from two sources:

  * EXPERT  — replay v13's cached solutions (goal-directed, high value)
  * EXPLORE — random + macro walks from RESET (broad dynamics coverage)

Action classes: [1,2,3,4, CLICK]. Appends to a pooled .npz so the 25 games can
be swept in batches under the sandbox time cap.

  python gen_pooled.py --games ar25,bp35 --explore-eps 6 --steps 40
"""
from __future__ import annotations
import os, sys, json, argparse, random
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import engine as E
from vlog import get_logger

HERE = os.path.dirname(__file__)
POOL = os.path.join(HERE, "models", "pool.npz")
ACTION_CLASSES = [1, 2, 3, 4, 6]
ALL_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
             "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
             "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]


def _aclass(a):
    return ACTION_CLASSES.index(a) if a in ACTION_CLASSES else ACTION_CLASSES.index(6)


def _scal(g):
    return dict(E.scalar_state(g))


def collect_game(game, explore_eps, steps, lg):
    """Return list of (obj76, aclass, next_obj76, progressed, win, gameidx)."""
    rows = []
    gi = ALL_GAMES.index(game)
    cache, _ = E.load_cache(game)
    # ---- EXPERT: replay cached solutions, chained ----
    try:
        g = E.load_game(game); r = E.reset(g)
        root = _scal(g)
        f = E.frame_of(r)
        for lvl in range(len(cache)):
            for act in cache[str(lvl)]:
                a = act[0]; d = act[1] if len(act) > 1 and act[1] else None
                of = E.object_features(f)
                r2 = E.perform(g, a, d); f2 = E.frame_of(r2)
                if f2 is None:
                    break
                prog = sum(1 for k, v in _scal(g).items() if root.get(k) != v)
                win = bool(r2.levels_completed > 0 and a == act[0] and lvl < r2.levels_completed)
                rows.append((of, _aclass(a), E.object_features(f2), 1 if prog else 0,
                             1 if r2.levels_completed > lvl else 0, gi))
                f = f2
    except Exception as e:
        lg.info(f"[POOL] {game} expert skip: {repr(e)[:60]}")
    # ---- EXPLORE: random+macro walks from RESET ----
    try:
        for ep in range(explore_eps):
            g = E.load_game(game); r = E.reset(g); root = _scal(g)
            f = E.frame_of(r)
            if f is None:
                break
            for _ in range(steps):
                if random.random() < 0.15:
                    cl = E.dynamic_clicks(f, limit=6)
                    a, d = random.choice(cl) if cl else (random.choice(E.MOVES), None)
                else:
                    a, d = random.choice(E.MOVES), None
                of = E.object_features(f)
                r2 = E.perform(g, a, d); f2 = E.frame_of(r2)
                if f2 is None:
                    break
                prog = sum(1 for k, v in _scal(g).items() if root.get(k) != v)
                win = bool(r2.levels_completed > 0)
                rows.append((of, _aclass(a), E.object_features(f2),
                             1 if prog else 0, 1 if win else 0, gi))
                f = f2
    except Exception as e:
        lg.info(f"[POOL] {game} explore skip: {repr(e)[:60]}")
    lg.info(f"[POOL] {game}: collected {len(rows)} transitions")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(ALL_GAMES))
    ap.add_argument("--explore-eps", type=int, default=6)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()
    lg = get_logger("pool", "pool")
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)

    if args.reset or not os.path.exists(POOL):
        S = np.zeros((0, 76), np.float32); A = np.zeros((0,), np.int64)
        NS = np.zeros((0, 76), np.float32); P = np.zeros((0,), np.int64)
        W = np.zeros((0,), np.int64); GI = np.zeros((0,), np.int64)
    else:
        d = np.load(POOL)
        S, A, NS, P, W, GI = d["s"], d["a"], d["ns"], d["p"], d["w"], d["gi"]

    for game in args.games.split(","):
        rows = collect_game(game, args.explore_eps, args.steps, lg)
        if not rows:
            continue
        s = np.array([r[0] for r in rows], np.float32)
        a = np.array([r[1] for r in rows], np.int64)
        ns = np.array([r[2] for r in rows], np.float32)
        p = np.array([r[3] for r in rows], np.int64)
        w = np.array([r[4] for r in rows], np.int64)
        gi = np.array([r[5] for r in rows], np.int64)
        S = np.vstack([S, s]); A = np.concatenate([A, a]); NS = np.vstack([NS, ns])
        P = np.concatenate([P, p]); W = np.concatenate([W, w]); GI = np.concatenate([GI, gi])
        np.savez(POOL, s=S, a=A, ns=NS, p=P, w=W, gi=GI)
    lg.info(f"[POOL] total pooled transitions={len(S)} (progress={int(P.sum())}, "
            f"win={int(W.sum())}, games={len(np.unique(GI))})")
    print(f"pooled={len(S)} progress={int(P.sum())} win={int(W.sum())} games={len(np.unique(GI))}")


if __name__ == "__main__":
    main()
