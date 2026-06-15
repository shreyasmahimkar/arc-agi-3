"""Test-Time Training (TTT) — the technique behind the 2025 ARC Prize winners
(NVARC, MindsAI, ARChitects all rely on it). Iter-4 showed one shared cross-game
policy can't serve 23 conflicting games (cd82 acc 0.27 vs ls20 0.56). TTT fixes
that by ADAPTING the base model to the specific game at test time:

  1. SCOUT  — briefly explore the target game/level (random + macro), collecting
     its own transitions (feat, action, next_feat, progressed).
  2. ADAPT  — fine-tune a copy of the base cross-game TRM on those transitions
     (a few gradient steps), specialising policy/value to THIS game.
  3. SEARCH — run the value-MCTS with the adapted model.

Scout actions are RHAE debt, so keep the scout short. This is the v15 TTT idea,
now grounded in the 2025 winners' recipe.
"""
from __future__ import annotations
import os, random
import numpy as np
import engine as E
import trm
from vlog import get_logger

ACTION_CLASSES = [1, 2, 3, 4, 6]
HERE = os.path.dirname(__file__)


def _aclass(a):
    return ACTION_CLASSES.index(a) if a in ACTION_CLASSES else ACTION_CLASSES.index(6)


def scout(game, level, eps=4, steps=30):
    """Collect this game's own transitions from the level start."""
    cache, _ = E.load_cache(game)
    S, A, P = [], [], []
    for _ in range(eps):
        g = E.load_game(game)
        try:
            r, _ = E.chain_to_level(g, level, cache)
        except Exception:
            r = E.reset(g)
        root = dict(E.scalar_state(g)); f = E.frame_of(r)
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
            prog = sum(1 for k, v in dict(E.scalar_state(g)).items() if root.get(k) != v)
            S.append(of); A.append(_aclass(a)); P.append(1 if prog else 0); f = f2
    return np.array(S, np.float32), np.array(A, np.int64), np.array(P, np.int64)


def adapt(game, level, base_path=None, eps=4, steps=30, epochs=60, lg=None):
    """Return a TTT-adapted TRM specialised to (game, level)."""
    lg = lg or get_logger("ttt", "ttt")
    base_path = base_path or os.path.join(HERE, "models", "puzzle_trm.npz")
    S, A, P = scout(game, level, eps, steps)
    if len(S) < 10:
        lg.info(f"[TTT] {game} L{level}: scout too small ({len(S)}), using base")
        return trm.TRM.load(base_path), len(S)
    m = trm.TRM.load(base_path)
    m._adam = {}                       # reset optimiser state for finetune
    acl = np.where(P == 1, A, -1)
    costs = np.where(P == 1, 0.0, 20.0).astype(np.float32)
    npos = int((P == 1).sum())
    lg.info(f"[TTT] {game} L{level}: scouted {len(S)} transitions ({npos} progress) "
            f"-> finetuning {epochs} epochs")
    m.fit(S, acl, costs, epochs=epochs, lr=2e-3, bsz=64, iter_tag="ttt")
    return m, len(S)
