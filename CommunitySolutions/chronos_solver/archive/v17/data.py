"""v17 training-data builder — turns v13's cached BFS solutions into labelled
states. This is the "BFS learnings" the heuristic and the TRM LM learn from.

For each solved level L (solution length S), replaying the solution visits
states s_0..s_S. We emit:
  * heuristic targets : (frame_i, obj_i, cost_to_go = S - i)
  * policy targets    : (frame_i, obj_i, expert_action_i)
  * off-path negatives: from s_i, one non-expert action -> s', labelled
    cost_to_go = (S - i) + 1  (a one-step-worse lower bound; gives the
    heuristic discriminative signal beyond the monotone on-path chain).

Action keys are integers 1..4 for moves and the tuple (6,x,y) for clicks; the
policy head predicts over {1,2,3,4, CLICK} (clicks collapsed to one class +
the chosen centroid is recorded for replay).
"""
from __future__ import annotations
import numpy as np
import engine as E
from vlog import get_logger

ACTION_CLASSES = [1, 2, 3, 4, 6]          # 6 == CLICK (any centroid)


def _akey(act):
    a, d = act[0], (act[1] if len(act) > 1 else None)
    return a if (d is None or a != 6) else (6, d.get("x"), d.get("y"))


def _aclass(act):
    a = act[0]
    return ACTION_CLASSES.index(a) if a in ACTION_CLASSES else ACTION_CLASSES.index(6)


def build_datasets(game="ls20", levels=(0, 1, 2, 3, 4), neg_per_state=1,
                   iter_tag="v17"):
    lg = get_logger("data", iter_tag)
    cache, cp = E.load_cache(game)
    frames, objs, costs, aclasses = [], [], [], []
    nstates = 0
    for L in levels:
        sol = cache.get(str(L))
        if not sol:
            lg.info(f"[DATA] L{L} no cached solution, skip")
            continue
        S = len(sol)
        g = E.load_game(game)
        r, _ = E.chain_to_level(g, L, cache)     # start of level L
        f = E.frame_of(r)
        for i, act in enumerate(sol):
            obj = E.object_features(f)
            frames.append(f.copy()); objs.append(obj)
            costs.append(float(S - i)); aclasses.append(_aclass(act))
            nstates += 1
            # off-path negative(s): try a non-expert move, label cost+1
            if neg_per_state:
                exp_a = act[0]
                alt = [a for a in E.MOVES if a != exp_a]
                if alt:
                    gg = E.load_game(game); E.chain_to_level(gg, L, cache)
                    for a2 in sol[:i]:
                        E.perform(gg, a2[0], a2[1] if len(a2) > 1 else None)
                    ra = E.perform(gg, alt[0])
                    fa = E.frame_of(ra)
                    if fa is not None:
                        frames.append(fa.copy()); objs.append(E.object_features(fa))
                        costs.append(float(S - i) + 1.0); aclasses.append(-1)
            # advance expert
            r = E.perform(g, act[0], act[1] if len(act) > 1 else None)
            f = E.frame_of(r)
        lg.info(f"[DATA] L{L} S={S} -> states so far={len(frames)}")
    lg.info(f"[DATA] total states={len(frames)} (on-path={nstates}) "
            f"cost_range=[{min(costs):.0f},{max(costs):.0f}] cache={cp}")
    return {
        "frames": frames, "objs": np.array(objs, dtype=np.float32),
        "costs": np.array(costs, dtype=np.float32),
        "aclasses": np.array(aclasses, dtype=np.int64),
    }
