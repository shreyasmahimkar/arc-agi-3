"""v17 benchmark — the always-on scorecard. Replays the cached solutions
through the REAL engine (ground truth, no self-reported wins) and reports
actions-per-level + the RHAE proxy. Also verifies any candidate L5 solution.
"""
from __future__ import annotations
import numpy as np
import engine as E
from vlog import get_logger

# v13's action counts are the RHAE baselines we try to beat (fewer = better).
V13_BASELINE = {0: 13, 1: 45, 2: 39, 3: 43, 4: 44}


def replay_and_verify(game="ls20", upto_level=5, extra=None, iter_tag="v17"):
    """Replay cache L0..upto-1 (+ optional `extra` L_upto solution) through the
    real engine. Returns levels_completed reached + actions/level + RHAE."""
    lg = get_logger("bench", iter_tag)
    cache, cp = E.load_cache(game)
    g = E.load_game(game)
    r = E.reset(g)
    apl = {}
    reached = r.levels_completed
    for L in range(upto_level):
        sol = cache.get(str(L))
        if not sol:
            lg.info(f"[BENCH] L{L} missing in cache, stop")
            break
        before = r.levels_completed
        for act in sol:
            r = E.perform(g, act[0], act[1] if len(act) > 1 else None)
        apl[L] = len(sol)
        ok = r.levels_completed > before
        lg.info(f"[BENCH] L{L} replay {len(sol)} acts -> levels_completed="
                f"{r.levels_completed} ({'OK' if ok else 'NO-ADVANCE'})")
        reached = r.levels_completed
    if extra:
        before = r.levels_completed
        for act in extra:
            r = E.perform(g, act[0], act[1] if len(act) > 1 else None)
        apl[upto_level] = len(extra)
        ok = r.levels_completed > before
        lg.info(f"[BENCH] L{upto_level} CANDIDATE {len(extra)} acts -> "
                f"levels_completed={r.levels_completed} ({'WIN' if ok else 'FAIL'})")
        reached = r.levels_completed
    rhae, detail = E.rhae_score(apl, V13_BASELINE)
    lg.info(f"[BENCH] levels_completed={reached} actions_per_level={apl} "
            f"RHAE={rhae} detail={detail}")
    return {"levels_completed": int(reached), "actions_per_level": apl,
            "rhae": rhae, "rhae_detail": detail}
