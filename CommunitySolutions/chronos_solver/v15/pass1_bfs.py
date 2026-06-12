#!/usr/bin/env python3
"""v15 PASS 1 — the symbolic scout.

Runs v13's pure-BFS solver against the LOCAL engine for a wall-clock
budget (default 600s/game) and writes everything it learns into the game
scratchpad: exact solutions per level, per-action effect probes, where it
got stuck and why. The scratchpad then feeds PASS 2 (gen_data distills
solutions into PLM training data) and documents the game for any future
reasoning layer.

OFFLINE ONLY — needs engine source. On the hidden eval this pass simply
doesn't exist; its knowledge arrives there inside the PLM's weights.

Usage (repo venv, from v15/):
    python pass1_bfs.py --games ar25,bp35 --budget 600
    python pass1_bfs.py --games all --budget 300   # whole library sweep

Scratchpads land in v15_scratch/<game>.json. Resumable: already-solved
levels (from v13/v12 caches OR a previous pass-1 run) are hydrated, the
budget goes entirely into the frontier.
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
V13 = os.path.abspath(os.path.join(HERE, '..', 'v13'))
sys.path.insert(0, os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))
sys.path.insert(0, V13)          # v13's my_agent = the BFS engine room
sys.path.insert(0, HERE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(HERE, "pass1.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from my_agent import BFSSolver, HW          # noqa: E402  (v13's!)
from arcengine import GameAction, ActionInput  # noqa: E402
for _m in GameAction:
    GameAction._value2member_map_.setdefault(_m.value, _m)

import scratchpad as SP                      # noqa: E402


def find_game_source(game_id):
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3',
                           'environment_files')
    m = glob.glob(os.path.join(env_dir, game_id, "**", f"{game_id}.py"),
                  recursive=True)
    return m[0] if m else None


def probe_actions(game_cls, sp, n_clicks=5):
    """Measure what each action DOES from the level-0 start state —
    pixels changed per action, click effects at object centroids.
    Pure observation; written to the scratchpad as effects + notes."""
    def fresh():
        g = game_cls()
        g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        return g, np.array(r.frame[-1], dtype=np.int16)

    for aid in (1, 2, 3, 4, 5, 7):
        try:
            g, f0 = fresh()
            r = g.perform_action(ActionInput(id=GameAction.from_id(aid)),
                                 raw=True)
            f1 = np.array(r.frame[-1], dtype=np.int16)
            d = int((f0 != f1).sum())
            SP.add_action_effect(sp, aid, d > 0, d)
            SP.add_note(sp, f"ACTION{aid}: {'changes ' + str(d) + 'px' if d else 'NO-OP from start'}")
        except Exception as e:
            SP.add_note(sp, f"ACTION{aid}: errored ({e})")
    # clicks at the smallest objects (the v13 _dyn_clicks heuristic)
    try:
        g, f0 = fresh()
        cnt = np.bincount(f0.flatten().clip(0, 15).astype(np.int64),
                          minlength=16)
        bg = int(cnt.argmax())
        targets = sorted((int(cnt[c]), c) for c in range(16)
                         if c != bg and 0 < cnt[c] <= f0.size // 2)
        for _, c in targets[:n_clicks]:
            ys, xs = np.where(f0 == c)
            x, y = int(np.median(xs)), int(np.median(ys))
            g, _ = fresh()
            r = g.perform_action(
                ActionInput(id=GameAction.from_id(6),
                            data={'x': x, 'y': y, 'game_id': 'probe'}),
                raw=True)
            f1 = np.array(r.frame[-1], dtype=np.int16)
            d = int((f0 != f1).sum())
            SP.add_action_effect(sp, 6, d > 0, d)
            SP.add_note(sp, f"click({x},{y}) on color {c}: "
                            f"{'changed ' + str(d) + 'px' if d else 'no-op'}")
    except Exception as e:
        SP.add_note(sp, f"click probing errored ({e})")


def run_game(gid, budget_s, out_dir, bfs_timeout, strategy):
    src = find_game_source(gid)
    if not src:
        logger.warning(f"{gid}: no source")
        return
    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', open(src).read()[:2000])
    cls_name = m.group(1) if m else gid.capitalize()

    sp = SP.new_scratchpad(gid, budget_s)
    solver = BFSSolver(src, cls_name, scan_timeout=5,
                       bfs_timeout=bfs_timeout, workers=HW["workers"])
    if not solver.load():
        SP.add_note(sp, "engine failed to load")
        SP.save(sp, out_dir)
        return

    # hydrate everything already known (v13/v12 caches + old scratchpads)
    for vdir in ('v13', 'v12'):
        p = os.path.join(HERE, '..', vdir, f'{vdir}_bfs_cache_{gid}.json')
        if os.path.exists(p):
            for k, v in json.load(open(p)).items():
                solver.solutions.setdefault(int(k), [tuple(x) for x in v])
    old = SP.load(gid, out_dir)
    if old:
        for k, v in old.get("solved", {}).items():
            solver.solutions.setdefault(int(k), [tuple(x) for x in v])

    try:
        n_levels = len(getattr(solver.game_cls(), '_levels', [])) or 12
    except Exception:
        n_levels = 12
    sp["levels"] = n_levels

    probe_actions(solver.game_cls, sp)

    t0 = time.time()
    for li in range(n_levels):
        if li in solver.solutions:
            SP.add_solution(sp, li, [list(x) for x in solver.solutions[li]])
            continue
        remaining = budget_s - (time.time() - t0)
        if remaining < 10:
            sp["stuck_at"] = li
            SP.add_note(sp, f"budget exhausted before L{li} "
                            f"({budget_s}s spent) — resume me")
            break
        solver.bfs_timeout = min(bfs_timeout, remaining - 5)
        prev = solver.solutions.get(li - 1) if li > 0 else None
        fp = f"/tmp/v15_pass1_{gid}_L{li}.{strategy}.pkl"
        logger.info(f"{gid} L{li}: BFS ({solver.bfs_timeout:.0f}s slice)...")
        sol = solver.solve_level(li, prev_solution=prev,
                                 frontier_path=fp, strategy=strategy)
        if sol:
            SP.add_solution(sp, li, [list(x) for x in sol])
            SP.save(sp, out_dir)            # checkpoint per level
        else:
            sp["stuck_at"] = li
            SP.add_note(sp, f"L{li} UNSOLVED in this pass "
                            f"(frontier persisted at {fp})")
            break
    if sp["stuck_at"] is None and len(sp["solved"]) >= n_levels:
        SP.add_note(sp, "GAME COMPLETE")
    out = SP.save(sp, out_dir)
    logger.info(f"{gid}: scratchpad -> {out} "
                f"(solved {sorted(sp['solved'])}, stuck_at {sp['stuck_at']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="all")
    ap.add_argument("--budget", type=float, default=600.0,
                    help="wall-clock budget per game (s)")
    ap.add_argument("--bfs-timeout", type=float, default=120.0,
                    help="max single BFS slice within the budget")
    ap.add_argument("--strategy", choices=["bfs", "greedy"], default="bfs")
    ap.add_argument("--out", default=os.path.join(HERE, "v15_scratch"))
    args = ap.parse_args()

    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3',
                           'environment_files')
    games = sorted(os.listdir(env_dir)) if args.games == "all" \
        else args.games.split(",")
    for gid in games:
        try:
            run_game(gid, args.budget, args.out, args.bfs_timeout,
                     args.strategy)
        except Exception as e:
            logger.warning(f"{gid}: pass-1 failed ({e})")


if __name__ == "__main__":
    main()
