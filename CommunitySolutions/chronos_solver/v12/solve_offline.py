#!/usr/bin/env python3
"""
v12 offline BFS pre-solver.

Solves levels of a game directly against the engine class (no env loop) and
persists solutions to v12_bfs_cache_<game>.json. Resumable: already-cached
levels are skipped, so it can be invoked repeatedly with a small time budget.

Usage:
    python solve_offline.py --game ls20 [--budget 35] [--bfs-timeout 60]
                            [--max-levels 10] [--level N]
"""
import argparse
import glob
import json
import logging
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))
sys.path.insert(0, HERE)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(HERE, "v12_run.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from my_agent import BFSSolver  # noqa: E402


def find_game_source(game_id):
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    matches = glob.glob(os.path.join(env_dir, game_id, "**", f"{game_id}.py"),
                        recursive=True)
    if not matches:
        # repo-root copy
        matches = glob.glob(os.path.join(REPO, 'environment_files', game_id, "**",
                                         f"{game_id}.py"), recursive=True)
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--budget", type=float, default=35.0,
                    help="overall wall-clock budget (s) for this invocation")
    ap.add_argument("--bfs-timeout", type=float, default=60.0)
    ap.add_argument("--scan-timeout", type=float, default=5.0)
    ap.add_argument("--max-levels", type=int, default=12)
    ap.add_argument("--level", type=int, default=None,
                    help="solve only this level index")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    src = find_game_source(args.game)
    if not src:
        logger.error(f"game source not found for {args.game}")
        return 1
    import re
    content = open(src).read()[:2000]
    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
    cls_name = m.group(1) if m else args.game.capitalize()

    cache_path = os.path.join(HERE, f"v12_bfs_cache_{args.game}.json")
    solver = BFSSolver(src, cls_name, scan_timeout=args.scan_timeout,
                       bfs_timeout=args.bfs_timeout, workers=args.workers)
    if not solver.load():
        return 1
    if os.path.exists(cache_path):
        cached = json.load(open(cache_path))
        for k, v in cached.items():
            solver.solutions[int(k)] = [tuple(x) for x in v]
        logger.info(f"resume: cached levels {sorted(solver.solutions)}")

    # how many levels does this game have?
    try:
        g = solver.game_cls()
        n_levels = len(getattr(g, '_levels', [])) or args.max_levels
    except Exception:
        n_levels = args.max_levels
    logger.info(f"{args.game}: {n_levels} levels, class {cls_name}")

    t0 = time.time()
    levels = [args.level] if args.level is not None else range(min(n_levels, args.max_levels))
    for li in levels:
        if li in solver.solutions:
            logger.info(f"L{li}: cached ({len(solver.solutions[li])} actions) — skip")
            continue
        remaining = args.budget - (time.time() - t0)
        if remaining < 8:
            logger.info(f"budget exhausted before L{li}; resume me")
            break
        solver.bfs_timeout = min(args.bfs_timeout, remaining - 3)
        prev = solver.solutions.get(li - 1) if li > 0 else None
        fp = f"/tmp/v12_frontier_{args.game}_L{li}.pkl"
        sol = solver.solve_level(li, prev_solution=prev, frontier_path=fp)
        if sol:
            json.dump({str(k): v for k, v in solver.solutions.items()},
                      open(cache_path, 'w'))
            logger.info(f"L{li}: saved ({len(sol)} actions)")
        else:
            logger.info(f"L{li}: unsolved this pass")
    logger.info(f"cache now: { {k: len(v) for k, v in sorted(solver.solutions.items())} }")
    return 0


if __name__ == "__main__":
    sys.exit(main())
