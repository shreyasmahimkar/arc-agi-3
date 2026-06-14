#!/usr/bin/env python3
"""
v13 multi-game offline pre-solver — designed to run UNINTERRUPTED on a real
machine (e.g. M1 Pro, 8 workers). Loops over every game in environment_files,
solves levels with resumable BFS/greedy search, and persists:

  - v13_3_bfs_cache_<game>.json   (solutions; consumed by my_agent at runtime)
  - v13_3_progress.json           (live status summary, refreshed continuously)
  - v13_3_run.log                 (full log)

Safe to Ctrl-C and relaunch any time: solved levels are skipped, in-progress
level searches resume from their frontier checkpoints.

Usage (from repo root, venv312 active):
    python CommunitySolutions/chronos_solver/v13/solve_all.py
    python CommunitySolutions/chronos_solver/v13/solve_all.py --games ls20,ar25
    python CommunitySolutions/chronos_solver/v13/solve_all.py --level-budget 900
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))
sys.path.insert(0, HERE)

# [v13] the offline solver is pure CPU. Hide the GPU BEFORE importing
# my_agent: (a) the HW probe would otherwise initialize a CUDA context and
# the BFS worker pools fork() afterwards — fork-after-CUDA-init is unsafe;
# (b) the CPU profile maximizes worker count on big multi-core GPU boxes.
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(os.path.join(HERE, "v13_3_run.log")),
              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

from my_agent import BFSSolver, HW  # noqa: E402

PROGRESS = os.path.join(HERE, "v13_3_progress.json")


def all_games():
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    return sorted(d for d in os.listdir(env_dir)
                  if os.path.isdir(os.path.join(env_dir, d)))


def game_source(game_id):
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    m = glob.glob(os.path.join(env_dir, game_id, "**", f"{game_id}.py"),
                  recursive=True)
    return m[0] if m else None


def update_progress(state):
    state['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    try:
        with open(PROGRESS, 'w') as f:
            json.dump(state, f, indent=1)
    except Exception:
        pass


def solve_game(game_id, args, progress):
    src = game_source(game_id)
    if not src:
        progress[game_id] = {"status": "no-source"}
        return
    content = open(src).read()[:2000]
    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
    cls_name = m.group(1) if m else game_id.capitalize()

    cache_path = os.path.join(HERE, f"v13_3_bfs_cache_{game_id}.json")
    solver = BFSSolver(src, cls_name, scan_timeout=args.scan_timeout,
                       bfs_timeout=args.level_budget, workers=args.workers)
    if not solver.load():
        progress[game_id] = {"status": "load-failed"}
        return
    if os.path.exists(cache_path):
        for k, v in json.load(open(cache_path)).items():
            solver.solutions[int(k)] = [tuple(x) for x in v]

    try:
        g = solver.game_cls()
        n_levels = len(getattr(g, '_levels', [])) or 10
    except Exception as e:
        progress[game_id] = {"status": f"init-failed: {e}"}
        return

    st = progress.setdefault(game_id, {})
    st.update(status="solving", levels=n_levels,
              solved={str(k): len(v) for k, v in sorted(solver.solutions.items())})
    update_progress(progress)

    for li in range(n_levels):
        if li in solver.solutions:
            continue
        # [v13_1] 'auto' runs the full rung ladder internally (bfs ->
        # waypoint -> astar -> iw1 -> iw2 -> greedy -> unmasked/hidden)
        for strategy in ("auto",):
            fp = f"/tmp/v13_3_frontier_{game_id}_L{li}.{strategy}.pkl"
            solver.bfs_timeout = args.level_budget
            t0 = time.time()
            sol = solver.solve_level(li, prev_solution=solver.solutions.get(li - 1),
                                     frontier_path=fp, strategy=strategy,
                                     max_states=5_000_000)
            if sol:
                json.dump({str(k): v for k, v in solver.solutions.items()},
                          open(cache_path, 'w'))
                st['solved'][str(li)] = len(sol)
                logger.info(f"[{game_id}] L{li}: saved ({len(sol)} actions, {strategy})")
                for stale in glob.glob(f"/tmp/v13_3_frontier_{game_id}_L{li}.*"):
                    try: os.unlink(stale)
                    except OSError: pass
                update_progress(progress)
                break
            logger.info(f"[{game_id}] L{li}: {strategy} pass exhausted budget "
                        f"({time.time()-t0:.0f}s)")
        else:
            st['stuck_at'] = li
            logger.info(f"[{game_id}] stuck at L{li} — moving to next game "
                        f"(frontiers checkpointed; rerun to continue)")
            update_progress(progress)
            return
    st['status'] = 'complete'
    update_progress(progress)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="all",
                    help="comma-separated game ids, or 'all'")
    ap.add_argument("--level-budget", type=float, default=600.0,
                    help="seconds of search per level per strategy per run")
    ap.add_argument("--scan-timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=HW["workers"])
    ap.add_argument("--rounds", type=int, default=3,
                    help="how many times to sweep all games (stuck levels get "
                         "another budget each round)")
    args = ap.parse_args()

    games = all_games() if args.games == "all" else args.games.split(",")
    logger.info(f"solve_all: {len(games)} games, workers={args.workers}, "
                f"budget={args.level_budget}s/level, HW={HW['mode']}")

    progress = {}
    if os.path.exists(PROGRESS):
        try:
            progress = json.load(open(PROGRESS))
        except Exception:
            progress = {}

    for rnd in range(args.rounds):
        logger.info(f"===== ROUND {rnd + 1}/{args.rounds} =====")
        pending = [g for g in games
                   if progress.get(g, {}).get('status') != 'complete']
        if not pending:
            break
        for gid in pending:
            logger.info(f"=== {gid} ===")
            try:
                solve_game(gid, args, progress)
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.warning(f"[{gid}] error: {e}")
                progress.setdefault(gid, {})['status'] = f"error: {e}"
                update_progress(progress)
    logger.info("solve_all finished. Summary:")
    for gid in games:
        logger.info(f"  {gid}: {progress.get(gid, {})}")


if __name__ == "__main__":
    main()
