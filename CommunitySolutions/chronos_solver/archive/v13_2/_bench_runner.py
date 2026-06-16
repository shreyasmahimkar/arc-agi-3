#!/usr/bin/env python3
"""
v13_1 benchmark runner — ONE (solver-version, game) pair per process.

Imports my_agent from the given version dir, builds a FRESH solver (no
cache hydration, no frontier resume) and solves levels 0..N-1 under a
fixed per-level wall budget. Emits JSON to --out.

Fairness contract (same total search budget per level per version):
  v13   : bfs pass with budget/2, then greedy pass with budget/2
          (mirrors solve_all.py's bfs->greedy loop)
  v13_1 : strategy='auto' with the full budget (ladder is internal)

Invoked by benchmark.py; can also be run by hand.
"""
import argparse
import glob
import json
import logging
import os
import re
import sys
import time

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version-dir", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--levels", type=int, default=2)
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=300_000)
    ap.add_argument("--scan-timeout", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--level", type=int, default=None,
                    help="solve ONLY this level (chunked driving)")
    ap.add_argument("--cache", default=None,
                    help="JSON file persisting solutions across invocations "
                         "(keeps true-baseline chaining when running one "
                         "level per process)")
    args = ap.parse_args()

    vdir = os.path.abspath(args.version_dir)
    version = os.path.basename(vdir)
    repo = os.path.abspath(os.path.join(vdir, '..', '..', '..'))
    sys.path.insert(0, os.path.join(repo, 'arc-prize-2026-arc-agi-3',
                                    'ARC-AGI-3-Agents'))
    sys.path.insert(0, vdir)

    logging.basicConfig(level=logging.INFO,
                        format=f"[{version}/{args.game}] %(message)s",
                        stream=sys.stdout)

    # capture per-level explored counts from solver log lines (v13 doesn't
    # return stats from solve_level; v13_1 has level_stats but log-parse
    # works uniformly for both)
    explored_seen = {}

    class _Cap(logging.Handler):
        def emit(self, rec):
            m = re.search(r"L(\d+).*?\((\d+) explored", rec.getMessage())
            if m:
                li, n = int(m.group(1)), int(m.group(2))
                explored_seen[li] = explored_seen.get(li, 0) + n

    from my_agent import BFSSolver  # noqa: E402
    logging.getLogger('my_agent').addHandler(_Cap())

    env_dir = os.path.join(repo, 'arc-prize-2026-arc-agi-3',
                           'environment_files')
    srcs = glob.glob(os.path.join(env_dir, args.game, "**",
                                  f"{args.game}.py"), recursive=True)
    if not srcs:
        json.dump({"version": version, "game": args.game,
                   "error": "no-source"}, open(args.out, 'w'))
        return 1
    src = srcs[0]
    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', open(src).read())
    cls_name = m.group(1) if m else args.game.capitalize()

    solver = BFSSolver(src, cls_name, scan_timeout=args.scan_timeout,
                       bfs_timeout=args.budget, workers=args.workers)
    if not solver.load():
        json.dump({"version": version, "game": args.game,
                   "error": "load-failed"}, open(args.out, 'w'))
        return 1

    is_v13_1 = hasattr(solver, 'level_stats')
    if args.cache and os.path.exists(args.cache):
        for k, v in json.load(open(args.cache)).items():
            solver.solutions[int(k)] = [tuple(x) for x in v]
    results = []
    levels = [args.level] if args.level is not None else list(range(args.levels))
    for li in levels:
        explored_seen.pop(li, None)
        t0 = time.time()
        sol = None
        try:
            if is_v13_1:
                solver.bfs_timeout = args.budget
                sol = solver.solve_level(
                    li, prev_solution=solver.solutions.get(li - 1),
                    frontier_path=None, strategy='auto',
                    max_states=args.max_states)
            else:
                for strat in ("bfs", "greedy"):
                    solver.bfs_timeout = args.budget / 2
                    sol = solver.solve_level(
                        li, prev_solution=solver.solutions.get(li - 1),
                        frontier_path=None, strategy=strat,
                        max_states=args.max_states)
                    if sol:
                        break
        except Exception as e:
            logging.warning(f"L{li} error: {e}")
        elapsed = time.time() - t0
        row = {"level": li, "solved": bool(sol),
               "actions": len(sol) if sol else None,
               "time_s": round(elapsed, 1),
               "explored": explored_seen.get(li)}
        if is_v13_1:
            st = solver.level_stats.get(li, {})
            row["strategy"] = st.get('strategy')
            row["rungs"] = st.get('rungs')
        results.append(row)
        logging.info(f"L{li}: {'SOLVED ' + str(len(sol)) + ' actions' if sol else 'unsolved'} "
                     f"in {elapsed:.1f}s")
        # write incrementally so the orchestrator can show progress
        json.dump({"version": version, "game": args.game,
                   "budget_per_level": args.budget, "levels": results},
                  open(args.out, 'w'), indent=1)
        if args.cache and sol:
            json.dump({str(k): v for k, v in solver.solutions.items()},
                      open(args.cache, 'w'))
    return 0


if __name__ == "__main__":
    sys.exit(main())
