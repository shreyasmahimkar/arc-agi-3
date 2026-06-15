#!/usr/bin/env python3
"""v19 solving campaign — solve EVERY game x level with the combined BFS ladder,
persist to the v19 corpus (solutions/<game>.json), fully resumable.

The corpus is (a) the fast-rerun cache for combined_agent (V19_STORE_SOLUTIONS=1),
and (b) the training signal for the PLM world model (solutions -> transitions).

Run it in escalating passes (cheap timeout first, then up to 30 min for the
stubborn levels) — each pass only attempts still-unsolved levels:
    python solve_all.py --bfs-timeout 180     # pass 1: grab the easy levels fast
    python solve_all.py --bfs-timeout 600      # pass 2
    python solve_all.py --bfs-timeout 1800     # pass 3: 30-min cap per level
"""
import argparse, glob, json, logging, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))
sys.path.insert(0, HERE)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')        # CPU BFS; safe fork pools

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.FileHandler(os.path.join(HERE, "solve_all.log")),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("solve_all")

from combined_agent import BFSSolver, HW, SOLUTIONS_DIR, _sol_path  # noqa: E402

# Roots scanned for game engine sources. Drop the 200-game testbed under any of
# these (or set V19_EXTRA_GAMES_DIR) and the whole pipeline picks them up — no
# hardcoded game list.
GAME_ROOTS = [
    os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files'),
    os.path.join(REPO, 'environment_files'),
]
if os.environ.get('V19_EXTRA_GAMES_DIR'):
    GAME_ROOTS.insert(0, os.environ['V19_EXTRA_GAMES_DIR'])


def discover_games():
    """Every game whose <id>/**/<id>.py engine source is reachable, sorted."""
    found = {}
    for base in GAME_ROOTS:
        if not os.path.isdir(base):
            continue
        for d in sorted(os.listdir(base)):
            if d in found:
                continue
            m = glob.glob(os.path.join(base, d, "**", f"{d}.py"), recursive=True)
            if m:
                found[d] = m[0]
    return found


def find_game_source(game_id):
    return discover_games().get(game_id)


ALL_GAMES = sorted(discover_games().keys())


def solve_game(game, bfs_timeout, max_levels, workers, max_states):
    src = find_game_source(game)
    if not src:
        logger.info(f"[{game}] no source — skip"); return (game, "no-source", {})
    cls = (re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', open(src).read()[:2000]) or [None, game.capitalize()])
    cls_name = cls.group(1) if hasattr(cls, "group") else game.capitalize()
    solver = BFSSolver(src, cls_name, scan_timeout=5, bfs_timeout=bfs_timeout, workers=workers)
    if not solver.load():
        logger.info(f"[{game}] load failed — skip"); return (game, "load-fail", {})
    os.makedirs(SOLUTIONS_DIR, exist_ok=True)
    cp = _sol_path(game)
    if os.path.exists(cp):
        for k, v in json.load(open(cp)).items():
            solver.solutions[int(k)] = [tuple(x) for x in v]
    try:
        n_levels = len(getattr(solver.game_cls(), '_levels', [])) or max_levels
    except Exception:
        n_levels = max_levels
    for li in range(min(n_levels, max_levels)):
        if li in solver.solutions:
            continue
        prev = solver.solutions.get(li - 1) if li > 0 else None
        fp = f"/tmp/v19_frontier_{game}_L{li}.pkl"
        t = time.time()
        sol = solver.solve_level(li, prev_solution=prev, frontier_path=fp,
                                 strategy='auto', max_states=max_states)
        dt = time.time() - t
        if sol:
            json.dump({str(k): v for k, v in solver.solutions.items()}, open(cp, 'w'))
            logger.info(f"[{game}] L{li} SOLVED {len(sol)} actions ({dt:.0f}s)")
            for stale in glob.glob(f"/tmp/v19_frontier_{game}_L{li}.*"):
                try: os.unlink(stale)
                except OSError: pass
        else:
            logger.info(f"[{game}] L{li} unsolved ({dt:.0f}s) — stop game this pass")
            break
    return (game, "ok", {k: len(v) for k, v in sorted(solver.solutions.items())})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(ALL_GAMES))
    ap.add_argument("--bfs-timeout", type=float, default=180.0, help="cap per level (s)")
    ap.add_argument("--max-levels", type=int, default=12)
    ap.add_argument("--workers", type=int, default=HW["workers"])
    ap.add_argument("--max-states", type=int, default=5_000_000)
    ap.add_argument("--shuffle", action="store_true",
                    help="randomise game order (so a time-boxed cycle rotates "
                         "across all games instead of always retrying the first few)")
    args = ap.parse_args()
    games = args.games.split(",")
    if args.shuffle:
        import random as _r
        # prioritise games with NO solved levels yet, then shuffle, so every
        # cycle makes progress on fresh games (frontier checkpoints resume the
        # in-progress ones across cycles).
        import glob as _g, json as _j
        solved = set()
        for f in _g.glob(os.path.join(SOLUTIONS_DIR, "*.json")):
            try:
                if _j.load(open(f)):
                    solved.add(os.path.basename(f)[:-5])
            except Exception:
                pass
        fresh = [x for x in games if x not in solved]
        done = [x for x in games if x in solved]
        _r.shuffle(fresh); _r.shuffle(done)
        games = fresh + done
    logger.info(f"[campaign] {len(games)} games, cap={args.bfs_timeout}s/level, workers={args.workers}")
    t0 = time.time(); summary = {}
    for g in games:
        try:
            _, status, levels = solve_game(g, args.bfs_timeout, args.max_levels,
                                           args.workers, args.max_states)
            summary[g] = {"status": status, "levels": levels, "n": len(levels)}
        except Exception as e:
            import traceback; traceback.print_exc()
            summary[g] = {"status": f"ERR {repr(e)[:60]}", "levels": {}, "n": 0}
    total = sum(v["n"] for v in summary.values())
    solved_games = sum(1 for v in summary.values() if v["n"] > 0)
    json.dump(summary, open(os.path.join(HERE, "solve_all_summary.json"), "w"), indent=2)
    logger.info(f"[campaign] DONE {time.time()-t0:.0f}s — {total} levels across "
                f"{solved_games} games. summary -> solve_all_summary.json")


if __name__ == "__main__":
    main()
