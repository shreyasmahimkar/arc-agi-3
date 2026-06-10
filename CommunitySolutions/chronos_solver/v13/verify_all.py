#!/usr/bin/env python3
"""
v13 replay verifier — run AFTER the solve_all sweep.

For every game with a solution cache, replays the cached solutions through
the real arc_agi environment via the agent and records the official
scorecard. Output: v13_scorecards.json + summary table on stdout.

Usage (repo root, venv312 active):
    python CommunitySolutions/chronos_solver/v13/verify_all.py
"""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    games = sorted(re.match(r'v13_bfs_cache_(\w+)\.json',
                            os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(HERE, 'v13_bfs_cache_*.json')))
    if not games:
        print("no caches found")
        return 1
    print(f"verifying {len(games)} games: {', '.join(games)}\n")
    results = {}
    env = dict(os.environ, V13_BFS_TIMEOUT="10")
    for gid in games:
        cache = json.load(open(os.path.join(HERE, f'v13_bfs_cache_{gid}.json')))
        try:
            out = subprocess.run(
                [sys.executable, os.path.join(HERE, 'play_game.py'),
                 '--game', gid, '--fast', '--max-steps', '250'],
                capture_output=True, text=True, timeout=900, env=env,
            ).stdout
            m = re.search(r'"score":\s*([\d.]+)', out)
            lv = re.findall(r'"levels_completed":\s*(\d+)', out)
            results[gid] = {
                'cached_levels': {k: len(v) for k, v in sorted(cache.items(), key=lambda x: int(x[0]))},
                'score': float(m.group(1)) if m else None,
                'levels_completed': max((int(x) for x in lv), default=None),
            }
        except Exception as e:
            results[gid] = {'error': str(e),
                            'cached_levels': {k: len(v) for k, v in cache.items()}}
        r = results[gid]
        print(f"  {gid}: score={r.get('score')} levels={r.get('levels_completed')} "
              f"cache={r['cached_levels']}")
    with open(os.path.join(HERE, 'v13_scorecards.json'), 'w') as f:
        json.dump(results, f, indent=1)
    total = sum(r.get('levels_completed') or 0 for r in results.values())
    print(f"\nTOTAL levels completed in env replay: {total}")
    print(f"saved -> {os.path.join(HERE, 'v13_scorecards.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
