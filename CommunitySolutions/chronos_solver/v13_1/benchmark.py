#!/usr/bin/env python3
"""
v13 vs v13_1 head-to-head benchmark.

Runs each (version, game) pair in a fresh subprocess via _bench_runner.py —
no caches, no frontier resume, identical per-level wall budget — and writes:

  benchmark_results.json   raw per-level rows
  BENCHMARK.md             comparison table

Usage:
  python benchmark.py                                   # default suite
  python benchmark.py --games ls20:3,ar25:2 --budget 60
  python benchmark.py --versions ../v13,. --workers 6   # M1 Pro

Per-game level counts use game:levels syntax. Budget is seconds of search
per level PER VERSION (v13 splits it bfs/greedy internally, v13_1 runs its
auto ladder inside it).
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_games(spec):
    out = []
    for part in spec.split(','):
        if ':' in part:
            g, n = part.split(':')
            out.append((g.strip(), int(n)))
        else:
            out.append((part.strip(), 2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="ls20:3,ar25:2,cd82:2,bp35:1")
    ap.add_argument("--versions", default=os.path.join(HERE, '..', 'v13') + ',' + HERE,
                    help="comma-separated solver version dirs")
    ap.add_argument("--budget", type=float, default=60.0)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=300_000)
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_results.json"))
    args = ap.parse_args()

    games = parse_games(args.games)
    versions = [os.path.abspath(v) for v in args.versions.split(',')]
    runs = []
    t_start = time.time()
    for gid, nlev in games:
        for vdir in versions:
            vname = os.path.basename(vdir)
            tmp = os.path.join(HERE, f".bench_{vname}_{gid}.json")
            cmd = [sys.executable, os.path.join(HERE, "_bench_runner.py"),
                   "--version-dir", vdir, "--game", gid,
                   "--levels", str(nlev), "--budget", str(args.budget),
                   "--workers", str(args.workers),
                   "--max-states", str(args.max_states),
                   "--out", tmp]
            print(f"=== {vname} / {gid} ({nlev} levels, {args.budget}s/level) ===",
                  flush=True)
            try:
                subprocess.run(cmd, timeout=nlev * args.budget * 1.6 + 180,
                               check=False)
            except subprocess.TimeoutExpired:
                print(f"!!! {vname}/{gid} runner timed out — partial results kept",
                      flush=True)
            if os.path.exists(tmp):
                runs.append(json.load(open(tmp)))
                os.unlink(tmp)
            json.dump({"budget": args.budget, "workers": args.workers,
                       "max_states": args.max_states,
                       "elapsed_s": round(time.time() - t_start, 1),
                       "runs": runs}, open(args.out, 'w'), indent=1)
    write_md(runs, args)
    print(f"done in {time.time() - t_start:.0f}s -> {args.out} + BENCHMARK.md")


def write_md(runs, args):
    by = {}
    versions = []
    for r in runs:
        if 'error' in r:
            continue
        v = r['version']
        if v not in versions:
            versions.append(v)
        for row in r['levels']:
            by.setdefault((r['game'], row['level']), {})[v] = row
    lines = [
        "# v13 vs v13_1 benchmark",
        "",
        f"Fresh solvers, no caches/frontiers. {args.budget:.0f}s search "
        f"budget per level per version, workers={args.workers}, "
        f"max-states={args.max_states}. v13 = bfs then greedy (budget "
        "split); v13_1 = auto ladder (bfs->waypoint->astar->iw1->iw2->"
        "greedy->rescues).", "",
        "| Game | Lvl | " + " | ".join(
            f"{v} solved | {v} time | {v} acts" for v in versions) +
        " | winning rung |",
        "|---|---|" + "---|" * (3 * len(versions) + 1),
    ]
    totals = {v: [0, 0, 0.0] for v in versions}  # solved, attempted, time
    for (g, li) in sorted(by):
        cells = [g, str(li)]
        rung = ""
        for v in versions:
            row = by[(g, li)].get(v)
            if row is None:
                cells += ["-", "-", "-"]
                continue
            totals[v][1] += 1
            totals[v][2] += row['time_s']
            if row['solved']:
                totals[v][0] += 1
            cells += ["YES" if row['solved'] else "no",
                      f"{row['time_s']:.1f}s",
                      str(row['actions'] if row['actions'] else "-")]
            if row.get('strategy'):
                rung = row['strategy']
        lines.append("| " + " | ".join(cells + [rung]) + " |")
    lines += ["", "## Totals", ""]
    for v in versions:
        s, a, t = totals[v]
        lines.append(f"- **{v}**: {s}/{a} levels solved, "
                     f"{t:.0f}s total search time")
    with open(os.path.join(HERE, "BENCHMARK.md"), 'w') as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
