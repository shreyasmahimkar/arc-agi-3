#!/usr/bin/env python3
"""v19 WM-attempt step (ExIt step 5) — let the world model crack the frontier.

For games that have a corpus, attempt the NEXT unsolved level (the one BFS gave
up on) with the WM-imagination planner. A solve is verified on the real engine
and appended to the corpus (so it feeds the next harvest+retrain — the ExIt
flywheel). Logs every attempt to PLANNER_LOG.md (v17-style improvement tracking).

Bounded by design (few games, small budget) so it fits inside a loop cycle; as
the WM improves each cycle it cracks more. Honors V19_STORE_SOLUTIONS.
"""
from __future__ import annotations
import os, sys, json, glob, random, argparse
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wm_planner import WMPlanner, solve_level_wm

HERE = os.path.dirname(os.path.abspath(__file__))
SOL = os.path.join(HERE, "solutions")
LOG = os.path.join(HERE, "PLANNER_LOG.md")
STORE = os.environ.get("V19_STORE_SOLUTIONS", "1") == "1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=8, help="how many frontier games to try")
    ap.add_argument("--budget", type=int, default=200, help="real actions per attempt")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    corpora = {}
    for f in glob.glob(os.path.join(SOL, "*.json")):
        try:
            d = json.load(open(f))
            if d:
                corpora[os.path.basename(f)[:-5]] = d
        except Exception:
            pass
    if not corpora:
        print("[wm_attempt] no corpus yet"); return
    planner = WMPlanner()
    if not planner.loaded:
        print("[wm_attempt] no wm_weights.pt yet — skip"); return

    games = list(corpora); rng.shuffle(games); games = games[:args.games]
    solved_new = []
    for game in games:
        corpus = corpora[game]
        nxt = max(int(k) for k in corpus) + 1
        try:
            res = solve_level_wm(game, nxt, corpus, planner, budget=args.budget)
        except Exception as e:
            print(f"  {game} L{nxt}: ERROR {repr(e)[:60]}"); continue
        if res["solved"]:
            corpus[str(nxt)] = res["path"]
            if STORE:
                json.dump(corpus, open(os.path.join(SOL, f"{game}.json"), "w"))
            solved_new.append(f"{game} L{nxt} ({res['actions']} acts)")
            print(f"  {game} L{nxt}: WM-SOLVED in {res['actions']} actions ✓")
        else:
            print(f"  {game} L{nxt}: not solved ({res['actions']} acts)")

    first = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        if first:
            f.write("# v19 WM-imagination planner log (v17-style)\n\n")
            f.write("Levels the WORLD MODEL cracked that BFS could not — the ExIt "
                    "payoff. Grows as the WM improves each cycle.\n\n")
            f.write("| time | games tried | NEW WM-solves |\n|---|---|---|\n")
        new = (", ".join(solved_new)) if solved_new else "—"
        f.write(f"| {datetime.now():%m-%d %H:%M} | {len(games)} | {new} |\n")
    print(f"[wm_attempt] {len(solved_new)} new WM-solve(s): {solved_new}")


if __name__ == "__main__":
    main()
