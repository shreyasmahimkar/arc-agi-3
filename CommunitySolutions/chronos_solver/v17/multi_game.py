"""v17 cross-game benchmark vs v13.

For each of the 25 official games: chain to v13's FRONTIER level (= number of
levels v13 cached) using v13's solutions, then run v17's GAME-AGNOSTIC search
(forward-rollout MCTS + progress shaping + macro-actions; NO ls20-specific
ForgeNet/TRM, which wouldn't transfer) and see whether it cracks the next level
v13 never solved. This isolates whether v17's *search* innovations generalise.

Usage: python multi_game.py --games ar25,bp35,...   (default: all)
Writes v17_multigame.json + a comparison table.
"""
from __future__ import annotations
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
import engine as E
import mcts
from vlog import get_logger

HERE = os.path.dirname(__file__)
ALL_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
             "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
             "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]


def v13_count(game):
    p = os.path.join(HERE, "..", "v13", f"v13_bfs_cache_{game}.json")
    if os.path.exists(p):
        return len(json.load(open(p)))
    return 0


_PUZZLE = {}


def _load_puzzle():
    if not _PUZZLE:
        import trm
        m = trm.TRM.load(os.path.join(HERE, "models", "puzzle_trm.npz"))
        _PUZZLE["trm"] = m
    return _PUZZLE["trm"]


def bench_game(game, sims, tb, lg, puzzle=False):
    frontier = v13_count(game)
    rec = {"game": game, "v13_levels": frontier, "puzzle": puzzle}
    try:
        if puzzle:
            m = _load_puzzle()
            vfn = lambda f: float(m.forward(E.object_features(f))[1][0])
            res = mcts.solve_mcts_az(game, frontier, sims=sims, time_budget=tb,
                                     c_puct=2.0, value_fn=vfn, policy_fn=m.policy_fn,
                                     macro_moves=True, macro_max=6, mask=None,
                                     micro_rollout=14, value_weight=0.5,
                                     iter_tag="multigame_pz", log_every=100000, logger=lg)
        else:
            res = mcts.solve_mcts(game, frontier, sims=sims, time_budget=tb,
                                  rollout_len=25, heuristic_fn=None, policy_fn=None,
                                  macro_moves=True, macro_max=6, mask=None,
                                  iter_tag="multigame", log_every=100000, logger=lg)
        solved = bool(res["solution"])
        rec.update({"v17_cracked_next": solved,
                    "v17_levels": frontier + (1 if solved else 0),
                    "best_progress": res["best_progress"], "sims": res["expansions"],
                    "time": res["time"]})
        lg.info(f"[MG] {game}: v13={frontier}  v17_next={'SOLVED' if solved else 'no'}  "
                f"progress={res['best_progress']} sims={res['expansions']} {res['time']}s")
    except Exception as e:
        rec.update({"error": repr(e)[:120], "v17_levels": frontier})
        lg.info(f"[MG] {game}: ERROR {repr(e)[:80]}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(ALL_GAMES))
    ap.add_argument("--sims", type=int, default=3000)
    ap.add_argument("--tb", type=float, default=9.0)
    ap.add_argument("--puzzle", action="store_true", help="use Puzzle-LM prior")
    args = ap.parse_args()
    lg = get_logger("multigame", "multigame")
    out_p = os.path.join(HERE, "v17_multigame_puzzle.json" if args.puzzle else "v17_multigame.json")
    out = json.load(open(out_p)) if os.path.exists(out_p) else {}
    for game in args.games.split(","):
        out[game] = bench_game(game, args.sims, args.tb, lg, puzzle=args.puzzle)
        json.dump(out, open(out_p, "w"), indent=2)
    # summary
    tv13 = sum(v.get("v13_levels", 0) for v in out.values())
    tv17 = sum(v.get("v17_levels", 0) for v in out.values())
    cracked = [g for g, v in out.items() if v.get("v17_cracked_next")]
    lg.info(f"[MG-SUMMARY] games={len(out)} v13_total={tv13} v17_total={tv17} "
            f"extra_cracked={len(cracked)}: {cracked}")
    print(f"v13 total levels={tv13}  v17 total levels={tv17}  extra={len(cracked)} {cracked}")


if __name__ == "__main__":
    main()
