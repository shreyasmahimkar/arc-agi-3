"""v19 FORGE eval — black-box coverage + action-cost (RHAE proxy) baseline.

Drives the FORGE agent through v18's blackbox_env (API-only surface: frame,
levels_completed, win_levels, available_actions, state — no engine internals).
Reports per game: levels reached and actions-to-each-level (the RHAE driver,
since score ~ (human/ai_actions)^2). Same frozen TRAIN/HELD-OUT split as v18.

Usage:
  python eval_forge.py --split heldout --budget 600
  python eval_forge.py --games cn04,ls20 --budget 800
"""
from __future__ import annotations
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v18")))
from blackbox_env import BlackBoxEnv          # noqa: E402
from forge_agent import ForgeAgent            # noqa: E402

HERE = os.path.dirname(__file__)
ALL_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
             "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
             "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]
HELDOUT = ["cn04", "ka59", "sk48", "tu93", "wa30"]
TRAIN = [g for g in ALL_GAMES if g not in HELDOUT]
KNOWN_BAD = {"lf52", "tn36"}


def run_game(game, budget, seed):
    env = BlackBoxEnv(game)
    obs = env.reset()
    agent = ForgeAgent(seed=seed)
    agent.reset(game)
    start = obs.levels_completed
    best = obs.levels_completed
    level_at = {}                      # level reached -> action index
    for step in range(budget):
        aid, data = agent.act(obs)
        obs = env.step(aid, data)
        if obs.levels_completed > best:
            best = obs.levels_completed
            level_at[best - start] = step + 1
        if obs.state == "WIN":
            break
    return {"levels": best - start, "win": obs.state == "WIN",
            "level_at_action": level_at, "actions_used": min(budget, env.action_counter)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["all", "train", "heldout"], default="heldout")
    ap.add_argument("--games", default=None)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    games = args.games.split(",") if args.games else \
        {"all": ALL_GAMES, "train": TRAIN, "heldout": HELDOUT}[args.split]
    tag = args.games or args.split
    print(f"[forge-eval] games={len(games)} budget={args.budget} split={tag}")
    res, t0 = {}, time.time()
    for g in games:
        if g in KNOWN_BAD:
            res[g] = {"skipped": "load-error"}; print(f"  {g}: skipped"); continue
        try:
            r = run_game(g, args.budget, args.seed)
            res[g] = r
            first = r["level_at_action"].get(1)
            print(f"  {g}: levels={r['levels']} win={r['win']} first_level@{first} "
                  f"actions={r['actions_used']}")
        except Exception as e:
            import traceback; traceback.print_exc()
            res[g] = {"error": repr(e)[:120]}; print(f"  {g}: ERROR {repr(e)[:80]}")
    dt = round(time.time() - t0, 1)
    solved = sum(1 for v in res.values() if v.get("levels", 0) > 0)
    total = sum(v.get("levels", 0) for v in res.values())
    scored = [g for g in games if g not in KNOWN_BAD and "error" not in res[g]]
    out = {"split": tag, "budget": args.budget, "games_scored": len(scored),
           "games_with_a_level": solved, "total_levels": total, "seconds": dt, "per_game": res}
    json.dump(out, open(os.path.join(HERE, f"v19_forge_{tag}.json"), "w"), indent=2)
    print(f"[forge-eval] games_with_a_level={solved} total_levels={total} ({dt}s)")


if __name__ == "__main__":
    main()
