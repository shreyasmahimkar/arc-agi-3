"""v18 held-out evaluation — THE metric.

Generalization is measured the only honest way: train on TRAIN games (offline),
then run the SAME agent black-box on HELD-OUT games it never saw, through the
Obs interface, within a fixed action budget. We report levels_completed per game
and the TRAIN-vs-HELDOUT aggregate. If held-out ~= train, it generalises; if
held-out collapses, the agent memorised.

This file does NOT train anything — it only scores an agent object. Learned
agents plug in via agent.make_agent(...). Per-game budget mirrors the official
harness ethos (each real action counts).

Usage:
  python evaluate.py --agent reactive --budget 200 --episodes 2 --split heldout
  python evaluate.py --agent reactive --split all
"""
from __future__ import annotations
import os, sys, json, argparse, time
sys.path.insert(0, os.path.dirname(__file__))
from blackbox_env import BlackBoxEnv, RESET
import agent as A

HERE = os.path.dirname(__file__)

ALL_GAMES = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59",
             "lf52", "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26",
             "sc25", "sk48", "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]

# Fixed, frozen split. The agent NEVER trains on HELDOUT. Rotate folds later for
# cross-validation, but keep this fold reproducible so iterations compare.
HELDOUT = ["cn04", "ka59", "sk48", "tu93", "wa30"]
TRAIN = [g for g in ALL_GAMES if g not in HELDOUT]

# games v17/v13 can't even instantiate (unpicklable / load errors); skip cleanly
KNOWN_BAD = {"lf52", "tn36"}


def run_episode(agent, game: str, budget: int):
    env = BlackBoxEnv(game)
    # Search agents drive the env themselves (reset+replay) within an action
    # budget; they own the loop. Step-wise agents use the act() loop below.
    if getattr(agent, "drives_env", False):
        r = agent.solve(env, budget)
        return {"levels": r.get("levels", 0), "win": r.get("win", False),
                "actions_to_first_level": r.get("actions_to_first_level"),
                "budget": budget, "actions_used": r.get("actions_used"),
                "solution_len": r.get("solution_len"),
                "sims": r.get("sims")}
    obs = env.reset()
    agent.reset(game)
    start_levels = obs.levels_completed
    best_levels = obs.levels_completed
    first_level_action = None
    for step in range(budget):
        act_id, data = agent.act(obs)
        obs = env.step(act_id, data)
        if obs.levels_completed > best_levels:
            best_levels = obs.levels_completed
            if first_level_action is None and best_levels > start_levels:
                first_level_action = step + 1
        if obs.state == "WIN":
            break
    return {"levels": best_levels - start_levels, "win": obs.state == "WIN",
            "actions_to_first_level": first_level_action, "budget": budget}


def evaluate(agent_name, games, budget, episodes, agent_kwargs=None):
    agent_kwargs = agent_kwargs or {}
    results = {}
    for game in games:
        if game in KNOWN_BAD:
            results[game] = {"skipped": "load-error"}
            print(f"  {game}: skipped (known load error)")
            continue
        best = {"levels": 0, "win": False, "actions_to_first_level": None}
        try:
            for ep in range(episodes):
                ag = A.make_agent(agent_name, seed=ep, **agent_kwargs)
                r = run_episode(ag, game, budget)
                if (r["levels"], r["win"]) > (best["levels"], best["win"]):
                    best = r
            results[game] = best
            extra = ""
            if best.get("solution_len") is not None:
                extra = (f" sol_len={best['solution_len']} "
                         f"search_actions={best.get('actions_used')}")
            print(f"  {game}: levels={best['levels']} win={best['win']} "
                  f"first@{best['actions_to_first_level']}{extra}")
        except Exception as e:
            results[game] = {"error": repr(e)[:120]}
            print(f"  {game}: ERROR {repr(e)[:80]}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="reactive")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--split", choices=["all", "train", "heldout"], default="heldout")
    ap.add_argument("--games", default=None,
                    help="comma-separated game ids; overrides --split (e.g. ls20)")
    ap.add_argument("--exclude_self", action="store_true",
                    help="clone agent: drop the played game's own harvest samples "
                         "(score a TRAIN game as if held-out)")
    args = ap.parse_args()
    agent_kwargs = {"exclude_self": True} if (args.exclude_self and args.agent == "clone") else {}

    if args.games:
        games = args.games.split(",")
        args.split = args.games
    else:
        games = {"all": ALL_GAMES, "train": TRAIN, "heldout": HELDOUT}[args.split]
    print(f"[v18-eval] agent={args.agent} split={args.split} games={len(games)} "
          f"budget={args.budget} episodes={args.episodes}")
    t0 = time.time()
    res = evaluate(args.agent, games, args.budget, args.episodes, agent_kwargs)
    dt = round(time.time() - t0, 1)

    solved = sum(1 for v in res.values() if v.get("levels", 0) > 0)
    total_levels = sum(v.get("levels", 0) for v in res.values())
    scored = [g for g in games if "skipped" not in res[g] and "error" not in res[g]]
    summary = {"agent": args.agent, "split": args.split, "budget": args.budget,
               "episodes": args.episodes, "games_scored": len(scored),
               "games_with_a_level": solved, "total_levels": total_levels,
               "seconds": dt, "per_game": res}
    out_p = os.path.join(HERE, f"v18_eval_{args.agent}_{args.split}.json")
    json.dump(summary, open(out_p, "w"), indent=2)
    print(f"[v18-eval] scored={len(scored)}  games_with_a_level={solved}  "
          f"total_levels={total_levels}  ({dt}s)  -> {os.path.basename(out_p)}")


if __name__ == "__main__":
    main()
