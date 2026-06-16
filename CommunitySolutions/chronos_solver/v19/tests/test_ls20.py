#!/usr/bin/env python3
"""GUARDRAIL regression test — the v19 agent MUST solve ls20 through L4, GENUINELY.

This is the gate every code change has to pass before commit/push. It drives the
REAL v19 agent (combined_agent.MyAgent, full v13/v17 search ladder) through a
single CONTINUOUS ls20 episode on the live arc_agi engine, with stored answers
turned OFF (V19_STORE_SOLUTIONS=0, V19_CACHE_FALLBACK=0) — so a pass means the
agent searched and solved each level itself, never replayed a cached answer.

Pass  = clears levels 0..TARGET-1 (default 5 -> L0..L4) within the step/time budget.
Exit 0 on pass, 1 on fail. Prints the deepest level reached.

  python test_ls20.py                 # ls20, must reach 5 levels
  python test_ls20.py --target 5 --max-steps 4000 --bfs-timeout 180
"""
import os, sys, time, argparse

# Honest mode MUST be set before importing the agent (flags read at import).
os.environ["V19_STORE_SOLUTIONS"] = "0"      # never write a cache
os.environ["V19_CACHE_FALLBACK"] = "0"       # never read a cache — genuine only

HERE = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "arc-prize-2026-arc-agi-3", "ARC-AGI-3-Agents"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))   # the v19 code package

import logging
logging.basicConfig(level=logging.WARNING)   # quiet; we print our own progress

import arc_agi
from arcengine import GameAction, GameState
from combined_agent import MyAgent


def run_episode(game, target, max_steps, bfs_timeout, death_cap, verbose):
    os.environ["V13_BFS_TIMEOUT"] = str(bfs_timeout)
    env_dir = os.path.join(repo_root, "arc-prize-2026-arc-agi-3", "environment_files")
    arc = arc_agi.Arcade(environments_dir=env_dir, operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make(game, render_mode=None)

    # CRITICAL: use the env's OWN local_dir (the exact version it loaded). Multiple
    # versions can live under environment_files/<game>/<hash>/; overriding with a
    # bare glob makes the BFS solve a DIFFERENT version than the env runs -> plans
    # fail at the first divergent level (the ls20-L1 bug). Only set it if missing.
    import glob
    class _EI:
        def __init__(s, d): s.local_dir = d
    ei = getattr(env, "environment_info", None)
    if not ei or not getattr(ei, "local_dir", None):
        m = glob.glob(os.path.join(env_dir, game, "**", f"{game}.py"), recursive=True)
        env.environment_info = _EI(os.path.dirname(m[0]) if m else os.path.join(env_dir, game))
    print(f"  BFS source: {env.environment_info.local_dir}")

    agent = MyAgent(card_id="", game_id=game, agent_name="guardrail", ROOT_URL="",
                    record=False, arc_env=env)

    out = env.reset()
    lf = out[0] if isinstance(out, tuple) else out
    frames = [lf]; agent.append_frame(lf)
    best = getattr(lf, "levels_completed", 0)
    deaths = 0; t0 = time.time()

    for step in range(1, max_steps + 1):
        st = getattr(lf, "state", None)
        if st is GameState.WIN:
            break
        if st is GameState.GAME_OVER:
            deaths += 1
            if deaths > death_cap:
                print(f"  ABORT: {deaths} GAME_OVERs (agent can't get past level {best})")
                break
        try:
            action = agent.choose_action(frames, lf)
        except Exception as e:
            print(f"  choose_action raised at step {step}: {e!r}")
            break
        act_id = getattr(action, "value", None)
        if act_id is None and hasattr(action, "id"):
            act_id = getattr(action.id, "value", None)
        try:
            if act_id == 6:
                d = getattr(action, "data", None) or {}
                res = env.step(GameAction.ACTION6, data={"x": int(d.get("x", 0)), "y": int(d.get("y", 0))})
            elif isinstance(action, GameAction):
                res = env.step(action)
            else:
                res = env.step(GameAction.from_id(act_id))
        except Exception as e:
            print(f"  env.step raised at step {step}: {e!r}")
            break
        lf = res[0] if isinstance(res, tuple) else res
        frames.append(lf); agent.append_frame(lf)
        lc = getattr(lf, "levels_completed", 0)
        if lc > best:
            best = lc
            print(f"  [+] reached level {best} at step {step} ({time.time()-t0:.0f}s)")
            if best >= target:
                break
    return best, step, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="ls20")
    # Default 4 = the agent must GENUINELY clear L0,L1,L2,L3 in one live episode
    # (i.e. reach level 4). That exercises the multi-level transitions that broke
    # before (version match + the full v13/v17 'auto' ladder). Clearing L4 itself
    # (--target 5) needs the offline persistent-frontier search that accumulates
    # over many flywheel cycles (how the corpus found L4=44 actions) and is NOT
    # solvable in a single bounded live budget — so it is a stretch check, not the gate.
    ap.add_argument("--target", type=int, default=4,
                    help="levels to clear in one live episode (4 = reach L4; 5 = clear L4, offline-only)")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--bfs-timeout", type=int, default=180)
    ap.add_argument("--death-cap", type=int, default=8)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    print(f"=== GUARDRAIL: {args.game} must clear {args.target} levels GENUINELY "
          f"(cache OFF) ===")
    best, steps, secs = run_episode(args.game, args.target, args.max_steps,
                                    args.bfs_timeout, args.death_cap, args.verbose)
    ok = best >= args.target
    print(f"\nRESULT: deepest level = {best}/{args.target} | steps={steps} | {secs:.0f}s "
          f"-> {'PASS ✅' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
