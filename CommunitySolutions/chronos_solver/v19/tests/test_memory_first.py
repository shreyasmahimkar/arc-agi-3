#!/usr/bin/env python3
"""MEMORY-FIRST local test — exercises the V19_CACHE_FIRST sweep on the MacBook.

This is the local mirror of what the memory-first Kaggle notebook does at scoring
time: for a game whose solution is already in solutions/, the agent should clear
each level by REPLAYING the learned plan (MEMORY HIT) — no live BFS search — and
only fall through to the v13/v17 BFS ladder when a level's memory is missing/stale.

It drives the REAL agent (combined_agent.MyAgent) through one continuous episode on
the live arc_agi engine with:
    V19_CACHE_FIRST=1      memory leads
    V19_CACHE_FALLBACK=0   (irrelevant in cache-first, kept off for clarity)
    V19_STORE_SOLUTIONS=0  never rewrite the cache during the test

PASS = clears `--target` levels AND every solved level was a MEMORY HIT (proving the
memory-first sweep, not live search, did the work). Exit 0 on pass, 1 on fail.

  python test_memory_first.py                 # ls20, clear 5 levels from memory
  python test_memory_first.py --game ar25 --target 2
"""
import os, sys, time, argparse, logging

# Flags are read at import time, so set them BEFORE importing the agent.
os.environ["V19_CACHE_FIRST"] = "1"          # MEMORY-FIRST: replay learned plans
os.environ["V19_CACHE_FALLBACK"] = "0"       # not the timeout backstop path
os.environ["V19_STORE_SOLUTIONS"] = "0"      # never write the cache during a test

HERE = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(repo_root, "arc-prize-2026-arc-agi-3", "ARC-AGI-3-Agents"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))   # the v19 code package

# Capture the agent's log so we can prove HOW each level was solved.
class _Tap(logging.Handler):
    def __init__(s): super().__init__(); s.lines = []
    def emit(s, r): s.lines.append(r.getMessage())

tap = _Tap()
logging.basicConfig(level=logging.INFO, handlers=[tap])

import arc_agi
from arcengine import GameAction, GameState
from combined_agent import MyAgent, CACHE_FIRST


def run_episode(game, target, max_steps, bfs_timeout, death_cap):
    os.environ["V13_BFS_TIMEOUT"] = str(bfs_timeout)
    env_dir = os.path.join(repo_root, "arc-prize-2026-arc-agi-3", "environment_files")
    arc = arc_agi.Arcade(environments_dir=env_dir, operation_mode=arc_agi.OperationMode.OFFLINE)
    env = arc.make(game, render_mode=None)

    # Pin BFS to the env's OWN loaded version (same care as test_ls20.py).
    import glob
    class _EI:
        def __init__(s, d): s.local_dir = d
    ei = getattr(env, "environment_info", None)
    if not ei or not getattr(ei, "local_dir", None):
        m = glob.glob(os.path.join(env_dir, game, "**", f"{game}.py"), recursive=True)
        env.environment_info = _EI(os.path.dirname(m[0]) if m else os.path.join(env_dir, game))
    print(f"  source: {env.environment_info.local_dir}")

    agent = MyAgent(card_id="", game_id=game, agent_name="memfirst", ROOT_URL="",
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
                print(f"  ABORT: {deaths} GAME_OVERs (stuck at level {best})")
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
    ap.add_argument("--target", type=int, default=5,
                    help="levels to clear from memory (ls20 cache covers L0..L4)")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--bfs-timeout", type=int, default=180)
    ap.add_argument("--death-cap", type=int, default=8)
    ap.add_argument("--allow-bfs-fallthrough", action="store_true",
                    help="pass even if some levels needed live BFS (memory miss/stale)")
    args = ap.parse_args()

    assert CACHE_FIRST, "V19_CACHE_FIRST did not take effect — check import order"
    print(f"=== MEMORY-FIRST: {args.game} should clear {args.target} levels from cache ===")
    best, steps, secs = run_episode(args.game, args.target, args.max_steps,
                                    args.bfs_timeout, args.death_cap)

    hits = sum(1 for l in tap.lines if l.startswith("MEMORY HIT"))
    stale = sum(1 for l in tap.lines if l.startswith("MEMORY STALE"))
    miss = sum(1 for l in tap.lines if l.startswith("MEMORY MISS"))
    print(f"\n  memory: {hits} HIT, {stale} STALE, {miss} MISS")
    reached = best >= args.target
    pure_memory = hits >= best and stale == 0 and miss == 0
    ok = reached and (pure_memory or args.allow_bfs_fallthrough)
    print(f"RESULT: level {best}/{args.target} | steps={steps} | {secs:.0f}s | "
          f"{'pure-memory' if pure_memory else 'used BFS fallthrough'} "
          f"-> {'PASS ✅' if ok else 'FAIL ❌'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
