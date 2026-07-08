#!/usr/bin/env python3
# =====================================================================
# Standalone Mac test for the BRAIN PLANNER (Epic B3 / ladder macro-BFS) on the
# ls20 L5 wall — isolated from the full cadence so you can iterate fast.
#
#   ../../../.venv312/bin/python test_ladder_mac.py --selftest   # offline maze, no engine
#   ../../../.venv312/bin/python test_ladder_mac.py              # ls20 L5 on the real engine
#   ../../../.venv312/bin/python test_ladder_mac.py ls20 6       # game + level
#
# Tunables (env): V21_PLANNER_STATES (search budget), V21_PLANNER_MACRO (macro cap).
# =====================================================================
import os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def selftest():
    """Offline proof (no arcengine): corridor+turn maze — macro-BFS collapses each
    corridor to one edge; single-step BFS would need depth 10."""
    import ladder
    def clone(s): return dict(s)
    def play(s, step):
        a, _ = step
        if a == 2 and s["x"] < 5: s["x"] += 1
        elif a == 3 and s["x"] >= 5 and s["y"] < 5: s["y"] += 1
        return 1 if (s["x"] >= 5 and s["y"] >= 5) else 0
    def hashf(s): return (s["x"], s["y"])
    plan = ladder.macro_bfs({"x": 0, "y": 0}, clone, play, [2, 3, 4], hashf, goal=1)
    s = {"x": 0, "y": 0}; lc = 0
    for st in (plan or []): lc = play(s, st)
    ok = bool(plan) and lc >= 1
    print(f"[selftest] macro-BFS maze: plan_len={len(plan) if plan else None} wins={lc>=1} -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def engine_test(gid, lvl):
    import cadence_runner as cr
    for p in (cr.V19_SRC, cr.V20_SRC):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    from combined_agent import BFSSolver
    info = cr.resolve_source(gid)
    if not info:
        print(f"no source for {gid}"); return 1
    path, cls, ver = info
    solver = BFSSolver(path, cls, bfs_timeout=60)
    if not solver.load():
        print("solver.load() failed"); return 1

    # chain the corpus so _make_start_state(lvl) re-roots at the true L{lvl-1} end
    corpus = cr.load_corpus(gid)
    for i in range(lvl):
        if i in corpus:
            solver.solutions[i] = corpus[i]
    print(f"chained {sorted(k for k in solver.solutions)} into solver for {gid} L{lvl} (engine {ver})")
    print(f"planner budget: states={os.environ.get('V21_PLANNER_STATES','200000')} "
          f"macro={os.environ.get('V21_PLANNER_MACRO','64')}")

    t = time.time()
    plan = cr._brain_planner_for_solver(solver, lvl)
    dt = time.time() - t
    if plan and solver.verify_solution(lvl, plan):
        print(f"\n✅ ls20 L{lvl} SOLVED by brain planner: {len(plan)} actions in {dt:.0f}s")
        print(f"   plan (first 15): {plan[:15]}")
        return 0
    print(f"\n❌ brain planner did not crack {gid} L{lvl} in {dt:.0f}s "
          f"(plan={'none' if not plan else str(len(plan))+' actions, failed verify'})")
    print("   Iterate: raise V21_PLANNER_STATES / V21_PLANNER_MACRO, or pair with V21_RUNTIME_CODER.")
    return 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    gid = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else "ls20"
    lvl = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"=== brain-planner test: {gid} L{lvl} ===")
    return engine_test(gid, lvl)


if __name__ == "__main__":
    sys.exit(main())
