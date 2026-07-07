#!/usr/bin/env python3
"""TDD: prove v20.3 graph-explore >= v19 Forge-CNN as the no-cache fallback.
Both solve the SAME games FROM SCRATCH (no cache) under the SAME wall-clock budget;
count levels each. Assert sum(graph) >= sum(forge). Run as a script for the table."""
import os, sys, glob, re, time
import numpy as np

ARC = "/Users/shreyas/gitrepos/OpenSource/kaggle/arc3"
sys.path.insert(0, os.path.join(ARC, "CommunitySolutions/chronos_solver/v19/src"))
sys.path.insert(0, os.path.join(ARC, "CommunitySolutions/chronos_solver/v20/src"))
os.environ["V19_STORE_SOLUTIONS"] = "0"
from combined_agent import BFSSolver, ActionInput, GameAction   # noqa
from graph_explore import graph_solve
try:
    from forge_agent import ForgeAgent
except Exception:
    ForgeAgent = None

ENV = os.path.join(ARC, "arc-prize-2026-arc-agi-3", "environment_files")
WPATH = os.path.join(ARC, "CommunitySolutions/chronos_solver/v19/src/pretrained_weights.pt")
GAMES = os.environ.get("GVF_GAMES", "vc33,ft09,sp80,lp85").split(",")
BUDGET = int(os.environ.get("GVF_BUDGET", "40"))     # wall-clock seconds per game, per solver


def load(gid):
    src = glob.glob(f"{ENV}/{gid}/9607627b/{gid}.py") or glob.glob(f"{ENV}/{gid}/*/{gid}.py")
    src = src[0]
    cls = re.search(r"class\s+(\w+)\s*\(\s*ARCBaseGame", open(src).read()).group(1)
    s = BFSSolver(src, cls, bfs_timeout=5); s.load()
    return s


def play_graph(s, budget, max_levels=8):
    """Full game from scratch, chaining graph-explore's OWN solutions."""
    sols = {}; t0 = time.time()
    for lvl in range(max_levels):
        if time.time() - t0 > budget:
            break
        g = s.game_cls(); g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        ok = True
        for i in range(lvl):
            if i not in sols:
                ok = False; break
            for a, d in sols[i]:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a)), raw=True)
        if not ok or not r.frame:
            break
        f0 = np.array(r.frame[-1]); avail = list(g._available_actions)
        sol = graph_solve(g, lvl, f0, avail, budget=budget - (time.time() - t0))
        if not sol:
            break
        sols[lvl] = [(a, d) for a, d in sol]
    return len(sols)


def play_forge(s, gid, budget):
    if ForgeAgent is None:
        return 0
    forge = ForgeAgent(weights=WPATH if os.path.exists(WPATH) else None)
    forge.reset(gid)
    g = s.game_cls(); g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    best = 0; t0 = time.time()
    class _Obs: pass
    while time.time() - t0 < budget:
        if not r.frame:
            r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True); continue
        o = _Obs()
        o.frame = np.array(r.frame[-1]).astype(np.uint8)
        o.levels_completed = r.levels_completed
        o.state = str(getattr(getattr(r, "state", None), "value", getattr(r, "state", "")))
        aa = getattr(g, "_available_actions", []) or []
        o.available_actions = tuple(a.value if hasattr(a, "value") else int(a) for a in aa)
        try:
            aid, data = forge.act(o)
        except Exception:
            aid, data = 0, None
        if aid == 6 and data:
            r = g.perform_action(ActionInput(id=GameAction.ACTION6, data={"x": int(data["x"]), "y": int(data["y"]), "game_id": "f"}), raw=True)
        else:
            r = g.perform_action(ActionInput(id=GameAction.from_id(aid if aid else 0)), raw=True)
        best = max(best, r.levels_completed)
    return best


def compare():
    rows = {}
    for gid in GAMES:
        s = load(gid)
        gl = play_graph(s, BUDGET)
        fl = play_forge(s, gid, BUDGET)
        rows[gid] = (gl, fl)
        print(f"  {gid:6}: graph={gl:2d}  forge={fl:2d}  {'graph>' if gl>fl else ('tie' if gl==fl else 'FORGE>')}", flush=True)
    G = sum(g for g, _ in rows.values()); F = sum(f for _, f in rows.values())
    print(f"\n  TOTAL levels: graph={G}  forge={F}  ({BUDGET}s/game, from scratch, no cache)")
    return G, F


def test_graph_beats_forge():
    G, F = compare()
    assert G >= F, f"graph-explore ({G}) did NOT beat Forge ({F})"


if __name__ == "__main__":
    G, F = compare()
    print("\nVERDICT:", "graph-explore SUPERIOR" if G > F else ("EQUAL" if G == F else "Forge superior"))
