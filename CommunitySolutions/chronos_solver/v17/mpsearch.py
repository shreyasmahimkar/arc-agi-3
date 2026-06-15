"""v17 #1 — multiprocess best-first search.

The single-process searcher is throughput-bound by snapshot restore (pickle
~140-300/s). Each node expansion needs ~12 independent state copies (one per
candidate action), so cores help linearly. v13 had a multiprocess expander;
this is the v17 port, with one twist for low IPC:

  * frontier nodes carry their ACTION PATH, not a 120KB snapshot;
  * a fork-inherited L5-start snapshot lives in every worker's memory;
  * a worker reconstructs a node by restoring that snapshot and replaying the
    (short) path, then expands all candidate actions and returns only small
    payloads (hashes, progress, child paths, 4KB frames) — never a snapshot.

So the heavy engine work (restore+perform) parallelises across cores while the
global state (visited set, novelty table, ForgeNet heuristic) stays in the main
process. Same progress-shaping / macro / novelty semantics as search.py.
"""
from __future__ import annotations
import heapq, pickle, time, os
import numpy as np
import multiprocessing as mp

import engine as E
from vlog import get_logger, Counters

_G = {}            # worker globals (fork-inherited): l5start snap, mask, etc.


def _restore_replay(path):
    g = pickle.loads(_G["l5start"])
    g._clean_levels = _G["clean"]
    for act in path:
        a = act[0]; d = act[1] if len(act) > 1 and act[1] else None
        E.perform(g, a, d)
    return g


def _worker_expand(task):
    """Expand one node. task=(path, depth, cands). Returns list of child dicts."""
    path, depth, cands = task
    base = _restore_replay(path)
    basesnap = pickle.dumps(base, protocol=5)
    level = _G["level"]; start_lc = _G["start_lc"]; mask = _G["mask"]
    root_scalar = _G["root_scalar"]; macro = _G["macro"]; macro_max = _G["macro_max"]
    out = []
    for (a, data) in cands:
        gg = pickle.loads(basesnap); gg._clean_levels = _G["clean"]
        rr = E.perform(gg, a, data)
        f = E.frame_of(rr)
        if f is None:
            continue
        steps = [(a, data)]
        win = bool(rr.levels_completed > start_lc or gg._current_level_index > level)
        if macro and data is None and a in E.MOVES and not win:
            prevf = f
            base_prog = _progress(gg, root_scalar)
            for _ in range(macro_max - 1):
                rr2 = E.perform(gg, a, None)
                f2 = E.frame_of(rr2)
                if f2 is None or np.array_equal(f2, prevf):
                    break
                steps.append((a, None)); prevf = f2; f = f2; rr = rr2
                if bool(rr2.levels_completed > start_lc or gg._current_level_index > level):
                    win = True; break
                if _progress(gg, root_scalar) > base_prog:
                    break
        h = E.state_hash(gg, f, mask)
        full = [list(x) for x in path] + [list(s) for s in steps]
        out.append({"hash": h, "win": win, "progress": _progress(gg, root_scalar),
                    "_path": full, "frame": f.astype(np.uint8).tobytes(),
                    "shape": f.shape, "depth": depth + len(steps)})
    return out


def _progress(g, root_scalar):
    cur = dict(E.scalar_state(g))
    return sum(1 for k, v in cur.items() if root_scalar.get(k) != v)


def solve_level_mp(game, level, *, workers=4, batch=8, node_budget=2000,
                   time_budget=30.0, heuristic_fn=None, policy_fn=None,
                   astar_w=1.0, strategy="astar", progress_weight=0.0,
                   macro_moves=False, macro_max=10, mask=None, prefix_path=None,
                   iter_tag="v17mp", log_every=100, logger=None):
    lg = logger or get_logger("mpsearch", iter_tag)
    ctr = Counters(); t0 = time.time()
    cache, _ = E.load_cache(game)
    g = E.load_game(game)
    r, chained = E.chain_to_level(g, level, cache)
    root_scalar = dict(E.scalar_state(g))
    # l5start = ABSOLUTE level-5 start (BEFORE any re-root prefix), because
    # frontier paths include the prefix and workers replay the FULL path from
    # l5start. Capturing it post-prefix would double-apply the prefix.
    l5start_snap = pickle.dumps(g, protocol=5)
    prefix_path = prefix_path or []
    for act in prefix_path:
        r = E.perform(g, act[0], act[1] if len(act) > 1 and act[1] else None)
    f0 = E.frame_of(r)
    start_lc = r.levels_completed
    # set worker globals BEFORE pool fork
    _G.update(l5start=l5start_snap, clean=g._clean_levels,
              level=level, start_lc=start_lc, mask=mask, root_scalar=root_scalar,
              macro=macro_moves, macro_max=macro_max)
    root_hash = E.state_hash(g, f0, mask)
    lg.info(f"[MP-START] {game} L{level} workers={workers} batch={batch} "
            f"strategy={strategy} budget={node_budget} time={time_budget}s "
            f"chained={chained} prefix={len(prefix_path)} start_lc={start_lc}")

    # node = (path, frame, depth, hash, progress)
    visited = {root_hash: 0}
    frontier = []
    counter = 0
    heapq.heappush(frontier, (0.0, counter, ([tuple(x) for x in prefix_path], f0, 0, root_hash, 0)))
    best_score = (0, 0); best_path = list(prefix_path)
    expansions = 0; solution = None

    ctx = mp.get_context("fork")
    pool = ctx.Pool(workers)
    try:
        while frontier and expansions < node_budget and (time.time()-t0) < time_budget:
            # pop a batch of the best nodes
            popped = []
            while frontier and len(popped) < batch:
                _, _, nd = heapq.heappop(frontier)
                popped.append(nd)
            tasks = []
            for (path, frame, depth, h, prog) in popped:
                cands = _cands(frame, policy_fn)
                tasks.append((path, depth, cands))
            results = pool.map(_worker_expand, tasks)
            expansions += len(popped)
            ctr.inc("expansions", len(popped))
            for children in results:
                for c in children:
                    ctr.inc("child_evals")
                    if c["win"]:
                        solution = [tuple(x) for x in c["_path"]]
                        lg.info(f"[MP-WIN] L{level} at depth {c['depth']} "
                                f"after {expansions} expansions, {time.time()-t0:.1f}s")
                        break
                    if c["hash"] in visited:
                        continue
                    visited[c["hash"]] = c["depth"]
                    f = np.frombuffer(c["frame"], dtype=np.uint8).reshape(c["shape"])
                    hh = float(heuristic_fn(f)) if heuristic_fn else 0.0
                    if strategy == "greedy":
                        pr = hh
                    else:
                        pr = c["depth"] + astar_w*hh
                    pr -= progress_weight * c["progress"]
                    counter += 1
                    # child path = parent path + steps; find parent path
                    heapq.heappush(frontier, (pr, counter,
                                   (c["_path"], f, c["depth"], c["hash"], c["progress"])))
                    sc = (c["progress"], c["depth"])
                    if sc > best_score:
                        best_score = sc; best_path = c["_path"]
                if solution:
                    break
            if solution:
                break
            if expansions % log_every < batch:
                dt = time.time()-t0
                lg.info(f"[MP-PROG] exp={expansions} visited={len(visited)} "
                        f"frontier={len(frontier)} best=({best_score[0]},{best_score[1]}) "
                        f"rate={expansions/max(dt,1e-6):.1f}n/s elapsed={dt:.1f}s "
                        f"counters={ctr.summary()}")
    finally:
        pool.close(); pool.join()

    dt = time.time()-t0
    status = "SOLVED" if solution else ("BUDGET" if expansions >= node_budget else "TIMEOUT")
    lg.info(f"[MP-END] L{level} status={status} expansions={expansions} "
            f"visited={len(visited)} best_progress={best_score[0]} "
            f"best_depth={best_score[1]} time={dt:.1f}s rate={expansions/max(dt,1e-6):.1f}n/s")
    return {"game": game, "level": level, "strategy": f"mp-{strategy}", "status": status,
            "solution": [list(x) for x in solution] if solution else None,
            "solution_len": len(solution) if solution else None,
            "expansions": expansions, "visited": len(visited),
            "best_depth": best_score[1], "best_progress": best_score[0],
            "time": round(dt, 2), "nodes_per_s": round(expansions/max(dt, 1e-6), 2),
            "harvest": {"path": [list(x) for x in best_path], "best_progress": best_score[0]}}


def _cands(frame, policy_fn, topk=None):
    cands = [(a, None) for a in E.MOVES] + E.dynamic_clicks(frame, limit=10)
    if policy_fn is not None:
        probs = policy_fn(frame)
        def keyf(c):
            k = c[0] if c[1] is None else (6, c[1]["x"], c[1]["y"])
            return probs.get(k, probs.get(c[0], 0.0))
        cands.sort(key=keyf, reverse=True)
    return cands
