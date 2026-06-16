"""v17 iteration driver. ONE iteration per invocation (so each fits a bounded
compute slice and produces a self-contained per-iteration log). Appends a
record to v17_results.json; iteration 5 also writes ITERATIONS.md.

  python run_iterations.py --iter 1     # BFS baseline (breadth-death)
  python run_iterations.py --iter 2     # trace gen + dataset build (v13 parts)
  python run_iterations.py --iter 3     # ForgeNet heuristic -> greedy/A* search
  python run_iterations.py --iter 4     # TRM policy/value -> PUCT search
  python run_iterations.py --iter 5     # ForgeNet+TRM + expert-iteration on L5

Each iteration: (a) verifies L0-L4 via the real engine (benchmark), (b) does
its L5 work, (c) logs everything to logs/iterN.log. Budgets via env:
  V17_L5_TIME (sec, default 20)   V17_L5_NODES (default 400)
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import engine as E
import search as S
import benchmark as B
from vlog import get_logger, banner

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "models")
os.makedirs(MODELS, exist_ok=True)
RESULTS = os.path.join(HERE, "v17_results.json")
GAME, L5 = "ls20", 5
L5_TIME = float(os.environ.get("V17_L5_TIME", "20"))
L5_NODES = int(os.environ.get("V17_L5_NODES", "400"))


def _load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {"game": GAME, "target": "L5", "iterations": {}}


def _save_results(r):
    json.dump(r, open(RESULTS, "w"), indent=2)


def _mask():
    p = os.path.join(MODELS, "transient_mask.npy")
    if os.path.exists(p):
        return np.load(p)
    cache, _ = E.load_cache(GAME)
    m = E.detect_transient_mask(GAME, L5, cache)
    if m is not None:
        np.save(p, m)
    return m


def _dataset(iter_tag):
    """Cache the replayed + featurized dataset so iters 3-5 don't rebuild."""
    import data, forgenet
    p = os.path.join(MODELS, "dataset.npz")
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return d["X"], d["objs"], d["costs"], d["aclasses"]
    ds = data.build_datasets(GAME, levels=(0, 1, 2, 3, 4), iter_tag=iter_tag)
    fn = forgenet.ForgeNet()
    X = np.array([fn.featurize(f, o) for f, o in zip(ds["frames"], ds["objs"])], np.float32)
    np.savez(p, X=X, objs=ds["objs"], costs=ds["costs"], aclasses=ds["aclasses"])
    return X, ds["objs"], ds["costs"], ds["aclasses"]


def iter1(lg):
    banner(lg, "ITER 1 — BFS baseline (reproduce v13 breadth-death on L5)")
    bench = B.replay_and_verify(GAME, upto_level=5, iter_tag="iter1")
    res = S.solve_level(GAME, L5, strategy="bfs", node_budget=L5_NODES,
                        time_budget=L5_TIME, mask=_mask(), iter_tag="iter1",
                        log_every=100, logger=lg)
    return {"name": "BFS baseline", "bench": bench, "l5": _strip(res)}


def iter2(lg):
    banner(lg, "ITER 2 — instrumented BFS: trace gen + training dataset (v13 parts)")
    bench = B.replay_and_verify(GAME, upto_level=5, iter_tag="iter2")
    # short trace-emitting BFS to demonstrate trace extraction
    res = S.solve_level(GAME, L5, strategy="bfs", node_budget=80,
                        time_budget=12, mask=_mask(), iter_tag="iter2",
                        emit_trace=True, log_every=40, logger=lg)
    ntrace = len(res.get("trace") or [])
    lg.info(f"[ITER2] BFS emitted {ntrace} labelled transitions from L5 probe")
    X, objs, costs, acl = _dataset("iter2")
    lg.info(f"[ITER2] training dataset built: {X.shape[0]} states, feat_dim={X.shape[1]}, "
            f"cost_range=[{costs.min():.0f},{costs.max():.0f}], "
            f"policy_labeled={int((acl>=0).sum())}")
    return {"name": "trace+dataset", "bench": bench,
            "l5_probe": _strip(res), "dataset": {"states": int(X.shape[0]),
            "feat_dim": int(X.shape[1]), "trace_transitions": ntrace}}


def iter3(lg):
    banner(lg, "ITER 3 — CNN ForgeNet cost-to-go heuristic -> A* search on L5")
    import forgenet
    bench = B.replay_and_verify(GAME, upto_level=5, iter_tag="iter3")
    X, objs, costs, acl = _dataset("iter3")
    fn = forgenet.ForgeNet()
    fn.fit(X, costs, epochs=200, lr=3e-3, bsz=64, iter_tag="iter3")
    fn.save(os.path.join(MODELS, "forgenet.npz"))
    hpath = os.path.join(MODELS, "forgenet.npz")
    fn = forgenet.ForgeNet.load(hpath)
    hfn = lambda frame: fn.predict(frame, E.object_features(frame))
    res = S.solve_level(GAME, L5, strategy="astar", node_budget=L5_NODES,
                        time_budget=L5_TIME, heuristic_fn=hfn, astar_w=2.0,
                        mask=_mask(), iter_tag="iter3", log_every=100, logger=lg)
    return {"name": "ForgeNet A*", "bench": bench, "l5": _strip(res)}


def iter4(lg):
    banner(lg, "ITER 4 — TRM policy/value prior -> PUCT search on L5")
    import trm
    bench = B.replay_and_verify(GAME, upto_level=5, iter_tag="iter4")
    X, objs, costs, acl = _dataset("iter4")
    m = trm.TRM(in_dim=76, hidden=48, T=3)
    m.fit(objs, acl, costs, epochs=250, lr=3e-3, bsz=64, iter_tag="iter4")
    m.save(os.path.join(MODELS, "trm.npz"))
    m = trm.TRM.load(os.path.join(MODELS, "trm.npz"))
    res = S.solve_level(GAME, L5, strategy="puct", node_budget=L5_NODES,
                        time_budget=L5_TIME, policy_fn=m.policy_fn,
                        heuristic_fn=lambda f: float(1.0 - m.forward(E.object_features(f))[1][0]),
                        policy_topk=6, mask=_mask(), iter_tag="iter4",
                        log_every=100, logger=lg)
    return {"name": "TRM PUCT", "bench": bench, "l5": _strip(res)}


def iter5(lg):
    banner(lg, "ITER 5 — ForgeNet + TRM + expert-iteration: final L5 attack")
    import forgenet, trm
    bench = B.replay_and_verify(GAME, upto_level=5, iter_tag="iter5")
    X, objs, costs, acl = _dataset("iter5")
    fn = (forgenet.ForgeNet.load(os.path.join(MODELS, "forgenet.npz"))
          if os.path.exists(os.path.join(MODELS, "forgenet.npz"))
          else forgenet.ForgeNet().fit(X, costs, epochs=200, iter_tag="iter5"))
    m = (trm.TRM.load(os.path.join(MODELS, "trm.npz"))
         if os.path.exists(os.path.join(MODELS, "trm.npz"))
         else trm.TRM().fit(objs, acl, costs, epochs=250, iter_tag="iter5"))
    hfn = lambda frame: (fn.predict(frame, E.object_features(frame))
                         * (1.2 - m.forward(E.object_features(frame))[1][0]))
    # round 1
    res = S.solve_level(GAME, L5, strategy="astar", node_budget=L5_NODES,
                        time_budget=L5_TIME, heuristic_fn=hfn,
                        policy_fn=m.policy_fn, astar_w=2.5, mask=_mask(),
                        iter_tag="iter5", log_every=100, logger=lg)
    lg.info(f"[ITER5] round-1 status={res['status']} best_depth={res['best_depth']} "
            f"expansions={res['expansions']}")
    out = {"name": "ForgeNet+TRM+ExIt", "bench": bench, "l5_round1": _strip(res)}
    if res["solution"]:
        sol = [(a, d) for a, d in res["solution"]]
        ver = B.replay_and_verify(GAME, upto_level=5, extra=sol, iter_tag="iter5")
        out["L5_SOLVED"] = True
        out["l5_solution_len"] = len(sol)
        out["verified_levels_completed"] = ver["levels_completed"]
        # persist into a v17 cache
        cache, _ = E.load_cache(GAME)
        cache["5"] = [[a, d] for (a, d) in sol]
        json.dump(cache, open(os.path.join(HERE, f"v17_bfs_cache_{GAME}.json"), "w"))
        lg.info(f"[ITER5] *** L5 SOLVED in {len(sol)} actions — cached ***")
    else:
        out["L5_SOLVED"] = False
        lg.info("[ITER5] L5 not solved within budget — see ITERATIONS.md for the "
                "honest status and the compute/scale path forward.")
    _write_iterations_md()
    return out


def _strip(res):
    return {k: v for k, v in res.items() if k != "trace"}


def _write_iterations_md():
    r = _load_results()
    md = ["# v17 — iteration log (auto-generated)\n",
          f"Target: **{GAME} L5** (the dual-key level; v13 solved L0–L4 = "
          "13/45/39/43/44 actions and breadth-died on L5).\n",
          "All L0–L4 numbers are re-verified through the REAL engine every "
          "iteration (no self-reported wins).\n",
          "| iter | approach | L0–L4 verified | RHAE | L5 status | L5 best_depth | L5 expansions | nodes/s |",
          "|---|---|---|---|---|---|---|---|"]
    for i in range(1, 6):
        it = r["iterations"].get(str(i))
        if not it:
            md.append(f"| {i} | (not run) | - | - | - | - | - | - |")
            continue
        l5 = it.get("l5") or it.get("l5_round1") or it.get("l5_probe") or {}
        bench = it.get("bench", {})
        lc = bench.get("levels_completed", "-")
        status = "SOLVED" if it.get("L5_SOLVED") else l5.get("status", "-")
        md.append(f"| {i} | {it.get('name','-')} | lc={lc} | "
                  f"{bench.get('rhae','-')} | {status} | {l5.get('best_depth','-')} | "
                  f"{l5.get('expansions','-')} | {l5.get('nodes_per_s','-')} |")
    md.append("\nSee `logs/iterN.log` for the full per-iteration trace.")
    open(os.path.join(HERE, "ITERATIONS.md"), "w").write("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int, required=True)
    args = ap.parse_args()
    lg = get_logger("run", f"iter{args.iter}")
    t0 = time.time()
    fn = {1: iter1, 2: iter2, 3: iter3, 4: iter4, 5: iter5}[args.iter]
    rec = fn(lg)
    rec["wall_time_s"] = round(time.time() - t0, 1)
    results = _load_results()
    results["iterations"][str(args.iter)] = rec
    _save_results(results)
    _write_iterations_md()
    lg.info(f"[DONE] iter {args.iter} in {rec['wall_time_s']}s -> appended to v17_results.json")
    print(json.dumps(rec, indent=2)[:1500])


if __name__ == "__main__":
    main()
