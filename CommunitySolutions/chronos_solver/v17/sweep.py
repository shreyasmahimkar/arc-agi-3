"""v17 iterations 6-25 — the experimental sweep + expert-iteration loop.

ONE iteration per invocation (or a small batch via --batch), each appended to
v17_results.json and logged to logs/iterN.log. Three phases:

  6-11  HYPERPARAM SWEEP   : tune progress_weight / astar_w / strategy on the
                            progress-shaped search built in this round.
  12-19 EXPERT ITERATION   : search -> harvest the most-progressed path ->
                            bootstrap cost-to-go labels along it (SoS/ExIt) ->
                            retrain ForgeNet -> re-search. The engine is a
                            perfect verifier, so harvested progress is real.
  20-25 SCALED PUSH        : best config + the accumulated ExIt heuristic, with
                            larger node budgets, for the final L5 attempt.

Every iteration re-verifies L0-L4 through the real engine (benchmark) and
records best_depth + best_progress so the trajectory is auditable.
"""
from __future__ import annotations
import os, sys, json, time, argparse
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import engine as E
import search as S
import benchmark as B
import forgenet
import trm
from vlog import get_logger, banner

HERE = os.path.dirname(__file__)
MODELS = os.path.join(HERE, "models")
RESULTS = os.path.join(HERE, "v17_results.json")
GAME, L5 = "ls20", 5
NEEDED_PROGRESS = 4          # dual-key guess: 2 keys + 2 locks/goal-matches


# ---- config table for iters 6-25 -------------------------------------------
# strategy, astar_w, progress_weight, use_policy, node_budget, time_budget, exit
CONFIGS = {
    6:  dict(name="sweep pw=4",        strategy="astar", w=2.0, pw=4,  pol=False, nb=500, tb=22),
    7:  dict(name="sweep pw=8",        strategy="astar", w=2.0, pw=8,  pol=False, nb=500, tb=16),
    8:  dict(name="sweep pw=16",       strategy="astar", w=1.5, pw=16, pol=False, nb=500, tb=16),
    9:  dict(name="greedy+progress",   strategy="greedy", w=0.0, pw=8, pol=False, nb=500, tb=16),
    10: dict(name="astar w=3 pw=8",    strategy="astar", w=3.0, pw=8,  pol=False, nb=500, tb=16),
    11: dict(name="astar+policy pw=12",strategy="astar", w=2.5, pw=12, pol=True,  nb=500, tb=16),
    12: dict(name="ExIt round 1",      strategy="astar", w=2.0, pw=10, pol=True,  nb=550, tb=17, exit=True),
    13: dict(name="ExIt round 2",      strategy="astar", w=2.0, pw=10, pol=True,  nb=550, tb=17, exit=True),
    14: dict(name="ExIt round 3",      strategy="astar", w=2.0, pw=12, pol=True,  nb=600, tb=17, exit=True),
    15: dict(name="ExIt round 4",      strategy="astar", w=2.0, pw=12, pol=True,  nb=600, tb=17, exit=True),
    16: dict(name="ExIt round 5",      strategy="astar", w=1.8, pw=14, pol=True,  nb=650, tb=17, exit=True),
    17: dict(name="ExIt round 6",      strategy="astar", w=1.8, pw=14, pol=True,  nb=650, tb=17, exit=True),
    18: dict(name="ExIt round 7",      strategy="greedy", w=0.0, pw=14, pol=True, nb=650, tb=17, exit=True),
    19: dict(name="ExIt round 8",      strategy="astar", w=1.5, pw=16, pol=True,  nb=700, tb=17, exit=True),
    20: dict(name="scaled push 1",     strategy="astar", w=1.8, pw=14, pol=True,  nb=900, tb=30, exit=True),
    21: dict(name="scaled push 2",     strategy="astar", w=1.8, pw=14, pol=True,  nb=900, tb=30, exit=True),
    22: dict(name="scaled push 3",     strategy="greedy", w=0.0, pw=16, pol=True, nb=1000, tb=32, exit=True),
    23: dict(name="scaled push 4",     strategy="astar", w=1.5, pw=18, pol=True,  nb=1000, tb=32, exit=True),
    24: dict(name="scaled push 5",     strategy="astar", w=1.5, pw=18, pol=True,  nb=1100, tb=34, exit=True),
    25: dict(name="final attack",      strategy="astar", w=1.6, pw=16, pol=True,  nb=1200, tb=36, exit=True),
    # 26-30: research-driven (a web-searched technique per iteration)
    26: dict(name="Subgoal Search (waypoint re-root)", mode="waypoint",
             strategy="greedy", w=0.0, pw=14, pol=True, nb=350, tb=8, rounds=4),
    27: dict(name="BFWS novelty (Best-First Width)", strategy="greedy", w=0.0,
             pw=12, pol=True, nb=900, tb=34, novelty=12, nw=6.0),
    28: dict(name="Macro-actions (move-until-wall) + BFWS", strategy="greedy",
             w=0.0, pw=12, pol=True, nb=700, tb=34, novelty=12, nw=6.0, macro=10),
    29: dict(name="Type-based explore + waypoint + novelty + macro", mode="waypoint",
             strategy="greedy", w=0.0, pw=12, pol=True, nb=400, tb=8, rounds=4,
             novelty=12, nw=6.0, macro=8, explore=0.35),
    30: dict(name="Full-stack final attack", mode="waypoint", strategy="greedy",
             w=0.0, pw=14, pol=True, nb=600, tb=11, rounds=5,
             novelty=12, nw=6.0, macro=8, explore=0.4),
    # 31-33: #1 multiprocessing (4 cores)
    31: dict(name="MP greedy+macro (4 workers)", mode="mp", strategy="greedy",
             w=0.0, pw=12, pol=True, nb=700, tb=34, macro=8, workers=4, batch=8),
    32: dict(name="MP greedy+macro big-batch", mode="mp", strategy="greedy",
             w=0.0, pw=12, pol=True, nb=900, tb=34, macro=8, workers=4, batch=12),
    33: dict(name="MP waypoint (landmark re-root on cores)", mode="mp_waypoint",
             strategy="greedy", w=0.0, pw=12, pol=True, nb=400, tb=10, macro=8,
             workers=4, batch=8, rounds=3),
    # 34-40: #2 imagination / forward-rollout MCTS
    34: dict(name="MCTS forward-rollout (UCT+heavy playout)", mode="mcts",
             pol=True, sims=6000, tb=30, rollout=40, cuct=1.4, macro=8),
    35: dict(name="MCTS short rollout (more sims)", mode="mcts", pol=True,
             sims=9000, tb=30, rollout=18, cuct=1.4, macro=6),
    36: dict(name="MCTS high exploration (c=2.5)", mode="mcts", pol=True,
             sims=8000, tb=30, rollout=25, cuct=2.5, macro=6),
    37: dict(name="MCTS progress-bonus 20", mode="mcts", pol=True,
             sims=8000, tb=30, rollout=25, cuct=1.6, macro=6, pbonus=20.0),
    38: dict(name="MCTS waypoint re-root", mode="mcts_waypoint", pol=True,
             sims=4000, tb=12, rollout=20, cuct=1.6, macro=6, rounds=3),
    39: dict(name="MCTS waypoint deep", mode="mcts_waypoint", pol=True,
             sims=5000, tb=14, rollout=22, cuct=1.8, macro=6, rounds=3),
    40: dict(name="MCTS final big-budget attack", mode="mcts", pol=True,
             sims=20000, tb=38, rollout=22, cuct=1.8, macro=6, pbonus=15.0),
    # 41-46: #1 AlphaZero-style value-guided MCTS (PUCT + value, micro-rollout)
    41: dict(name="AZ value-MCTS (PUCT+value)", mode="az", pol=True,
             sims=12000, tb=34, cuct=2.0, macro=6, micro=12, vw=0.3),
    42: dict(name="AZ high c_puct=3.5", mode="az", pol=True,
             sims=12000, tb=34, cuct=3.5, macro=6, micro=12, vw=0.3),
    43: dict(name="AZ longer micro-rollout=20", mode="az", pol=True,
             sims=10000, tb=34, cuct=2.5, macro=6, micro=20, vw=0.3),
    44: dict(name="AZ value-weight 0.8", mode="az", pol=True,
             sims=12000, tb=34, cuct=2.5, macro=6, micro=12, vw=0.8),
    45: dict(name="AZ low c_puct=1.2 (exploit policy)", mode="az", pol=True,
             sims=12000, tb=34, cuct=1.2, macro=6, micro=14, vw=0.3),
    46: dict(name="AZ best-config big budget", mode="az", pol=True,
             sims=22000, tb=38, cuct=2.5, macro=6, micro=16, vw=0.4),
    # 47-60: #2 expert iteration — retrain TRM value/policy on progress-3 traces
    47: dict(name="ExIt round 1 (harvest+retrain value)", mode="az_exit", pol=True,
             sims=9000, tb=30, cuct=2.0, macro=6, micro=14, vw=0.5),
    48: dict(name="ExIt round 2", mode="az_exit", pol=True, sims=9000, tb=30, cuct=2.0, macro=6, micro=14, vw=0.5),
    49: dict(name="ExIt round 3", mode="az_exit", pol=True, sims=9000, tb=30, cuct=2.2, macro=6, micro=14, vw=0.5),
    50: dict(name="ExIt round 4", mode="az_exit", pol=True, sims=10000, tb=30, cuct=2.2, macro=6, micro=14, vw=0.6),
    51: dict(name="ExIt round 5", mode="az_exit", pol=True, sims=10000, tb=30, cuct=2.0, macro=6, micro=16, vw=0.6),
    52: dict(name="ExIt round 6", mode="az_exit", pol=True, sims=10000, tb=32, cuct=2.0, macro=6, micro=16, vw=0.6),
    53: dict(name="ExIt round 7", mode="az_exit", pol=True, sims=11000, tb=32, cuct=1.8, macro=6, micro=16, vw=0.7),
    54: dict(name="ExIt round 8", mode="az_exit", pol=True, sims=11000, tb=32, cuct=1.8, macro=6, micro=18, vw=0.7),
    55: dict(name="ExIt round 9", mode="az_exit", pol=True, sims=12000, tb=34, cuct=1.8, macro=6, micro=18, vw=0.7),
    56: dict(name="ExIt round 10", mode="az_exit", pol=True, sims=12000, tb=34, cuct=1.6, macro=6, micro=18, vw=0.8),
    57: dict(name="ExIt round 11", mode="az_exit", pol=True, sims=13000, tb=34, cuct=1.6, macro=6, micro=20, vw=0.8),
    58: dict(name="ExIt round 12", mode="az_exit", pol=True, sims=13000, tb=34, cuct=1.6, macro=6, micro=20, vw=0.8),
    59: dict(name="ExIt + value-MCTS big budget", mode="az_exit", pol=True, sims=20000, tb=36, cuct=1.8, macro=6, micro=18, vw=0.7),
    60: dict(name="ExIt FINAL attack (trained value)", mode="az_exit", pol=True, sims=24000, tb=37, cuct=1.7, macro=6, micro=18, vw=0.7),
}


def _mask():
    p = os.path.join(MODELS, "transient_mask.npy")
    return np.load(p) if os.path.exists(p) else None


def _load_base():
    d = np.load(os.path.join(MODELS, "dataset.npz"), allow_pickle=True)
    return d["X"], d["objs"], d["costs"], d["aclasses"]


def _forgenet():
    p = os.path.join(MODELS, "forgenet_exit.npz")
    if os.path.exists(p):
        return forgenet.ForgeNet.load(p)
    return forgenet.ForgeNet.load(os.path.join(MODELS, "forgenet.npz"))


def _trm():
    return trm.TRM.load(os.path.join(MODELS, "trm.npz"))


def _harvest_to_training(harvest, lg):
    """Replay the harvested best path, record frames, label SoS cost-to-go:
    state i along a length-D path gets cost = (D - i) + slack. Teaches the
    heuristic to value the deep progressed states it actually reached."""
    if not harvest or not harvest.get("path"):
        return None, None
    path = harvest["path"]
    D = len(path)
    cache, _ = E.load_cache(GAME)
    g = E.load_game(GAME)
    E.chain_to_level(g, L5, cache)
    fn = forgenet.ForgeNet()           # only for featurize (deterministic conv)
    frames, objs, costs = [], [], []
    f = E.frame_of(E.perform(g, 0)) if False else None
    # replay path step by step
    g = E.load_game(GAME); r, _ = E.chain_to_level(g, L5, cache)
    f = E.frame_of(r)
    for i, act in enumerate(path):
        objf = E.object_features(f)
        frames.append(fn.featurize(f, objf)); objs.append(objf)
        costs.append(float(D - i))
        a = act[0]; d = act[1] if len(act) > 1 and act[1] else None
        r = E.perform(g, a, d); f = E.frame_of(r)
    lg.info(f"[EXIT] harvested path D={D} best_progress={harvest.get('best_progress')} "
            f"-> {len(frames)} bootstrapped states (cost {min(costs):.0f}..{max(costs):.0f})")
    return np.array(frames, np.float32), np.array(costs, np.float32)


def _trm_current():
    """Prefer the expert-iteration-retrained TRM if present."""
    p = os.path.join(MODELS, "trm_exit.npz")
    return trm.TRM.load(p) if os.path.exists(p) else _trm()


def _harvest_trm_data(path, lg):
    """Replay a harvested progress-N path; emit per-state (obj, action-class,
    cost-to-go = D-i). Hindsight: the deepest states of a good path get the
    highest value (lowest cost), teaching the value head which progress-3
    states are close to the goal."""
    import data as _data
    cache, _ = E.load_cache(GAME)
    g = E.load_game(GAME); r, _ = E.chain_to_level(g, L5, cache)
    f = E.frame_of(r); D = len(path)
    objs, acl, costs = [], [], []
    for i, act in enumerate(path):
        objs.append(E.object_features(f))
        a = act[0]
        acl.append(_data.ACTION_CLASSES.index(a) if a in _data.ACTION_CLASSES
                   else _data.ACTION_CLASSES.index(6))
        costs.append(float(D - i))
        r = E.perform(g, a, act[1] if len(act) > 1 and act[1] else None); f = E.frame_of(r)
    lg.info(f"[EXIT] harvested progress path D={D} -> {len(objs)} value/policy states")
    return np.array(objs, np.float32), np.array(acl, np.int64), np.array(costs, np.float32)


def _exit_buffer(add=None, cap=4000):
    """Persistent replay buffer of harvested progress states across rounds."""
    p = os.path.join(MODELS, "exit_buffer.npz")
    if os.path.exists(p):
        d = np.load(p)
        O, A, C = d["objs"], d["acl"], d["costs"]
    else:
        O = np.zeros((0, 76), np.float32); A = np.zeros((0,), np.int64); C = np.zeros((0,), np.float32)
    if add is not None:
        O = np.vstack([O, add[0]])[-cap:]
        A = np.concatenate([A, add[1]])[-cap:]
        C = np.concatenate([C, add[2]])[-cap:]
        np.savez(p, objs=O, acl=A, costs=C)
    return O, A, C


def waypoint_search(fn, m, pol, mask, lg, rounds=4, per_round_nb=350,
                    per_round_tb=8.0, strategy="greedy", pw=14, w=0.0,
                    novelty=0, nw=0.0, macro=0, explore_p=0.0):
    """Iterated Subgoal Search (Czechowski et al. 2021 / HIGL landmarks):
    search to the most-progressed landmark, RE-ROOT there, search again for the
    next progress level. Decomposes L5's ~45-step dual-key depth into a chain of
    short ~20-step sub-searches — quadratically easier for best-first."""
    hfn = lambda f: fn.predict(f, E.object_features(f))
    prefix = []
    cur_prog = 0
    best = {"best_depth": 0, "best_progress": 0, "expansions": 0,
            "status": "TIMEOUT", "solution": None, "nodes_per_s": 0.0, "visited": 0}
    total_exp = 0
    for rd in range(rounds):
        res = S.solve_level(GAME, L5, strategy=strategy, node_budget=per_round_nb,
                            time_budget=per_round_tb, heuristic_fn=hfn, policy_fn=pol,
                            policy_topk=8, astar_w=w, progress_weight=pw, mask=mask,
                            harvest_k=1, prefix_path=prefix, novelty_bins=novelty,
                            novelty_weight=nw, macro_moves=bool(macro), macro_max=macro or 10,
                            explore_p=explore_p, iter_tag=lg.name.split(":")[0],
                            log_every=200, logger=lg)
        total_exp += res["expansions"]
        best["best_depth"] = max(best["best_depth"], res["best_depth"])
        best["best_progress"] = max(best["best_progress"], res.get("best_progress", 0))
        best["visited"] += res["visited"]
        if res["solution"]:
            best["solution"] = res["solution"]; best["status"] = "SOLVED"
            lg.info(f"[WAYPOINT] WIN at round {rd}"); break
        h = res.get("harvest")
        np_ = h.get("best_progress", 0) if h else 0
        lg.info(f"[WAYPOINT] round {rd}: reached progress {np_} (was {cur_prog}), "
                f"depth {res['best_depth']}, exp {res['expansions']}")
        if h and np_ > cur_prog:
            prefix = h["path"]; cur_prog = np_          # re-root at the landmark
        else:
            lg.info(f"[WAYPOINT] round {rd}: no new landmark — stop")
            break
    best["expansions"] = total_exp
    best["status"] = "SOLVED" if best["solution"] else "TIMEOUT"
    return best


def run_iter(n):
    cfg = CONFIGS[n]
    lg = get_logger("sweep", f"iter{n}")
    banner(lg, f"ITER {n} — {cfg['name']}  [mode={cfg.get('mode','single')} "
               f"strategy={cfg.get('strategy','-')} pol={cfg.get('pol')} "
               f"exit={cfg.get('exit',False)}]")
    bench = B.replay_and_verify(GAME, upto_level=5, iter_tag=f"iter{n}")

    fn = _forgenet()
    m = _trm()
    hfn = lambda f: fn.predict(f, E.object_features(f))
    pol = m.policy_fn if cfg["pol"] else None

    if cfg.get("mode") in ("az", "az_exit"):
        import mcts
        if cfg["mode"] == "az_exit":
            m = _trm_current()                 # use the ExIt-retrained TRM
        vfn = lambda f: float(m.forward(E.object_features(f))[1][0])
        pol = m.policy_fn if cfg.get("pol") else None
        res = mcts.solve_mcts_az(GAME, L5, sims=cfg["sims"], time_budget=cfg["tb"],
                                 c_puct=cfg.get("cuct", 2.0), value_fn=vfn, policy_fn=pol,
                                 macro_moves=bool(cfg.get("macro")), macro_max=cfg.get("macro", 8),
                                 mask=_mask(), micro_rollout=cfg.get("micro", 8),
                                 value_weight=cfg.get("vw", 0.3), prefix_path=None,
                                 iter_tag=f"iter{n}")
        if cfg["mode"] == "az_exit" and not res["solution"]:
            # EXPERT ITERATION: harvest the best path, retrain TRM value+policy
            # and ForgeNet on base + accumulated progress buffer.
            Xo, acl, costs = _harvest_trm_data(res["harvest"]["path"], lg)
            bO, bA, bC = _exit_buffer(add=(Xo, acl, costs))
            X0, o0, c0, a0 = _load_base()
            O = np.vstack([o0, bO]); A = np.concatenate([a0, bA]); C = np.concatenate([c0, bC])
            m2 = trm.TRM(in_dim=76, hidden=48, T=3)
            m2.fit(O, A, C, epochs=220, lr=3e-3, bsz=64, iter_tag=f"iter{n}")
            m2.save(os.path.join(MODELS, "trm_exit.npz"))
            res["exit_retrain"] = {"exit_buffer": int(len(bO)), "total_states": int(len(O))}
            lg.info(f"[EXIT] retrained TRM value+policy on {len(O)} states "
                    f"(+{len(bO)} progress buffer) -> trm_exit.npz")
        res["trace"] = None; res.pop("harvest", None)
    elif cfg.get("mode") in ("mcts", "mcts_waypoint"):
        import mcts
        mk = dict(rollout_len=cfg.get("rollout", 30), c_uct=cfg.get("cuct", 1.4),
                  heuristic_fn=hfn, policy_fn=pol, macro_moves=bool(cfg.get("macro")),
                  macro_max=cfg.get("macro", 8), mask=_mask(),
                  progress_bonus=cfg.get("pbonus", 10.0))
        if cfg["mode"] == "mcts":
            res = mcts.solve_mcts(GAME, L5, sims=cfg["sims"], time_budget=cfg["tb"],
                                  iter_tag=f"iter{n}", **mk)
        else:                       # mcts_waypoint: re-root at the best landmark
            prefix = []; cur = 0; tot = 0; best = None
            for rd in range(cfg.get("rounds", 3)):
                rr = mcts.solve_mcts(GAME, L5, sims=cfg["sims"], time_budget=cfg["tb"],
                                     prefix_path=prefix, iter_tag=f"iter{n}", **mk)
                tot += rr["expansions"]
                if best is None or rr["best_progress"] > best["best_progress"]:
                    best = rr
                if rr["solution"]:
                    best = rr; break
                np_ = rr["harvest"]["best_progress"]
                lg.info(f"[MCTS-WAYPOINT] round {rd}: progress {np_} (was {cur})")
                if np_ > cur:
                    prefix = rr["harvest"]["path"]; cur = np_
                else:
                    break
            res = best; res["expansions"] = tot
        res["trace"] = None; res.pop("harvest", None)
    elif cfg.get("mode") in ("mp", "mp_waypoint"):
        import mpsearch
        kw = dict(workers=cfg.get("workers", 4), batch=cfg.get("batch", 8),
                  strategy=cfg["strategy"], progress_weight=cfg["pw"], heuristic_fn=hfn,
                  policy_fn=pol, macro_moves=bool(cfg.get("macro")),
                  macro_max=cfg.get("macro", 10), mask=_mask(), astar_w=cfg["w"])
        if cfg["mode"] == "mp":
            res = mpsearch.solve_level_mp(GAME, L5, node_budget=cfg["nb"],
                                          time_budget=cfg["tb"], iter_tag=f"iter{n}", **kw)
        else:                       # mp_waypoint: re-root on cores per landmark
            prefix = []; cur = 0; tot = 0; res = None; best = None
            for rd in range(cfg.get("rounds", 3)):
                rr = mpsearch.solve_level_mp(GAME, L5, node_budget=cfg["nb"],
                                             time_budget=cfg["tb"], prefix_path=prefix,
                                             iter_tag=f"iter{n}", **kw)
                tot += rr["expansions"]
                if best is None or rr["best_progress"] > best["best_progress"]:
                    best = rr
                if rr["solution"]:
                    best = rr; break
                np_ = rr["harvest"]["best_progress"]
                lg.info(f"[MP-WAYPOINT] round {rd}: progress {np_} (was {cur})")
                if np_ > cur:
                    prefix = rr["harvest"]["path"]; cur = np_
                else:
                    break
            res = best; res["expansions"] = tot
        res["trace"] = None; res.pop("harvest", None)
    elif cfg.get("mode") == "waypoint":
        res = waypoint_search(fn, m, pol, _mask(), lg, rounds=cfg.get("rounds", 4),
                              per_round_nb=cfg["nb"], per_round_tb=cfg["tb"],
                              strategy=cfg["strategy"], pw=cfg["pw"], w=cfg["w"],
                              novelty=cfg.get("novelty", 0), nw=cfg.get("nw", 0.0),
                              macro=cfg.get("macro", 0), explore_p=cfg.get("explore", 0.0))
        res["trace"] = None; res["harvest"] = None
    else:
        res = S.solve_level(GAME, L5, strategy=cfg["strategy"], node_budget=cfg["nb"],
                            time_budget=cfg["tb"], heuristic_fn=hfn, policy_fn=pol,
                            policy_topk=8, astar_w=cfg["w"], progress_weight=cfg["pw"],
                            mask=_mask(), harvest_k=1, novelty_bins=cfg.get("novelty", 0),
                            novelty_weight=cfg.get("nw", 0.0),
                            macro_moves=bool(cfg.get("macro")), macro_max=cfg.get("macro", 10),
                            iter_tag=f"iter{n}", log_every=150, logger=lg)
    rec = {"name": cfg["name"], "config": {k: v for k, v in cfg.items() if k != "name"},
           "bench": bench, "l5": {k: v for k, v in res.items() if k not in ("trace", "harvest")}}

    if res["solution"]:
        sol = [(a, d) for a, d in res["solution"]]
        ver = B.replay_and_verify(GAME, upto_level=5, extra=sol, iter_tag=f"iter{n}")
        rec["L5_SOLVED"] = True
        rec["verified_levels_completed"] = ver["levels_completed"]
        cache, _ = E.load_cache(GAME)
        cache["5"] = [[a, d] for (a, d) in sol]
        json.dump(cache, open(os.path.join(HERE, f"v17_bfs_cache_{GAME}.json"), "w"))
        lg.info(f"[ITER{n}] *** L5 SOLVED in {len(sol)} actions, verified lc="
                f"{ver['levels_completed']} ***")

    # expert iteration: retrain ForgeNet on base + harvested demonstration
    if cfg.get("exit") and res.get("harvest"):
        Xh, ch = _harvest_to_training(res["harvest"], lg)
        if Xh is not None and len(Xh):
            Xb, objb, cb, acl = _load_base()
            X = np.vstack([Xb, Xh]); c = np.concatenate([cb, ch])
            fn2 = forgenet.ForgeNet()
            fn2.fit(X, c, epochs=160, lr=3e-3, bsz=64, iter_tag=f"iter{n}")
            fn2.save(os.path.join(MODELS, "forgenet_exit.npz"))
            rec["exit_retrain"] = {"added_states": int(len(Xh)),
                                   "total_states": int(len(X))}
            lg.info(f"[ITER{n}] ExIt retrained ForgeNet on {len(X)} states "
                    f"(+{len(Xh)} harvested) -> forgenet_exit.npz")

    rec["best_progress"] = res.get("best_progress", 0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iter", type=int)
    ap.add_argument("--batch", type=str, help="e.g. 6-8")
    args = ap.parse_args()
    iters = []
    if args.batch:
        a, b = args.batch.split("-"); iters = list(range(int(a), int(b) + 1))
    elif args.iter:
        iters = [args.iter]
    results = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {"iterations": {}}
    for n in iters:
        t0 = time.time()
        rec = run_iter(n)
        rec["wall_time_s"] = round(time.time() - t0, 1)
        results["iterations"][str(n)] = rec
        json.dump(results, open(RESULTS, "w"), indent=2)
        print(f"iter {n}: {rec['name']} -> best_depth={rec['l5']['best_depth']} "
              f"best_progress={rec['best_progress']} status={rec['l5']['status']} "
              f"solved={rec.get('L5_SOLVED', False)} ({rec['wall_time_s']}s)")


if __name__ == "__main__":
    main()
