"""v17 #2 — imagination search via forward-rollout MCTS.

The throughput realisation. Best-first snapshots EVERY node (pickle ~140-300/s
-> 8-28 nodes/s). MCTS instead restores the L5-start snapshot ONCE per
simulation and rolls FORWARD with perform() (~28,000/s), exploring a whole
depth-~50 trajectory per restore. Net: ~100 simulations/s × ~50 states each ≈
several thousand state-touches/s — two orders of magnitude over best-first.

This is the CPU realisation of v15's latent planner: v15 rolls forward in a
*learned* world model; here we roll forward in the *real* model, because the
real model's perform() is 200× faster than its snapshot, so it IS the cheap
imagination engine — with perfect fidelity (no model error to fight).

UCT selection + heavy (policy-biased) playouts (Browne et al. 2012;
heavy-playout MCTS). Reward = max progress (key/lock events) reached in the
subtree + ForgeNet value; a real WIN backs up reward 100 and stores the plan.
"""
from __future__ import annotations
import pickle, time, math, random
import numpy as np

import engine as E
from vlog import get_logger, Counters

_CLEAN = {}


class MNode:
    __slots__ = ("untried", "children", "N", "W", "progress", "best_prog")

    def __init__(self):
        self.untried = None          # list of candidate actions (lazy)
        self.children = {}           # action_key -> MNode
        self.N = 0
        self.W = 0.0
        self.progress = 0
        self.best_prog = 0


class ZNode:
    """AlphaZero node: priors + value backup, NO rollout."""
    __slots__ = ("N", "W", "children", "priors", "acts", "progress", "evaluated")

    def __init__(self):
        self.N = 0
        self.W = 0.0
        self.children = {}           # action_key -> ZNode
        self.priors = {}             # action_key -> P(a)
        self.acts = []               # candidate (a, data) tuples
        self.progress = 0
        self.evaluated = False


def _akey(a, data):
    return a if data is None else (6, data["x"], data["y"])


def solve_mcts_az(game, level, *, sims=8000, time_budget=30.0, c_puct=2.0,
                  value_fn=None, policy_fn=None, macro_moves=True, macro_max=8,
                  mask=None, prefix_path=None, max_progress=4, micro_rollout=4,
                  value_weight=0.3, world_model=None, wm_policy=None,
                  imagine_len=10, wm_weight=1.0, iter_tag="v17az",
                  clean_progress=True, log_every=600, logger=None):
    """AlphaZero-style value-guided MCTS: PUCT selection with a TRM policy
    prior + value-network leaf evaluation (NO long random rollout). value_fn
    maps a frame -> scalar value in [0,1] (the TRM value head). A short
    micro-rollout (default 4 cheap random steps) gives a bit of lookahead so a
    progress event just past the leaf is still credited. Reward shaping:
    leaf_value = progress/max_progress + 0.3*value_fn(frame)."""
    lg = logger or get_logger("mctsaz", iter_tag)
    ctr = Counters(); t0 = time.time()
    cache, _ = E.load_cache(game)
    g = E.load_game(game)
    if game not in _CLEAN:
        _CLEAN[game] = g._clean_levels
    r, chained = E.chain_to_level(g, level, cache)
    root_scalar = dict(E.scalar_state(g))
    prefix_path = [tuple(x) for x in (prefix_path or [])]
    for act in prefix_path:
        r = E.perform(g, act[0], act[1] if len(act) > 1 and act[1] else None)
    start_lc = r.levels_completed
    l5snap = pickle.dumps(g, protocol=5)

    def restore():
        gg = pickle.loads(l5snap); gg._clean_levels = _CLEAN[game]; return gg

    prog_ignore = set()
    if clean_progress:
        try:
            prog_ignore = E.detect_transient_scalars(game, level, cache)
        except Exception as _e:
            prog_ignore = set()
    lg.info(f"[CLEAN-PROG] L{level} ignoring {len(prog_ignore)} transient scalars: "
            f"{sorted(prog_ignore)[:8]}")

    def prog(gg):
        cur = dict(E.scalar_state(gg))
        return sum(1 for k, v in cur.items()
                   if k not in prog_ignore and root_scalar.get(k) != v)

    def cands(frame):
        return [(a, None) for a in E.MOVES] + E.dynamic_clicks(frame, limit=8)

    def priors_of(frame, acts):
        if policy_fn is None:
            return {_akey(*c): 1.0/len(acts) for c in acts}
        pr = policy_fn(frame)
        raw = {_akey(*c): max(1e-4, pr.get(_akey(*c), pr.get(c[0], 0.0))) for c in acts}
        s = sum(raw.values()) or 1.0
        return {k: v/s for k, v in raw.items()}

    def step(gg, a, data):
        rr = E.perform(gg, a, data); ctr.inc("perform")
        f = E.frame_of(rr)
        if f is None:
            return None, False, [(a, data)]
        steps = [(a, data)]
        win = bool(rr.levels_completed > start_lc or gg._current_level_index > level)
        if macro_moves and data is None and a in E.MOVES and not win:
            prevf = f; bp = prog(gg)
            for _ in range(macro_max - 1):
                rr2 = E.perform(gg, a, None); ctr.inc("perform")
                f2 = E.frame_of(rr2)
                if f2 is None or np.array_equal(f2, prevf):
                    break
                steps.append((a, None)); prevf = f2; f = f2
                if bool(rr2.levels_completed > start_lc or gg._current_level_index > level):
                    win = True; break
                if prog(gg) > bp:
                    break
        return f, win, steps

    def wm_imagine(frame):
        """Iteration-2: roll forward in the Puzzle-LM world model (feature space,
        NO engine render) to estimate future progress from this leaf. Returns a
        scalar lookahead bonus = sum of predicted progress probabilities."""
        feat = E.object_features(frame).astype(np.float32)
        imagined = 0.0
        for _ in range(imagine_len):
            if wm_policy is not None:
                p, _ = wm_policy.forward(feat); a_cls = int(np.argmax(p[0]))
            else:
                a_cls = random.randint(0, 4)
            nxt, prog, _ = world_model.forward(feat, [a_cls])
            imagined += float(prog[0])
            feat = nxt[0]
        return imagined

    lg.info(f"[AZ-START] {game} L{level} sims={sims} time={time_budget}s "
            f"c_puct={c_puct} micro_rollout={micro_rollout} chained={chained} "
            f"prefix={len(prefix_path)} macro={macro_moves} "
            f"world_model={'on' if world_model is not None else 'off'} "
            f"imagine_len={imagine_len if world_model is not None else '-'}")

    root = ZNode()
    f0 = E.frame_of(r)
    root.acts = cands(f0); root.priors = priors_of(f0, root.acts)
    root.progress = prog(g); root.evaluated = True
    best_prog = root.progress; best_plan = list(prefix_path); solution = None
    sim = 0
    while sim < sims and (time.time() - t0) < time_budget:
        sim += 1
        gg = restore(); ctr.inc("restore")
        node = root; path_nodes = [root]; plan = list(prefix_path); cur_f = f0
        # --- PUCT selection to a leaf ---
        while True:
            if not node.acts:
                break
            sqrtN = math.sqrt(node.N + 1)
            best_u = -1e18; best_c = None
            for c in node.acts:
                ak = _akey(*c)
                ch = node.children.get(ak)
                q = (ch.W / ch.N) if (ch and ch.N) else 0.0
                u = q + c_puct * node.priors.get(ak, 1e-3) * sqrtN / (1 + (ch.N if ch else 0))
                if u > best_u:
                    best_u = u; best_c = c; best_ak = ak
            a, data = best_c
            cur_f, win, steps = step(gg, a, data); plan += steps
            child = node.children.get(best_ak)
            if child is None:
                child = ZNode(); node.children[best_ak] = child
            node = child; path_nodes.append(node)
            if win:
                solution = plan; break
            if cur_f is None:
                break
            if not node.evaluated:        # reached a new leaf -> expand+evaluate
                node.progress = prog(gg)
                node.acts = cands(cur_f)
                node.priors = priors_of(cur_f, node.acts)
                node.evaluated = True
                break
        # --- leaf evaluation (value net + micro-rollout, NO long rollout) ---
        reached = node.progress if cur_f is not None else 0
        if cur_f is not None and not solution and micro_rollout:
            for _ in range(micro_rollout):
                a2, d2 = (random.choice(E.MOVES), None)
                cur_f, win, steps = step(gg, a2, d2); plan += steps
                if cur_f is None:
                    break
                p = prog(gg); reached = max(reached, p)
                if win:
                    solution = plan; break
        if reached > best_prog:
            best_prog = reached; best_plan = list(plan); ctr.inc("best_updates")
        value = reached / max_progress
        if value_fn is not None and cur_f is not None:
            value += value_weight * float(value_fn(cur_f))
        if world_model is not None and cur_f is not None and not solution:
            value += wm_weight * wm_imagine(cur_f)        # imagination lookahead
            ctr.inc("wm_rollouts")
        if solution:
            value += 10.0
        for nd in path_nodes:
            nd.N += 1; nd.W += value
        if solution:
            break
        if sim % log_every == 0:
            dt = time.time() - t0
            lg.info(f"[AZ-PROG] sims={sim} best_progress={best_prog} "
                    f"root_children={len(root.children)} rate={sim/max(dt,1e-6):.0f}sim/s "
                    f"perform/s={ctr.get('perform')/max(dt,1e-6):.0f} elapsed={dt:.1f}s")

    dt = time.time() - t0
    status = "SOLVED" if solution else "TIMEOUT"
    lg.info(f"[AZ-END] L{level} status={status} sims={sim} best_progress={best_prog} "
            f"time={dt:.1f}s rate={sim/max(dt,1e-6):.0f}sim/s "
            f"perform/s={ctr.get('perform')/max(dt,1e-6):.0f} counters={ctr.summary()}")
    return {"game": game, "level": level, "strategy": "mcts-az", "status": status,
            "solution": [list(x) for x in solution] if solution else None,
            "solution_len": len(solution) if solution else None,
            "expansions": sim, "visited": sim, "best_depth": len(best_plan),
            "best_progress": best_prog, "time": round(dt, 2),
            "nodes_per_s": round(sim/max(dt, 1e-6), 2),
            "harvest": {"path": [list(x) for x in best_plan], "best_progress": best_prog}}


def solve_mcts(game, level, *, sims=4000, time_budget=30.0, rollout_len=40,
               c_uct=1.4, heuristic_fn=None, policy_fn=None, macro_moves=True,
               macro_max=8, mask=None, prefix_path=None, progress_bonus=10.0,
               clean_progress=True, iter_tag="v17mcts", log_every=400, logger=None):
    lg = logger or get_logger("mcts", iter_tag)
    ctr = Counters(); t0 = time.time()
    cache, _ = E.load_cache(game)
    g = E.load_game(game)
    if game not in _CLEAN:
        _CLEAN[game] = g._clean_levels
    r, chained = E.chain_to_level(g, level, cache)
    root_scalar = dict(E.scalar_state(g))
    prefix_path = [tuple(x) for x in (prefix_path or [])]
    for act in prefix_path:
        r = E.perform(g, act[0], act[1] if len(act) > 1 and act[1] else None)
    start_lc = r.levels_completed
    l5snap = pickle.dumps(g, protocol=5)         # restored once per simulation

    def restore():
        gg = pickle.loads(l5snap); gg._clean_levels = _CLEAN[game]
        return gg

    prog_ignore = set()
    if clean_progress:
        try:
            prog_ignore = E.detect_transient_scalars(game, level, cache)
        except Exception:
            prog_ignore = set()
    lg.info(f"[CLEAN-PROG] L{level} ignoring {len(prog_ignore)} transient scalars: "
            f"{sorted(prog_ignore)[:8]}")

    def prog(gg):
        cur = dict(E.scalar_state(gg))
        return sum(1 for k, v in cur.items()
                   if k not in prog_ignore and root_scalar.get(k) != v)

    def cands(frame):
        cs = [(a, None) for a in E.MOVES] + E.dynamic_clicks(frame, limit=8)
        if policy_fn is not None:
            pr = policy_fn(frame)
            cs.sort(key=lambda c: pr.get(_akey(*c), pr.get(c[0], 0.0)), reverse=True)
        return cs

    def step(gg, a, data):
        """apply one action (+macro for moves). returns (frame, win, steps)."""
        rr = E.perform(gg, a, data); ctr.inc("perform")
        f = E.frame_of(rr)
        if f is None:
            return None, False, [(a, data)]
        steps = [(a, data)]
        win = bool(rr.levels_completed > start_lc or gg._current_level_index > level)
        if macro_moves and data is None and a in E.MOVES and not win:
            prevf = f; bp = prog(gg)
            for _ in range(macro_max - 1):
                rr2 = E.perform(gg, a, None); ctr.inc("perform")
                f2 = E.frame_of(rr2)
                if f2 is None or np.array_equal(f2, prevf):
                    break
                steps.append((a, None)); prevf = f2; f = f2
                if bool(rr2.levels_completed > start_lc or gg._current_level_index > level):
                    win = True; break
                if prog(gg) > bp:
                    break
        return f, win, steps

    lg.info(f"[MCTS-START] {game} L{level} sims={sims} time={time_budget}s "
            f"rollout_len={rollout_len} c_uct={c_uct} chained={chained} "
            f"prefix={len(prefix_path)} macro={macro_moves}")

    root = MNode()
    f0 = E.frame_of(r)
    root.untried = cands(f0)
    best_prog = 0; best_plan = list(prefix_path); solution = None
    sim = 0
    while sim < sims and (time.time() - t0) < time_budget:
        sim += 1
        gg = restore(); ctr.inc("restore")
        node = root
        plan = list(prefix_path)
        path_nodes = [root]
        cur_f = f0
        # --- selection: descend fully-expanded nodes by UCT ---
        while not node.untried and node.children:
            best_a, best_c, best_u = None, None, -1e18
            for ak, ch in node.children.items():
                q = ch.W / ch.N if ch.N else 0.0
                u = q + c_uct * math.sqrt(math.log(node.N + 1) / (ch.N + 1))
                if u > best_u:
                    best_u, best_a, best_c = u, ak, ch
            a, data = (best_a, None) if not isinstance(best_a, tuple) else (6, {"x": best_a[1], "y": best_a[2], "game_id": "bfs"})
            cur_f, win, steps = step(gg, a, data)
            plan += steps
            node = best_c; path_nodes.append(node)
            if win:
                solution = plan; break
            if cur_f is None:
                break
        if solution:
            break
        # --- expansion ---
        if cur_f is not None and node.untried:
            a, data = node.untried.pop(0)
            cur_f, win, steps = step(gg, a, data)
            plan += steps
            child = MNode(); node.children[_akey(a, data)] = child
            node = child; path_nodes.append(node)
            if cur_f is not None:
                child.progress = prog(gg)
                child.untried = cands(cur_f)
            if win:
                solution = plan; break
        # --- LIGHT rollout: cheap random-move playout (no policy/feature calls
        # in the hot loop — that was the throughput killer). The smart policy is
        # used only at tree nodes; rollouts just need a fast value estimate.
        # Occasional click keeps key-selection reachable. ---
        reached = prog(gg) if cur_f is not None else 0
        if cur_f is not None and not solution:
            for _ in range(rollout_len):
                if random.random() < 0.12:
                    cl = E.dynamic_clicks(cur_f, limit=6)
                    a, data = random.choice(cl) if cl else (random.choice(E.MOVES), None)
                else:
                    a, data = random.choice(E.MOVES), None
                cur_f, win, steps = step(gg, a, data)
                plan += steps
                if cur_f is None:
                    break
                p = prog(gg)
                if p > reached:
                    reached = p
                if win:
                    solution = plan; break
        # --- record best landmark + backprop ---
        if reached > best_prog:
            best_prog = reached; best_plan = list(plan); ctr.inc("best_updates")
        reward = reached * progress_bonus
        if heuristic_fn is not None and cur_f is not None:
            reward += max(0.0, 5.0 - 0.1 * float(heuristic_fn(cur_f)))
        if solution:
            reward += 1000.0
        for nd in path_nodes:
            nd.N += 1; nd.W += reward
            nd.best_prog = max(nd.best_prog, reached)
        if solution:
            break
        if sim % log_every == 0:
            dt = time.time() - t0
            lg.info(f"[MCTS-PROG] sims={sim} best_progress={best_prog} "
                    f"tree_root_children={len(root.children)} "
                    f"rate={sim/max(dt,1e-6):.0f}sim/s perform/s="
                    f"{ctr.get('perform')/max(dt,1e-6):.0f} elapsed={dt:.1f}s")

    dt = time.time() - t0
    status = "SOLVED" if solution else "TIMEOUT"
    lg.info(f"[MCTS-END] L{level} status={status} sims={sim} best_progress={best_prog} "
            f"time={dt:.1f}s rate={sim/max(dt,1e-6):.0f}sim/s "
            f"perform/s={ctr.get('perform')/max(dt,1e-6):.0f} counters={ctr.summary()}")
    return {"game": game, "level": level, "strategy": "mcts", "status": status,
            "solution": [list(x) for x in solution] if solution else None,
            "solution_len": len(solution) if solution else None,
            "expansions": sim, "visited": sim, "best_depth": len(best_plan),
            "best_progress": best_prog, "time": round(dt, 2),
            "nodes_per_s": round(sim / max(dt, 1e-6), 2),
            "harvest": {"path": [list(x) for x in best_plan], "best_progress": best_prog}}
