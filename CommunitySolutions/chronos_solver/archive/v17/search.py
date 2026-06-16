"""v17 search — the BFS/ForgeNet core, ported from v13's best parts and
extended with a pluggable heuristic + policy prior (the v17 contribution).

Modes (selected by `strategy`):
  * "bfs"     : plain FIFO breadth-first — reproduces v13's breadth-death on
                L5, the baseline iteration-1 measures.
  * "greedy"  : best-first on a learned cost-to-go heuristic h(frame).
  * "astar"   : weighted A* — priority = depth + w * h(frame).
  * "puct"    : AlphaZero-style — frontier ordered by policy-prior * value,
                branching pruned to the policy's top-k actions.

Ported v13 lessons (see v13/README):
  * chained true-baseline start state (engine.chain_to_level)         [fix #4]
  * transient HUD-row mask in the dedup hash (engine.detect_transient) [fix #2]
  * scalar-attr state identity (key shape/color/rot, countdowns)       [fix #3]
  * dynamic click targets from the CURRENT frame's object centroids
  * deep BFS (no depth cap; visited-dedup bounds the search)           [fix #1]

v17 additions:
  * slim snapshots (drop the constant _clean_levels -> ~2x faster restore)
  * pluggable heuristic_fn(frame)->cost  and policy_fn(frame)->{action:prob}
  * full search-trace emission for training ForgeNet + the TRM LM
  * node/second + frontier telemetry logged every `log_every` expansions
"""
from __future__ import annotations
import heapq, pickle, time, json, os, random
from collections import deque, defaultdict
import numpy as np

import engine as E
from vlog import get_logger, Counters

# shared constant level templates — popped from every snapshot, restored after
_CLEAN = {}


def _snap(g):
    saved = g._clean_levels
    g._clean_levels = None
    b = pickle.dumps(g, protocol=5)
    g._clean_levels = saved
    return b


def _restore(b, game):
    g = pickle.loads(b)
    g._clean_levels = _CLEAN[game]
    return g


class Node:
    __slots__ = ("snap", "frame", "path", "depth", "hash", "h", "prio", "progress", "novel")

    def __init__(self, snap, frame, path, depth, h):
        self.snap = snap
        self.frame = frame
        self.path = path
        self.depth = depth
        self.hash = h
        self.h = 0.0
        self.prio = 0.0
        self.progress = 0          # # of game scalar-attrs changed vs level start
        self.novel = False         # BFWS: did this state make some atom true 1st time


def _progress(g, root_scalar):
    """v17 progress signal = how many of the engine's hidden scalar attrs
    (key colour/rotation, goal-match flags, countdowns) differ from the level
    start. Real key/lock interactions move these; this is the L5 dual-key win
    signal the L0-L4 heuristic can't have learned."""
    cur = dict(E.scalar_state(g))
    return sum(1 for k, v in cur.items() if root_scalar.get(k) != v)


def candidate_actions(frame, policy_fn=None, topk=None, click_limit=10):
    """MOVES + dynamic clicks; optionally reordered/pruned by a policy prior."""
    cands = [(a, None) for a in E.MOVES] + E.dynamic_clicks(frame, limit=click_limit)
    if policy_fn is not None:
        probs = policy_fn(frame)                       # {action_key: prob}
        def keyf(c):
            k = c[0] if c[1] is None else (6, c[1]["x"], c[1]["y"])
            return probs.get(k, probs.get(c[0], 0.0))
        cands.sort(key=keyf, reverse=True)
        if topk:
            cands = cands[:topk]
    return cands


def solve_level(game, level, *, strategy="bfs", node_budget=2000,
                time_budget=30.0, heuristic_fn=None, policy_fn=None,
                policy_topk=6, astar_w=1.0, mask=None, iter_tag="v17",
                log_every=200, emit_trace=False, checkpoint=None,
                progress_weight=0.0, harvest_k=0, prefix_path=None,
                novelty_bins=0, novelty_weight=0.0,
                macro_moves=False, macro_max=10, explore_p=0.0, logger=None):
    """Search for an action sequence that completes `level`. Returns a dict
    with the solution (if any) + rich telemetry. Heavily logged."""
    lg = logger or get_logger("search", iter_tag)
    ctr = Counters()
    t0 = time.time()

    g = E.load_game(game)
    if game not in _CLEAN:
        gtmp = E.load_game(game)
        E.chain_to_level(gtmp, level, E.load_cache(game)[0])
        _CLEAN[game] = gtmp._clean_levels
    cache, _ = E.load_cache(game)
    r, chained = E.chain_to_level(g, level, cache)
    # progress is measured vs the ABSOLUTE level start (captured here, before
    # any re-rooting prefix is replayed) so it accumulates across subgoals.
    root_scalar = dict(E.scalar_state(g))
    prefix_path = prefix_path or []
    for act in prefix_path:                       # re-root at a subgoal landmark
        a = act[0]; d = act[1] if len(act) > 1 and act[1] else None
        r = E.perform(g, a, d)
    f0 = E.frame_of(r)
    start_lc = r.levels_completed
    root_hash = E.state_hash(g, f0, mask)
    lg.info(f"[START] game={game} L{level} strategy={strategy} chained={chained} "
            f"prefix={len(prefix_path)} start_lc={start_lc} node_budget={node_budget} "
            f"time_budget={time_budget}s astar_w={astar_w} "
            f"root_progress={sum(1 for k,v in E.scalar_state(g) if root_scalar.get(k)!=v)} "
            f"policy_topk={policy_topk if policy_fn else '-'}")
    root = Node(_snap(g), f0, [], 0, 0.0)
    root.hash = root_hash
    visited = {root_hash: 0}
    trace = [] if emit_trace else None
    best_node = root                     # most-progressed-then-deepest node
    best_score = (0, 0)
    # IW(1) novelty (Lipovetzky & Geffner): a state is novel if some atom
    # (feature_index, discretized_value) is seen for the first time. Non-novel
    # states are pruned -> the frontier stays tiny and the search reaches deep.
    seen_atoms = set()

    def atoms_of(frame):
        # IW pixel-atoms (à la Atari B-PROST): downsample to GxG cells, atom =
        # (cell_index, colour). Far richer than coarse object features, so
        # width-1 novelty stays productive instead of exhausting immediately.
        G = novelty_bins
        h_, w_ = frame.shape
        bh, bw = max(1, h_ // G), max(1, w_ // G)
        atoms = set()
        idx = 0
        for i in range(0, h_, bh):
            for j in range(0, w_, bw):
                block = frame[i:i+bh, j:j+bw]
                c = int(np.bincount(block.flatten(), minlength=16).argmax())
                atoms.add((idx, c))
                idx += 1
        return atoms

    def is_novel(frame):
        a = atoms_of(frame)
        new = a - seen_atoms
        if new:
            seen_atoms.update(new)
            return True
        return False

    if novelty_bins:
        is_novel(f0)

    # frontier
    use_heap = strategy in ("greedy", "astar", "puct")
    # type-based exploration (Xie et al. 2014): a second set of queues bucketed
    # by "type" = progress level. With prob explore_p we pop from the
    # highest-progress bucket instead of the heuristic heap, so a heuristic
    # plateau (all h=0) can't starve the most-progressed frontier.
    buckets = defaultdict(deque)
    done = set()
    if use_heap:
        frontier = []
        counter = 0
        heapq.heappush(frontier, (0.0, counter, root))
    else:
        frontier = deque([root])

    best_depth = 0
    best_h = float("inf")
    expansions = 0
    solution = None

    def push(node):
        nonlocal counter
        if use_heap:
            if heuristic_fn is not None:
                node.h = float(heuristic_fn(node.frame))
            if strategy == "greedy":
                pr = node.h
            elif strategy == "astar":
                pr = node.depth + astar_w * node.h
            else:  # puct: lower prio = better; use -value as proxy via heuristic
                pr = node.h - 0.01 * node.depth
            pr -= progress_weight * node.progress      # reward key/lock progress
            pr -= novelty_weight * (1.0 if node.novel else 0.0)   # BFWS bonus
            node.prio = pr
            counter += 1
            heapq.heappush(frontier, (pr, counter, node))
        else:
            frontier.append(node)
        if explore_p:
            buckets[node.progress].append(node)

    counter = 0
    while (frontier or any(buckets.values())) and expansions < node_budget \
            and (time.time() - t0) < time_budget:
        node = None
        if explore_p and buckets and random.random() < explore_p:
            for prog in sorted(buckets, reverse=True):     # most-progressed type
                while buckets[prog]:
                    cand = buckets[prog].popleft()
                    if cand.hash not in done:
                        node = cand; ctr.inc("explore_pops"); break
                if node:
                    break
        if node is None:
            if use_heap:
                if not frontier:
                    if not any(buckets.values()):
                        break
                    continue
                _, _, node = heapq.heappop(frontier)
            else:
                node = frontier.popleft()
        if node.hash in done:
            continue
        done.add(node.hash)
        expansions += 1
        ctr.inc("expansions")

        g = _restore(node.snap, game)
        cands = candidate_actions(node.frame, policy_fn,
                                  topk=policy_topk if strategy == "puct" else None)
        for (a, data) in cands:
            gg = _restore(node.snap, game)
            rr = E.perform(gg, a, data)
            ctr.inc("child_evals")
            f = E.frame_of(rr)
            if f is None:
                ctr.inc("noop_frames"); continue
            steps = [(a, data)]
            win = bool(rr.levels_completed > start_lc or gg._current_level_index > level)
            # MACRO-ACTION (move-until-wall): repeat a movement action until the
            # frame stops changing (wall), a progress event fires, or a win.
            # Collapses corridor walks -> linear depth cut, exponential speedup.
            if macro_moves and data is None and a in E.MOVES and not win:
                prevf = f
                base_prog = _progress(gg, root_scalar)
                for _ in range(macro_max - 1):
                    rr2 = E.perform(gg, a, None)
                    ctr.inc("child_evals")
                    f2 = E.frame_of(rr2)
                    if f2 is None:
                        break
                    w2 = bool(rr2.levels_completed > start_lc or gg._current_level_index > level)
                    if np.array_equal(f2, prevf):
                        break                       # wall — stop, drop the no-op
                    steps.append((a, None)); prevf = f2; f = f2; rr = rr2
                    if w2:
                        win = True; break
                    if _progress(gg, root_scalar) > base_prog:
                        break                       # stop at a key/lock event
                    ctr.inc("macro_steps")
            h = E.state_hash(gg, f, mask)
            depth = node.depth + len(steps)
            outcome = "new"
            if win:
                outcome = "WIN"
            elif h in visited:
                outcome = "visited"
            if emit_trace:
                trace.append({"p": node.hash, "a": a if data is None else [6, data["x"], data["y"]],
                              "c": h, "d": depth, "o": outcome})
            if win:
                solution = [tuple(x) for x in prefix_path] + node.path + steps
                lg.info(f"[WIN] L{level} solved at depth {depth} after {expansions} "
                        f"expansions, {ctr.get('child_evals')} child-evals, "
                        f"{time.time()-t0:.1f}s. solution_len={len(solution)}")
                break
            if h in visited:
                continue
            visited[h] = depth
            child = Node(_snap(gg), f, node.path + steps, depth, 0.0)
            child.hash = h
            child.progress = _progress(gg, root_scalar)
            if novelty_bins:
                child.novel = is_novel(f)
                if child.novel:
                    ctr.inc("novel_states")
            push(child)
            if depth > best_depth:
                best_depth = depth
            score = (child.progress, depth)
            if score > best_score:
                best_score = score
                best_node = child
                ctr.inc("best_updates")
        if solution:
            break

        if heuristic_fn is not None and node.h < best_h:
            best_h = node.h
        if expansions % log_every == 0:
            dt = time.time() - t0
            fsz = len(frontier)
            lg.info(f"[PROG] exp={expansions} visited={len(visited)} frontier={fsz} "
                    f"depth(best)={best_depth} progress(best)={best_score[0]} "
                    f"h(best)={best_h if best_h<1e9 else '-'} "
                    f"rate={expansions/max(dt,1e-6):.1f}n/s elapsed={dt:.1f}s "
                    f"counters={ctr.summary()}")
            if checkpoint:
                _save_ckpt(checkpoint, visited, expansions, best_depth)

    dt = time.time() - t0
    status = "SOLVED" if solution else ("BUDGET" if expansions >= node_budget else
                                        ("TIMEOUT" if dt >= time_budget else "EXHAUSTED"))
    lg.info(f"[END] L{level} status={status} expansions={expansions} "
            f"visited={len(visited)} best_depth={best_depth} "
            f"best_progress={best_score[0]} time={dt:.1f}s "
            f"final_counters={ctr.summary()}")
    # harvest the best path as a SoS/ExIt partial demonstration (frames along
    # the most-progressed path get bootstrapped cost-to-go labels downstream)
    harvest = None
    if harvest_k and best_node is not root:
        full = [tuple(x) for x in prefix_path] + best_node.path
        harvest = {"path": [[a, d] for (a, d) in full],
                   "best_progress": best_score[0],
                   "best_path_depth": len(full)}
    return {
        "game": game, "level": level, "strategy": strategy, "status": status,
        "solution": [[a, d] for (a, d) in solution] if solution else None,
        "solution_len": len(solution) if solution else None,
        "expansions": expansions, "visited": len(visited),
        "best_depth": best_depth, "best_progress": best_score[0],
        "time": round(dt, 2),
        "nodes_per_s": round(expansions / max(dt, 1e-6), 2),
        "trace": trace, "harvest": harvest,
    }


def _save_ckpt(path, visited, expansions, best_depth):
    try:
        json.dump({"visited": len(visited), "expansions": expansions,
                   "best_depth": best_depth}, open(path, "w"))
    except Exception:
        pass
