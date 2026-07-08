# =====================================================================
# v21 ladder.py — Go-Explore / macro-BFS suffix search for DEEP levels.
#
# The frontier for ls20 L5–L6: plain single-step BFS explodes on long keyboard
# corridors. `macro_bfs` searches from the (already re-rooted) L4 end-state where
# each frontier edge is EITHER a single action OR a MACRO — the same action
# repeated until the state hash stops changing (corridor collapse) or a cap. This
# gives BFS both the reach of macros (deep corridors become one edge) AND the
# precision of single steps (so it can stop mid-corridor to turn — the documented
# ls20 macro-overshoot pitfall, v19 PROGRESS log).
#
# PURE + offline-testable: ALL engine interaction is via injected closures, so the
# synthetic-maze self-test runs with no arcengine (like blitz.py / runtime_coder).
#   clone(state)        -> a fresh independent fork of state
#   play(state, step)   -> mutates `state` by one (action_id, data); returns int
#                          levels_completed (goal reached when >= `goal`)
#   hash_fn(state)      -> a hashable dedup key (e.g. masked-frame hash)
# =====================================================================
from collections import deque


def macro_bfs(start, clone, play, actions, hash_fn, goal,
              max_states=200000, max_macro=64, use_macros=True, click_targets=None):
    """Shortest action sequence (list of (action_id, data)) from `start` that reaches
    levels_completed >= goal, or None. Frontier BFS with corridor-collapsing macros
    and hash dedup. `actions` = simple action ids to try; `click_targets` = optional
    list of {'x','y'} for ACTION6 single-clicks."""
    start_key = hash_fn(start)
    seen = {start_key}
    frontier = deque([(start, [])])
    budget = [max_states]

    def _try(state, path, step):
        """Apply one step to a fork; return (new_state, path+step, completed_bool)."""
        g = clone(state)
        lc = play(g, step)
        budget[0] -= 1
        return g, path + [step], lc >= goal

    while frontier and budget[0] > 0:
        state, path = frontier.popleft()

        # -- single-step edges (precision: can stop to turn) --------------------
        steps = [(a, None) for a in actions]
        for t in (click_targets or []):
            steps.append((6, dict(t)))
        for step in steps:
            g, npath, done = _try(state, path, step)
            if done:
                return npath
            k = hash_fn(g)
            if k not in seen:
                seen.add(k); frontier.append((g, npath))
            if budget[0] <= 0:
                return None

        # -- macro edges (reach: collapse a corridor into one push) -------------
        if use_macros:
            for a in actions:
                gm = clone(state)
                mpath = list(path)
                lastk = hash_fn(gm)
                grew = False
                for _ in range(max_macro):
                    lc = play(gm, (a, None)); budget[0] -= 1
                    mpath.append((a, None))
                    if lc >= goal:
                        return mpath
                    nk = hash_fn(gm)
                    if nk == lastk:            # frame stopped changing -> corridor end
                        break
                    lastk = nk; grew = True
                    if budget[0] <= 0:
                        return None
                if grew and lastk not in seen:
                    seen.add(lastk); frontier.append((gm, mpath))
    return None


def go_explore(start, clone, play, actions, cell_fn, goal,
               max_states=200000, max_macro=64, use_macros=True,
               click_targets=None, action_order=None, seed_plans=None):
    """Cell-archive Go-Explore for DEEP levels (Epic C1 — the ls20 L5-L6 lever).

    `macro_bfs` above is an exhaustive frontier BFS: it dedups on the FULL frame
    hash, so ls20 L5's long corridors blow the frontier up to ~19k live states.
    Go-Explore instead keeps ONE representative per COARSE `cell_fn(state)` cell,
    remembers the SHORTEST path that reached it, and repeatedly *returns to a
    promising cell* (fewest visits, then shortest path) to explore onward. Coarse
    cells merge near-identical frames, so the archive stays small while reach grows
    — the documented Go-Explore win on sparse deep tasks.

    Same injected-closure contract as `macro_bfs`, except dedup is on the coarse
    `cell_fn(state)` (e.g. a downsampled-frame signature) rather than the exact
    frame hash:
      clone(state)  -> independent fork
      play(state, step) -> int levels_completed (mutates the fork; win at >= goal)
      cell_fn(state) -> hashable COARSE cell key

    Optional guidance (both default to plain behaviour if omitted):
      action_order -> toddler intuition: simple action ids tried in this order first
      seed_plans   -> verified fragments (blackboard) replayed to prime the archive

    Returns the shortest winning [(action_id, data), ...] found, or None. Pure:
    no engine / network / global state, so the synthetic-maze self-test runs with
    no arcengine (like `macro_bfs`)."""
    budget = [max_states]
    # honour the toddler order but never invent actions outside `actions`
    ordered = [a for a in (action_order or []) if a in actions]
    acts = ordered + [a for a in actions if a not in ordered]
    clicks = [(6, dict(t)) for t in (click_targets or [])]
    steps = [(a, None) for a in acts] + clicks

    # archive: cell -> {"state","path"};  visits: cell -> int
    archive = {cell_fn(start): {"state": start, "path": []}}
    visits = {}

    def _consider(state, path):
        """Register a reached (state,path): keep the shortest path per cell."""
        c = cell_fn(state)
        cur = archive.get(c)
        if cur is None or len(path) < len(cur["path"]):
            archive[c] = {"state": state, "path": path}

    def _norm(seq):
        return [tuple(s) if isinstance(s, list) else s for s in seq]

    # -- prime the archive with Go-Explore seeds (blackboard fragments) -----------
    for seed in (seed_plans or []):
        if not seed:
            continue
        if budget[0] <= 0:
            break
        g = clone(start); lc = 0
        for step in seed:
            lc = play(g, tuple(step) if isinstance(step, list) else step)
            budget[0] -= 1
            if budget[0] <= 0 or lc >= goal:
                break
        if lc >= goal:
            return _norm(seed)
        _consider(g, _norm(seed))

    # over-exploration cap: once a cell has been expanded past its branch factor it
    # carries no new information, so drop it to keep the "return-to" set promising.
    max_visits = 1 + len(steps)

    while budget[0] > 0 and archive:
        # return to the most promising cell: fewest visits, then shortest path
        cell = min(archive, key=lambda c: (visits.get(c, 0), len(archive[c]["path"])))
        visits[cell] = visits.get(cell, 0) + 1
        if visits[cell] > max_visits:
            del archive[cell]
            continue
        base = archive[cell]
        base_state, base_path = base["state"], base["path"]

        # single-step edges (precision: can stop mid-corridor to turn)
        for step in steps:
            if budget[0] <= 0:
                return None
            g = clone(base_state); lc = play(g, step); budget[0] -= 1
            npath = base_path + [step]
            if lc >= goal:
                return npath
            _consider(g, npath)

        # macro edges (reach: sweep a corridor, dropping a breadcrumb in every NEW
        # cell along the way). Unlike macro_bfs (fine frame hash, stop on first
        # no-change), coarse cells change only every few steps, so we keep going
        # through same-cell steps and only give up after `patience` consecutive
        # no-change steps (stuck against a wall). Each new cell reached is archived
        # with its own fork, so a later turn can start from precisely there.
        if use_macros:
            patience = 3
            for a in acts:
                if budget[0] <= 0:
                    return None
                gm = clone(base_state); mpath = list(base_path)
                lastc = cell_fn(gm); stagnant = 0
                for _ in range(max_macro):
                    lc = play(gm, (a, None)); budget[0] -= 1
                    mpath.append((a, None))
                    if lc >= goal:
                        return mpath
                    nc = cell_fn(gm)
                    if nc == lastc:
                        stagnant += 1
                        if stagnant >= patience:   # wall / stuck -> end the corridor
                            break
                    else:                          # entered a new cell -> breadcrumb
                        stagnant = 0; lastc = nc
                        _consider(clone(gm), list(mpath))
                    if budget[0] <= 0:
                        return None
    return None
