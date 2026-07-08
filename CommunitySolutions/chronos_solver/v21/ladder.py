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
