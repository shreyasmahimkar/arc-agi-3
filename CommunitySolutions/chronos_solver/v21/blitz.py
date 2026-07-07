#!/usr/bin/env python3
# =====================================================================
# Chronos v21 — Blitz Stage-0 (BACKLOG #2)
#
# "Race cheap wins on a fork first." Before spending the full BFS budget on a
# level, try the small handful of trivially-shallow plans that crack many
# reflex/orchestration levels with ZERO wasted actions:
#   (1) each simple action once            (length-1 plan)
#   (2) each click target once (ACTION6)   (length-1 plan)
#   (3) repeat a single action ×K          (shortest k that wins)
#   (4) repeat a single click target ×K    (vc33-style: hammer one component)
#
# The core `blitz_solve` is PURE — it takes injected `clone`/`play` closures and
# a list of candidate actions, so it is fully offline-testable with a mock game
# (no arcengine / numpy / torch import). `blitz_for_solver` is the thin adapter
# that binds it to v19's read-only `BFSSolver` on the Mac; its engine imports are
# lazy so `import blitz` stays dependency-free.
#
# Every plan this returns is still routed through `BFSSolver.verify_solution`
# and the shortest-plan corpus gate by the caller — blitz only PROPOSES.
# =====================================================================


def blitz_solve(start_game, target_level, simple_actions, click_targets,
                clone, play, repeat_K=200):
    """Shortest cheap winning plan for `target_level`, or None.

    Args:
      start_game:     game object already positioned at the level's start state.
      target_level:   level index we want completed (win == levels_completed
                      reaching target_level + 1).
      simple_actions: iterable of directional/interact action ids (e.g. 1..5).
      click_targets:  iterable of ACTION6 `data` dicts (e.g. {'x':.,'y':.}).
      clone(game):    -> an independent deep copy (a fresh fork each try).
      play(game,(aid,data)) -> int levels_completed AFTER performing the action
                      on `game` (MUTATES game; caller passes a fork).
      repeat_K:       max repeats to try for the repeat-action tier.

    Returns the shortest plan `[(aid, data), ...]` whose replay reaches the goal,
    preferring length-1 wins. Tiers, cheapest-first: each simple action once,
    each click once, repeat-one-action ×K, repeat-one-click ×K (the last cracks
    vc33-style walls that finish by hammering a single component). Pure: no
    engine/network/global state.
    """
    goal = target_level + 1
    simple_actions = list(simple_actions or [])
    click_targets = list(click_targets or [])

    # Tier 1a: each simple action once (length-1 — can't be beaten, return now).
    for aid in simple_actions:
        g = clone(start_game)
        if _completed(play(g, (aid, None))) >= goal:
            return [(aid, None)]

    # Tier 1b: each click target once (also length-1).
    for data in click_targets:
        g = clone(start_game)
        if _completed(play(g, (6, data))) >= goal:
            return [(6, data)]

    # Tier 2: repeat a single action up to K times; keep the shortest winner.
    best = None
    for aid in simple_actions:
        g = clone(start_game)
        for k in range(1, repeat_K + 1):
            if _completed(play(g, (aid, None))) >= goal:
                if best is None or k < len(best):
                    best = [(aid, None)] * k
                break

    # Tier 3: repeat a single CLICK target ×K; keep the shortest winner.
    # vc33-style orchestration walls often finish by hammering ONE component
    # many times (its verified solutions end in runs like 9× the same ACTION6
    # coord). Plain BFS branches over every click target at every depth and
    # times out; a fixed-coord repeat is a depth-K line search blitz wins in
    # ≤K probes per target. Tier 1b already tested k=1 for each target, so a
    # repeat only helps when k≥2 — but we start at 1 to mirror Tier 2 and stay
    # self-contained (the k=1 re-probe is a single cheap fork step per target).
    for data in click_targets:
        cap = repeat_K if best is None else min(repeat_K, len(best) - 1)
        if cap < 1:
            continue  # can't beat an already-found shorter plan
        g = clone(start_game)
        for k in range(1, cap + 1):
            if _completed(play(g, (6, data))) >= goal:
                if best is None or k < len(best):
                    best = [(6, data)] * k
                break
    return best


def blitz_macros(start_game, target_level, macros, clone, play):
    """Shortest known-good MACRO plan that wins `target_level`, or None.

    A "macro" is a full plan `[(aid, data), ...]` harvested from an ALREADY-SOLVED
    sibling level (same game) — the cheapest Go-Explore seed there is (BACKLOG
    #4/#9): sibling levels of a game often share mechanics, so replaying a
    sibling's verified solution verbatim can crack a wall with ZERO search. Each
    macro is replayed on a fresh fork; on a win we keep only the shortest winning
    PREFIX (macros can overshoot the goal). Preferring shorter winners keeps the
    shortest-plan corpus gate happy.

    Pure: no engine/network/global state (injected `clone`/`play` closures). Every
    returned plan is still routed through `verify_solution` + the shortest gate by
    the caller — this only PROPOSES a replay.

    Args:
      start_game:    game positioned at the level's start state.
      target_level:  level index to complete (win == levels_completed >= +1).
      macros:        iterable of candidate plans (sibling-level solutions).
      clone/play:    same closures as `blitz_solve`.
    """
    goal = target_level + 1
    best = None
    for plan in (macros or []):
        if not plan:
            continue
        # A macro can't beat the current best if it's already at least as long.
        if best is not None and len(plan) >= len(best):
            continue
        g = clone(start_game)
        win_len = None
        for i, step in enumerate(plan):
            if _completed(play(g, step)) >= goal:
                win_len = i + 1
                break
        if win_len is not None and (best is None or win_len < len(best)):
            best = list(plan[:win_len])
    return best


def merge_click_targets(scan_clicks, frame, use_perception,
                        perception_fn=None, limit=None):
    """Merge engine-scanned ACTION6 click targets with perception's connected-
    component centroids (BACKLOG Epic B / B1), deduped by (x, y).

    v19's `_scan_actions` sources clicks from per-colour medians, so several
    spatially-separate blobs of the SAME colour collapse to one point that can
    land on background (between them). vc33-style click walls need a click ON
    each distinct component; `brain.perception.click_targets` gives one target
    per connected component. This helper appends those perception targets after
    the scan targets (scan first — they're the proven default), skipping any
    (x, y) already present, so the default ordering is preserved and only NEW,
    otherwise-missed component centroids are added.

    When `use_perception` is falsy (default), `scan_clicks` is returned de-duped
    but otherwise unchanged — so the wiring is a no-op unless V21_BRAIN_PERCEPTION
    is set. Pure: `perception_fn` defaults to `brain.perception.click_targets`
    but is injectable so this is fully offline-testable without a frame engine.
    Every returned target is still only a PROPOSAL — the caller verifies any
    plan a click seeds.
    """
    out, seen = [], set()
    for d in (scan_clicks or []):
        try:
            key = (d.get("x"), d.get("y"))
        except AttributeError:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    if use_perception and frame is not None:
        if perception_fn is None:
            from brain.perception import click_targets as perception_fn
        try:
            extra = perception_fn(frame)
        except Exception:
            extra = []
        for d in (extra or []):
            key = (d.get("x"), d.get("y"))
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            if limit is not None and len(out) >= limit:
                break
    return out


def _completed(v):
    """Coerce a play() return into an int levels_completed (defensive)."""
    try:
        return int(v)
    except Exception:
        return 0


# --------------------------------------------------------------------------
# Solver adapter (Mac-only). Kept out of module import so the offline test can
# `import blitz` without arcengine / numpy present.
# --------------------------------------------------------------------------
def blitz_for_solver(solver, level_idx, repeat_K=200):
    """Run blitz Stage-0 against a loaded v19 `BFSSolver` for `level_idx`.

    Builds the level's TRUE chained start state (reusing the solver's own
    `_make_start_state`), enumerates candidate simple actions + effective click
    targets via the solver's `_scan_actions`, then delegates to `blitz_solve`.
    Returns a candidate plan (UNVERIFIED — the caller verifies) or None.
    """
    from combined_agent import ActionInput, GameAction  # lazy: Mac-only deps
    import numpy as np

    # --- build the start state (chained for lvl>0, fresh reset for lvl 0) ---
    game, f0 = None, None
    try:
        res = solver._make_start_state(level_idx)
        if res is not None:
            game, f0 = res
    except Exception:
        game = None
    if game is None:
        game = solver.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])

    avail = list(getattr(game, "_available_actions", []) or [])
    simple = [a for a in avail if a <= 5]

    # Effective ACTION6 click targets (dedup by resulting-frame effect) reuse
    # the solver's scan so we only probe clicks that actually change something.
    clicks = []
    if 6 in avail:
        try:
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            for a, d in solver._scan_actions(game, f0, bg):
                if a == 6 and d is not None:
                    clicks.append(d)
        except Exception:
            clicks = []

    # B1: augment engine-scanned clicks with perception connected-component
    # centroids (one per distinct blob) so same-colour walls (vc33 L4–L6) get a
    # click ON each component, not the per-colour median between them. Env-gated
    # OFF by default — zero change to the proven default path unless opted in.
    import os as _os
    _use_perc = _os.environ.get("V21_BRAIN_PERCEPTION", "0") \
        not in ("", "0", "false", "False", "no", "off")
    if _use_perc:
        try:
            clicks = merge_click_targets(clicks, f0, True)
        except Exception:
            pass  # fall through to the scan-only clicks on any error

    def _clone(g):
        return solver._restore(solver._snap(g))

    def _play(g, step):
        aid, data = step
        ai = (ActionInput(id=GameAction.from_id(aid), data=data)
              if data else ActionInput(id=GameAction.from_id(aid)))
        r = g.perform_action(ai, raw=True)
        return int(getattr(r, "levels_completed", 0) or 0)

    # Tier 0: replay ALREADY-SOLVED sibling-level plans as Go-Explore seeds. Only
    # macros from OTHER (solved) levels of this game — never the target itself.
    macros = []
    sols = getattr(solver, "solutions", {}) or {}
    for lk, plan in sols.items():
        try:
            same = int(lk) == int(level_idx)
        except Exception:
            same = (lk == level_idx)
        if same or not plan:
            continue
        try:
            macros.append([(int(a), d) for a, d in plan])
        except Exception:
            continue
    m = blitz_macros(game, level_idx, macros, _clone, _play)
    if m:
        return m

    return blitz_solve(game, level_idx, simple, clicks, _clone, _play,
                       repeat_K=repeat_K)
