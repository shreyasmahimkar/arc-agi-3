#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Planner / MPC executor (Epic B, phase B3)  [interface]
#
# Two responsibilities, both grounded in Rodionov 2026:
#   1. plan_in_model: search a plan INSIDE the (unscored) executable world
#      model before spending any scored real-environment actions. The existing
#      blitz/BFS become planner "skills" invoked over the model here.
#   2. execute_and_verify: a model-predictive-control executor. It steps the
#      plan in the real environment AND the model in lockstep, comparing the
#      predicted next frame to the observed one; on the FIRST mismatch it
#      aborts and reports where — so plan execution doubles as an online test
#      of the world model (a wrong model reveals itself immediately, cheaply).
#
# Both cores are PURE (injected transition/goal/play closures, like blitz.py).
# =====================================================================


def plan_in_model(start_state, actions, transition, goal_reached, max_depth=64):
    """Breadth-first search for a shortest action plan that reaches the goal
    INSIDE the world model (no real environment actions spent).

    Args:
      transition(state, action) -> next_state   (model dynamics)
      goal_reached(state) -> bool               (induced goal)
      actions: candidate actions (each an (action_id, data) or opaque token).
      max_depth: plan-length cap.

    Returns the shortest winning plan [action, ...] or None. States are keyed
    by `repr` for the visited-set, so model states should be repr-stable. Pure.
    """
    from collections import deque
    if goal_reached(start_state):
        return []
    seen = {repr(start_state)}
    q = deque([(start_state, [])])
    while q:
        state, plan = q.popleft()
        if len(plan) >= max_depth:
            continue
        for a in actions:
            try:
                nxt = transition(state, a)
            except Exception:
                continue
            key = repr(nxt)
            if key in seen:
                continue
            seen.add(key)
            newplan = plan + [a]
            if goal_reached(nxt):
                return newplan
            q.append((nxt, newplan))
    return None


def plan_in_model_macro(start_state, actions, clone, play, hash_fn, goal=1,
                        max_states=200000, max_macro=64, click_targets=None):
    """DEEP-corridor planner skill (B3, the ls20 L5–L6 frontier).

    Single-step `plan_in_model` above cannot reach ls20 L5's depth (long keyboard
    corridors blow up BFS). This variant plans over the trusted model with
    Go-Explore / macro edges — the same action repeated until the state hash stops
    changing collapses a whole corridor into one edge — while still offering single
    steps so it can stop mid-corridor to turn. For WHITE-BOX games the engine IS
    the trusted model, so this searches the real dynamics directly (unscored, on
    forks). Delegates to the pure `ladder.macro_bfs`. Returns the shortest winning
    plan [(action_id, data), ...] or None.

    Contract (mutation-style, matches the engine):
      clone(state) -> independent fork;  play(state, step) -> int levels_completed
      (mutating the fork);  hash_fn(state) -> hashable dedup key.
    """
    try:
        import ladder
    except Exception:                       # packaged import fallback
        from .. import ladder               # type: ignore
    return ladder.macro_bfs(start_state, clone, play, actions, hash_fn, goal,
                            max_states=max_states, max_macro=max_macro,
                            click_targets=click_targets)


def execute_and_verify(plan, real_play, model_predict, observe,
                       compare=None, goal_completed=1):
    """Model-predictive execution of `plan` in the real environment.

    Steps the plan one action at a time: applies it in the real env, reads the
    observed frame, and compares to the model's predicted frame. Stops on the
    FIRST divergence (recording where), on reaching the goal, or at plan end.

    Args:
      real_play(action) -> levels_completed         (MUTATES real env)
      model_predict(action) -> predicted_frame       (advances the model)
      observe() -> observed_frame                     (current real frame)
      compare(a, b) -> bool (default ==)
      goal_completed: levels_completed value that counts as a win.

    Returns {'steps', 'levels_completed', 'mismatch_at': i|None, 'won': bool}.
    Pure: no engine/network/global state (all effects via injected closures).
    """
    if compare is None:
        compare = lambda a, b: a == b
    steps, completed, mismatch_at = 0, 0, None
    for i, action in enumerate(plan or []):
        try:
            predicted = model_predict(action)
        except Exception:
            predicted = None
        completed = int(real_play(action) or 0)
        steps += 1
        if completed >= goal_completed:
            return {"steps": steps, "levels_completed": completed,
                    "mismatch_at": None, "won": True}
        observed = observe()
        if predicted is not None and not compare(predicted, observed):
            mismatch_at = i
            break
    return {"steps": steps, "levels_completed": completed,
            "mismatch_at": mismatch_at, "won": completed >= goal_completed}
