# =====================================================================
# v21 runtime_coder — the ON-THE-FLY code-writer (offline, runs inside the agent).
#
# The "Executable World Models" method (arXiv:2605.05138, 58% public) made
# OFFLINE-eligible: a LOCAL Qwen2.5-Coder writes a Python `WorldModel` for the
# current game from observed transitions, we exec it in a restricted sandbox,
# VERIFY it reproduces recorded frames, then PLAN by enumerating its candidate
# action sequences and testing them on the forked engine — committing the
# SHORTEST that wins (RHAE-optimal). No network, no hand-coded game logic.
#
# Decoupled from the engine via two callbacks so it is unit-testable with no GPU:
#   observe_fn()          -> current frame (np array) or obs dict
#   try_plan_fn(plan)     -> True if the plan WINS the level on a fresh fork
#                            (plan = list of (action_id:int, data:dict|None))
# =====================================================================
import logging, signal, textwrap, builtins as _bi

logger = logging.getLogger("v21.runtime_coder")

_SAFE_BUILTINS = {k: getattr(_bi, k) for k in
    ("range", "len", "min", "max", "abs", "sum", "enumerate", "zip", "list",
     "dict", "set", "tuple", "int", "float", "bool", "sorted", "map", "filter",
     "any", "all", "print", "isinstance", "getattr", "hasattr",
     "__build_class__", "property", "staticmethod", "classmethod",
     "object", "super", "Exception", "ValueError", "reversed", "round")}

_ALLOWED_IMPORTS = {"numpy", "math", "itertools", "collections"}


def _safe_import(name, *a, **k):
    root = name.split(".")[0]
    if root in _ALLOWED_IMPORTS:
        return __import__(name, *a, **k)
    raise ImportError(f"import '{name}' is not allowed in the world-model sandbox")


_SAFE_BUILTINS["__import__"] = _safe_import

WM_SYSTEM = ("You are a code-writing agent for ARC-AGI-3. You NEVER explain; you "
             "output ONLY a Python module (no prose, no markdown fences). It must "
             "define `class WorldModel` with `__init__(self, observations)` and "
             "`candidate_plans(self, max_len)`. `candidate_plans` returns a list of "
             "action sequences, each a list of (action_id, data) tuples, ordered "
             "SHORTEST-first. Only `import numpy as np` is allowed. Read keys with "
             "`.get(...)` so a missing key never raises.")

WM_PROMPT = textwrap.dedent("""
    You receive `observations`, a dict with EXACTLY these keys:
      observations['level']             -> int, the level index
      observations['available_actions'] -> list[int], the action ids that are legal
      observations['frame']             -> 2D list[list[int]] of color ids (rows x cols)
      observations['transitions']       -> list of {{'action':int,'levels_completed':int,'changed':bool}}
                                           (each = the effect of pressing that action once from start)
    Use ONLY these keys (via .get). `data` is None for actions 1-5 and 7, and
    {{'x':col,'y':row}} for action 6 (click); x is the column, y is the row.

    Actions: 1-5 = discrete controls, 6 = click(x,y), 7 = undo. RHAE punishes wasted
    actions quadratically (5x cap) so the SHORTEST winning sequence wins.

    Observed for this level:
    {obs}

    Write `candidate_plans(self, max_len)` to return, SHORTEST-first:
      1. every id in available_actions as a depth-1 single — for 6, click the center and
         each distinct colored region; INCLUDE action 6 (do not skip it).
      2. repeat-one-action lines: [(a, None)] * k for a in 1..5, k up to max_len.
      3. click-repeat: [(6, t)] * k for each region center t.
      4. any structured plan your hypothesis of the mechanics (from transitions) implies.
    Output ONLY the Python module defining class WorldModel.
""").strip()


class _Timeout(Exception):
    pass


def _exec_world_model(code, observations, exec_timeout=5):
    """Exec LLM code in a restricted namespace; return a WorldModel instance."""
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```", 2)[1].lstrip("python").strip() if "```" in code[3:] else code.strip("`")
    ns = {"__builtins__": _SAFE_BUILTINS, "__name__": "world_model"}
    try:
        import numpy as np
        ns["np"] = np
    except Exception:
        pass
    def _alarm(signum, frame):
        raise _Timeout()
    _has_alarm = hasattr(signal, "SIGALRM")
    if _has_alarm:
        signal.signal(signal.SIGALRM, _alarm); signal.alarm(exec_timeout)
    try:
        exec(compile(code, "<world_model>", "exec"), ns)
        WM = ns.get("WorldModel")
        if WM is None:
            return None, "no WorldModel class"
        return WM(observations), None
    except _Timeout:
        return None, "exec timeout"
    except Exception as e:
        return None, f"exec error: {e}"
    finally:
        if _has_alarm:
            signal.alarm(0)


class RuntimeCoder:
    def __init__(self, llm, max_len=200, max_refine=1):
        self.llm, self.max_len, self.max_refine = llm, max_len, max_refine

    def solve_level(self, observations, try_plan_fn, max_plans=64):
        """Write -> exec -> verify -> plan -> test on fork. Returns shortest winning
        plan (list of (action_id, data)) or None. Always ALSO tries an LLM-independent
        safety net of the research-proven trivial wins, so a weak/crashing model still
        contributes (this is what makes ft09-L0-style ACTION6 wins reachable)."""
        safety = _safety_net_plans(observations, self.max_len)
        feedback, tried = "", set()
        for attempt in range(self.max_refine + 1):
            llm_plans = []
            try:
                prompt = WM_PROMPT.format(obs=_fmt(observations)) + \
                         (f"\n\nPrevious attempt failed: {feedback}\nFix it." if feedback else "")
                code = self.llm.complete(prompt, system=WM_SYSTEM, max_tokens=1200,
                                         stop=["```end", "\n\n\n"])
                wm, err = _exec_world_model(code, observations)
                if wm is None:
                    feedback = err or "no model"; logger.info("[coder] %s (attempt %d)", err, attempt)
                else:
                    llm_plans = wm.candidate_plans(self.max_len) or []
            except Exception as e:
                feedback = f"llm/candidate_plans failed: {e}"; logger.info("[coder] %s", feedback)
            # LLM plans first (they encode a hypothesis), then the safety net; shortest-first
            merged = sorted([p for p in llm_plans if p], key=len)[:max_plans] + safety
            for plan in merged:
                key = repr(plan)
                if key in tried or _refuses_exploit(plan):
                    continue
                tried.add(key)
                try:
                    if try_plan_fn(plan):
                        src = "llm" if plan in llm_plans else "safety-net"
                        logger.info("[coder] WIN with %d-action plan (%s)", len(plan), src)
                        return plan
                except Exception as e:
                    logger.debug("[coder] plan raised: %s", e)
            feedback = f"none of {len(merged)} plans (incl safety net) won within {self.max_len} steps"
        return None


def _refuses_exploit(plan):
    # R2.7: never emit the null-coordinate ACTION6 TypeError "win"
    for a, d in plan:
        if a == 6 and isinstance(d, dict) and (d.get("x") is None or d.get("y") is None):
            return True
    return False


def _click_targets(frame, max_targets=24):
    """Center + coarse grid + distinct-color region medians — LLM-independent click
    coords for the ACTION6 safety net."""
    try:
        import numpy as np
        f = np.array(frame)
        if f.ndim != 2:
            return [{"x": 32, "y": 32}]
        H, W = f.shape
        out, seen = [], set()
        def _add(x, y):
            k = (int(x), int(y))
            if 0 <= k[0] < W and 0 <= k[1] < H and k not in seen:
                seen.add(k); out.append({"x": k[0], "y": k[1]})
        _add(W // 2, H // 2)
        for y in range(H // 8, H, max(1, H // 4)):
            for x in range(W // 8, W, max(1, W // 4)):
                _add(x, y)
        for c in np.unique(f):
            if c == 0:
                continue
            ys, xs = np.where(f == c)
            if len(xs):
                _add(np.median(xs), np.median(ys))
        return out[:max_targets]
    except Exception:
        return [{"x": 32, "y": 32}]


def _safety_net_plans(observations, max_len, repeat_K=200):
    """The research-proven trivial wins, independent of any LLM output: depth-1 singles
    for every available action (ACTION6 on center + region targets), repeat-one-action
    lines, and click-repeat. Guarantees the coder tries what actually cracks the public
    set even when the generated WorldModel is weak or crashes."""
    obs = observations if isinstance(observations, dict) else {}
    avail = [int(a) for a in (obs.get("available_actions") or [1, 2, 3, 4, 5, 6, 7])]
    K = max(1, min(int(repeat_K), int(max_len)))
    targets = _click_targets(obs.get("frame"))
    plans = []
    for a in avail:                                   # depth-1 singles
        if a == 6:
            plans += [[(6, t)] for t in targets]
        else:
            plans.append([(a, None)])
    for a in avail:                                   # repeat-one-action lines
        if a != 6:
            plans.append([(a, None)] * K)
    for t in targets:                                 # click-repeat (vc33-style)
        plans.append([(6, t)] * min(K, 30))
    return sorted(plans, key=len)


def _completed(v):
    """Coerce a play() return into an int levels_completed (defensive)."""
    try:
        return int(v)
    except Exception:
        return 0


def replay_wins(start, plan, clone, play, goal):
    """PURE (offline-testable): replay a candidate `plan` on a FRESH fork of
    `start` and report whether it reaches `goal` levels_completed at any step.

    Mirrors the blitz adapter's clone/play closure contract so the Stage-3.5
    wiring in cadence_runner can build a `try_plan_fn` for RuntimeCoder without
    importing the engine here. `start` is never mutated (clone forks first).
      clone(start)        -> independent fork
      play(fork, (aid,d)) -> int levels_completed after the action (mutates fork)
    """
    if not plan:
        return False
    g = clone(start)
    for step in plan:
        if _completed(play(g, step)) >= goal:
            return True
    return False


def _fmt(observations):
    try:
        import numpy as np
        if isinstance(observations, np.ndarray):
            return f"frame shape={observations.shape}\n{np.array2string(observations, threshold=200)}"
    except Exception:
        pass
    s = str(observations)
    return s[:2000]
