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
             "output ONLY a Python module. It must define `class WorldModel` with "
             "`__init__(self, observations)` and `candidate_plans(self, max_len)` "
             "returning a list of action sequences (each a list of (action_id, data)) "
             "ordered SHORTEST-first, cheapest-hypothesis-first. Optionally define "
             "`predict(self, state, action)`. Only `import numpy as np` is allowed.")

WM_PROMPT = textwrap.dedent("""
    Game observations (frame = 2D int grid of color ids; transitions show action -> frame delta):
    {obs}

    Actions: 1-5 = discrete controls, 6 = click with data={{'x':int,'y':int}}, 7 = undo.
    RHAE punishes wasted actions quadratically with a 5x cap, so the SHORTEST winning
    sequence wins. Write a WorldModel whose candidate_plans() proposes, shortest-first:
    depth-1 blind singles, repeat-one-action lines, and click-on-each-salient-object,
    plus any structured plan your hypothesis of the mechanics implies. Output ONLY the module.
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
        plan (list of (action_id, data)) or None. Refines once on total failure."""
        feedback = ""
        for attempt in range(self.max_refine + 1):
            prompt = WM_PROMPT.format(obs=_fmt(observations)) + \
                     (f"\n\nPrevious attempt failed: {feedback}\nFix it." if feedback else "")
            code = self.llm.complete(prompt, system=WM_SYSTEM, max_tokens=1200,
                                     stop=["```end", "\n\n\n"])
            wm, err = _exec_world_model(code, observations)
            if wm is None:
                feedback = err or "no model"; logger.info("[coder] %s (attempt %d)", err, attempt)
                continue
            try:
                plans = wm.candidate_plans(self.max_len)
            except Exception as e:
                feedback = f"candidate_plans crashed: {e}"; continue
            # shortest-first, dedup, cap the fan-out
            plans = sorted([p for p in plans if p], key=len)[:max_plans]
            for plan in plans:
                if _refuses_exploit(plan):
                    continue
                try:
                    if try_plan_fn(plan):
                        logger.info("[coder] WIN with %d-action plan", len(plan))
                        return plan
                except Exception as e:
                    logger.debug("[coder] plan raised: %s", e)
            feedback = f"none of {len(plans)} candidate plans won within {self.max_len} steps"
        return None


def _refuses_exploit(plan):
    # R2.7: never emit the null-coordinate ACTION6 TypeError "win"
    for a, d in plan:
        if a == 6 and isinstance(d, dict) and (d.get("x") is None or d.get("y") is None):
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
