# =====================================================================
# v21 brain/teacher.py — the OPUS TEACHER (Epic R13, the API-key path).
#
# When the local cascade (blitz→BFS→Go-Explore→Qwen coder) fails a wall, this
# calls a STRONG cloud model (Claude Opus) with the game's WHITE-BOX source code
# + the stuck state, and asks it to reason out the exact winning action sequence.
# The returned plan is UNVERIFIED — the caller still runs `verify_solution` +
# the shortest-plan gate + exploit-refusal, so a hallucinated plan is caught and
# discarded. Opus only PROPOSES; the engine is the judge.
#
# Because ls20/ft09/vc33 ship their source, Opus can literally READ the mechanics
# and construct the plan — the reason this is the strongest single lever for the
# resistant walls (ls20 L5 etc.).
#
# Key handling: read ANTHROPIC_API_KEY from the ENVIRONMENT only (never a file in
# the repo, never logged). Put it in an untracked ~/.chronos_secrets sourced by
# run_cadence.sh. Network via stdlib urllib (cadence box runs with --allow-network).
# Env: V21_OPUS_TEACHER=1 to enable; V21_OPUS_MODEL (default claude-opus-4-8).
# =====================================================================
import os, json, logging, re

logger = logging.getLogger("v21.teacher")
_API = "https://api.anthropic.com/v1/messages"
_MODEL = os.environ.get("V21_OPUS_MODEL", "claude-opus-4-8")

SYSTEM = ("You are an expert ARC-AGI-3 solver with FULL access to a game's Python "
          "source. Read the source, work out the exact mechanics of the target level, "
          "and output the SHORTEST action sequence that completes it. Actions: 1-5 are "
          "discrete controls, 6 is a click needing {\"x\":col,\"y\":row}, 7 is undo. "
          "Output ONLY a JSON object: {\"plan\": [[action_id, data_or_null], ...]}. "
          "No prose. data is null for actions 1-5 and 7.")

PROMPT = ("Game source ({gid}.py):\n```python\n{src}\n```\n\n"
          "TARGET: complete LEVEL {level} (i.e. reach levels_completed >= {goal}). The "
          "levels are sequential; assume levels 0..{prev} are already solved and you START "
          "at the level {level} entry state. Available discrete actions: {avail}. "
          "{state}"
          "Notes from failed local attempts: {notes}\n\n"
          "Reason through the source privately, then output ONLY the JSON plan (shortest).")


ARCH_SYSTEM = (
    "You are an expert ML engineer improving a TINY frame-change/action predictor for "
    "ARC-AGI-3 (a StochasticGoose/TRM-style net trained on a Mac MPS GPU). Output ONLY a "
    "Python module (no prose, no markdown fences) defining "
    "`def build_net(n_colors, n_actions, grid):` that returns a torch.nn.Module. The net "
    "takes a LongTensor (B, grid, grid) of colour ids and returns a tuple "
    "(change_logits, win_logits), each shape (B, n_actions). Keep it SMALL (<2M params), "
    "MPS-safe (only standard nn: Embedding, Conv2d, Linear, ReLU/GELU, LayerNorm/BatchNorm, "
    "pooling, residual adds — NO custom CUDA, NO external deps). `import torch` and "
    "`import torch.nn as nn` are allowed.")

ARCH_PROMPT = (
    "Current architecture:\n```python\n{cur}\n```\n\n"
    "Held-out validation accuracy of the current net: {val:.3f} over {n} samples. "
    "Propose an IMPROVED build_net that should RAISE held-out change/win accuracy — e.g. "
    "residual conv blocks, a small self-attention over grid cells, better motion inductive "
    "bias, light regularisation. Keep it tiny and MPS-safe. Output ONLY the Python module "
    "defining build_net.")


def _key():
    return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("V21_OPUS_KEY")


def _is_transient(e):
    """True for network errors worth RETRYING (R13 robustness). The Mac cadence
    box has intermittent DNS on the launchd network path — run 213152Z lost the
    Opus teacher on ft09 to a bare `urlopen error [Errno 8] nodename nor servname
    provided` at 18:09/18:27 EDT even though the SAME endpoint answered for ls20 at
    17:48. A single transient blip must not silently kill the teacher on a wall it
    could crack. Retry DNS/connection/timeout + HTTP 429/5xx; do NOT retry 4xx
    (bad key / bad request) — those won't fix themselves."""
    import urllib.error, socket
    if isinstance(e, urllib.error.HTTPError):
        return e.code == 429 or 500 <= e.code < 600
    if isinstance(e, urllib.error.URLError):
        return True            # DNS ('nodename nor servname'), refused, unreachable
    if isinstance(e, (socket.timeout, TimeoutError, ConnectionError, OSError)):
        return True
    return False


def _with_retries(fn, tries, base_backoff, sleep=None):
    """Call fn(); on a TRANSIENT error retry up to `tries` total attempts with
    exponential backoff (capped 8s). Non-transient errors raise immediately. Pure
    control-flow (network lives in fn), so it is fully offline-testable with a fake
    fn + a no-op sleep. Returns fn()'s value or re-raises the last error."""
    import time
    if sleep is None:
        sleep = time.sleep
    tries = max(1, int(tries))
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:                      # noqa: BLE001 — classify below
            last = e
            if i + 1 >= tries or not _is_transient(e):
                raise
            back = min(float(base_backoff) * (2 ** i), 8.0)
            logger.info("[opus] transient network error (attempt %d/%d): %s%s",
                        i + 1, tries, e,
                        (" — retrying in %.1fs" % back) if back > 0 else " — retrying")
            if back > 0:
                sleep(back)
    if last is not None:                            # pragma: no cover — loop always raises/returns
        raise last


class OpusTeacher:
    def __init__(self, model=_MODEL, max_tokens=4096):
        self.model, self.max_tokens = model, max_tokens

    def available(self):
        return bool(_key())

    def _call(self, system, user):
        import urllib.request
        body = {"model": self.model, "max_tokens": self.max_tokens,
                "system": system, "messages": [{"role": "user", "content": user}]}
        payload = json.dumps(body).encode()
        deadline = int(os.environ.get("V21_OPUS_DEADLINE", "180"))

        def _once():
            req = urllib.request.Request(
                _API, data=payload,
                headers={"content-type": "application/json",
                         "x-api-key": _key(),
                         "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=deadline) as r:
                data = json.loads(r.read())
            # messages API returns {"content":[{"type":"text","text":...}], ...}
            parts = data.get("content", [])
            return "".join(p.get("text", "") for p in parts if p.get("type") == "text")

        # R13 robustness: recover from intermittent DNS/network on the cadence box.
        tries = int(os.environ.get("V21_OPUS_RETRIES", "3"))
        backoff = float(os.environ.get("V21_OPUS_RETRY_BACKOFF", "1.5"))
        return _with_retries(_once, tries, backoff)

    def solve_wall(self, gid, source_code, level_idx, avail, notes="", state=""):
        """Ask Opus for a candidate plan. Returns list[(action_id, data)] or None.
        UNVERIFIED — caller must verify_solution + shortest-gate + refuse exploits.

        `state` (R14 grounding): a symbolic digest of the REAL level-start frame +
        a per-action effect table (each action pressed once -> changed / levels_
        completed), captured from the live engine by the caller. Without it Opus
        reads the source (rules) but never sees the board it's playing on — it
        plans blind (this run: ls20 L5 plans changed 86-90 cells but never crossed
        the goal; vc33 L4 round-1 first action was a no-op). Empty -> unchanged
        (identical to the old prompt), so the upgrade is fully additive."""
        if not self.available():
            return None
        src = source_code or ""
        if len(src) > 60000:                    # keep the request bounded
            src = src[:60000] + "\n# ...truncated..."
        state_block = ""
        if state and str(state).strip():
            state_block = ("CURRENT OBSERVED STATE (ground truth from the live engine at "
                           "the level entry — TRUST THIS over your mental simulation of the "
                           "source; the action->outcome table shows what each action ACTUALLY "
                           "does from here):\n" + str(state).strip()[:2400] + "\n\n")
        user = PROMPT.format(gid=gid, src=src, level=level_idx, goal=level_idx + 1,
                             prev=max(level_idx - 1, 0), avail=list(avail),
                             state=state_block, notes=(notes or "none")[:1500])
        try:
            raw = self._call(SYSTEM, user)
        except Exception as e:
            logger.warning("[%s] opus teacher call failed: %s", gid, e)
            return None
        return parse_plan(raw)

    def write_toddler_arch(self, current_code, val_acc, n_samples):
        """OPUS-AS-ML-ENGINEER: write an IMPROVED PyTorch architecture for the neural
        toddler (the frame-change/win predictor). Returns a Python module defining
        `build_net(n_colors, n_actions, grid) -> nn.Module` (UNVERIFIED — the caller
        trains it and only ADOPTS it if it beats the current net on held-out accuracy).
        This is Opus designing the tiny net, not solving a level."""
        if not self.available():
            return None
        user = ARCH_PROMPT.format(cur=(current_code or "")[:8000],
                                  val=float(val_acc or 0.0), n=int(n_samples or 0))
        try:
            raw = self._call(ARCH_SYSTEM, user)
        except Exception as e:
            logger.warning("opus arch call failed: %s", e)
            return None
        return _strip_module(raw)

    def solve_wall_iterative(self, gid, source_code, level_idx, avail,
                             try_plan, max_rounds=None, notes="", state=""):
        """R7 teach-with-feedback: propose a plan, let the caller EXECUTE+VERIFY it
        via try_plan(plan) -> (solved: bool, feedback: str); on failure, fold the
        engine's failure report back into the next prompt as a negative-constraint
        counterexample and re-ask. Returns the first VERIFIED plan or None.

        The single-shot teacher (this run: ls20 L5 got a 19-action plan that failed
        verification and was discarded, learning nothing) throws away the strongest
        signal it has — how far the plan actually got. This loop turns that failure
        into a textual gradient (DREAMTEAM/R7). Pure control-flow: the network lives
        only in solve_wall, so this is fully offline-testable with a mock try_plan +
        a mocked _call. try_plan MUST be side-effect-safe (runs on a fresh fork)."""
        if not self.available():
            return None
        if max_rounds is None:
            try:
                max_rounds = int(os.environ.get("V21_OPUS_ROUNDS", "2"))
            except Exception:
                max_rounds = 2
        acc_notes = notes or ""
        for rnd in range(max(1, max_rounds)):
            # `state` (the live-engine ground truth) is CONSTANT across rounds — each
            # round re-roots to the same level-start — while `acc_notes` grows with the
            # per-round failure gradient. So Opus always sees the real board AND the
            # accumulating "what didn't work" feedback.
            plan = self.solve_wall(gid, source_code, level_idx, avail,
                                   notes=acc_notes, state=state)
            if not plan:
                return None
            try:
                solved, feedback = try_plan(plan)
            except Exception as e:
                solved, feedback = False, "executor error: %s" % e
            if solved:
                return plan
            acc_notes = _augment_notes(notes, rnd, plan, feedback)
        return None


def _augment_notes(base, rnd, plan, feedback):
    """Fold a failed attempt into the next prompt as a bounded negative constraint
    (R7 textual-feedback gradient). Pure; length-capped so the prompt stays bounded."""
    fb = (str(feedback) if feedback else "no engine feedback").strip()[:600]
    head = (base or "").strip()
    line = ("Attempt %d FAILED (%d-action plan): %s. Do NOT repeat that exact "
            "sequence; reconsider the mechanics and try a materially different plan."
            % (rnd + 1, len(plan or []), fb))
    return ((head + "\n" + line).strip())[:1500]


WM_SYSTEM = (
    "You are an expert ARC-AGI-3 engine analyst WITH the game's Python source. "
    "Output ONLY a Python module (no prose, no markdown fences) defining "
    "`class WorldModel` with `__init__(self, observations)` and "
    "`candidate_plans(self, max_len)` that returns a list of action sequences (each "
    "a list of [action_id, data]) ordered SHORTEST-first. ENCODE the ACTUAL dynamics "
    "you read in the source: represent the player/objects, the transition for each "
    "action, and the win condition for the target level; then have candidate_plans() "
    "return the plan(s) your model predicts will complete the level. Optionally add "
    "`predict(self, state, action)`. Only `import numpy as np`, `math`, `itertools`, "
    "`collections` are allowed. Read observation keys with `.get(...)`.")

WM_PROMPT = (
    "Game source ({gid}.py):\n```python\n{src}\n```\n\n"
    "`observations` is a dict with keys: level (int), available_actions (list[int]), "
    "frame (2D int grid). data is null for actions 1-5 and 7, and {{\"x\":col,\"y\":row}} "
    "for action 6.\n\nWrite a WorldModel that, from ITS understanding of this source, "
    "proposes the SHORTEST action sequence(s) to complete LEVEL {level} (levels 0..{prev} "
    "already solved; you start at the level {level} entry state). Available discrete "
    "actions: {avail}. Output ONLY the Python module (class WorldModel).")


class _OpusWM:
    pass  # marker for tooling; the real work is OpusTeacher.write_world_model


def _wm_methods():
    """Attach write_world_model to OpusTeacher (kept here to group WM prompts)."""
    def write_world_model(self, gid, source_code, level_idx, avail):
        """Ask Opus to WRITE an executable WorldModel .py from the white-box source.
        Returns the Python module text (UNVERIFIED — caller execs it in the sandbox,
        plans, and verifies on the engine) or None."""
        if not self.available():
            return None
        src = (source_code or "")
        if len(src) > 60000:
            src = src[:60000] + "\n# ...truncated..."
        user = WM_PROMPT.format(gid=gid, src=src, level=level_idx,
                                prev=max(level_idx - 1, 0), avail=list(avail))
        try:
            raw = self._call(WM_SYSTEM, user)
        except Exception as e:
            logger.warning("[%s] opus world-model call failed: %s", gid, e)
            return None
        return _strip_module(raw)
    OpusTeacher.write_world_model = write_world_model


def _strip_module(raw):
    if not raw:
        return None
    txt = raw.strip()
    if txt.startswith("```"):
        m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
        if m:
            txt = m.group(1).strip()
    return txt or None


def parse_plan(raw):
    """Extract [[action_id, data], ...] from the model's JSON reply. Tolerant of
    code fences / surrounding text. Returns list[(int, dict|None)] or None."""
    if not raw:
        return None
    txt = raw.strip()
    if "```" in txt:
        m = re.search(r"```(?:json)?\s*(.*?)```", txt, re.S)
        if m:
            txt = m.group(1).strip()
    obj = None
    try:
        obj = json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)      # salvage the first {...}
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                return None
    if not isinstance(obj, dict):
        return None
    steps = obj.get("plan")
    if not isinstance(steps, list) or not steps:
        return None
    out = []
    for s in steps:
        if isinstance(s, (list, tuple)) and s:
            a = int(s[0])
            d = s[1] if len(s) > 1 and isinstance(s[1], dict) else None
            out.append((a, d))
        elif isinstance(s, int):
            out.append((s, None))
    return out or None


_wm_methods()  # attach OpusTeacher.write_world_model at import
