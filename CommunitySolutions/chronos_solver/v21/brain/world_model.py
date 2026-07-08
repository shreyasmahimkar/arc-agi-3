#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Executable World Model (Epic B, phase B2)  [interface]
#
# Generalises the one-shot `runtime_coder` into a PERSISTENT, VERIFIED,
# self-refactoring executable world model per game, following Rodionov 2026
# (Executable World Models for ARC-AGI-3), WorldCoder and DreamCoder. A world
# model is an executable Python object exposing three functions the coding
# agent fills in and the loop maintains across levels:
#
#   parse(frame)                -> state      (object-centric, via perception)
#   transition(state, action)   -> state'     (predicted dynamics)
#   goal_reached(state)         -> bool       (induced goal, see goal.py)
#
# The model is only trusted to the extent it REPRODUCES recorded observations
# (verifier below). The MDL/simplicity bias is applied by the loop as a
# refactor pass (replace special cases with shared rules) between runs.
#
# This module ships the PURE verifier core now; the executable-model authoring
# + on-disk persistence (v21/brain/wm/<game>/) land in later B2 cycles.
# =====================================================================

# Template the coding agent fills per game (kept as text so it is dependency-free).
MODEL_TEMPLATE = '''
class WorldModel:
    """Executable hypothesis about ONE ARC-AGI-3 game's dynamics."""
    def parse(self, frame):
        """Frame (2-D int grid) -> internal state (use brain.perception)."""
        raise NotImplementedError
    def transition(self, state, action):
        """(state, (action_id, data)) -> predicted next state."""
        raise NotImplementedError
    def render(self, state):
        """state -> predicted frame (2-D int grid) for verifier comparison."""
        raise NotImplementedError
    def goal_reached(self, state):
        """state -> True if the level goal is satisfied."""
        raise NotImplementedError
'''


def verify_model(predict_fn, records, compare=None):
    """Score an executable world model against recorded transitions.

    This is the verifier at the heart of the executable-world-model loop: a
    model is only trusted insofar as it reproduces what was actually observed.

    Args:
      predict_fn(prev, action) -> predicted_next   (the model under test)
      records: iterable of (prev, action, observed_next) triples.
      compare(a, b) -> bool: equality test for observations (default ==).

    Returns {'n_ok', 'n_total', 'accuracy', 'mismatches': [index, ...]}.
    Pure: no engine/network/global state (injected predict_fn).
    """
    if compare is None:
        compare = lambda a, b: a == b
    n_ok, n_total, mism = 0, 0, []
    for i, rec in enumerate(records or []):
        prev, action, observed = rec
        n_total += 1
        try:
            pred = predict_fn(prev, action)
        except Exception:
            pred = None
        if pred is not None and compare(pred, observed):
            n_ok += 1
        else:
            mism.append(i)
    acc = (n_ok / n_total) if n_total else 0.0
    return {"n_ok": n_ok, "n_total": n_total, "accuracy": acc, "mismatches": mism}


def is_trusted(report, threshold=1.0):
    """A model is trusted for planning only if it reproduces the record set
    at (or above) `threshold` accuracy. Default 1.0 == must reproduce ALL."""
    return report.get("n_total", 0) > 0 and report.get("accuracy", 0.0) >= threshold


# =====================================================================
# Epic C2 (=T2) — PERSISTENT executable world model substrate.
#
# The verifier above says whether a model is trusted; this section provides the
# smallest trustworthy model (a table of observed transitions), an MDL refactor
# toward a shorter rule, and on-disk persistence at brain/wm/<game>/ so a model
# learned this run is reused next run. The tabular model reproduces every
# recorded transition BY CONSTRUCTION, so is_trusted(verify_model(...)) is True
# out of the box; the MDL pass then trades the table for a shorter equivalent
# rule when one exists (better generalisation, Rodionov 2026 / DreamCoder).
#
# All pure + dependency-free (json/os only). Wiring into cadence_runner's live
# solve loop (record transitions -> build -> save -> seed next run) is env-gated
# V21_WORLD_MODEL (default OFF) and lands once this substrate is proven offline.
# =====================================================================
import json as _json
import os as _os

MODEL_FILENAME = "model.json"


def _canon(x):
    """Recursively convert x into a canonical, deterministic, JSON-serialisable
    form (tuples -> lists, dict keys -> str + sorted, array-likes via .tolist()).
    The same logical state always yields the same canonical form, so keys and
    equality comparisons are stable across build / persist / reload."""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    tolist = getattr(x, "tolist", None)
    if callable(tolist):
        try:
            return _canon(tolist())
        except Exception:
            pass
    return x


def _key(prev, action):
    """Deterministic string key for a (state, action) pair (JSON, sorted)."""
    return _json.dumps([_canon(prev), _canon(action)], sort_keys=True, separators=(",", ":"))


def build_tabular_model(records):
    """Build a trusted-by-construction executable world model from recorded
    (prev, action, next) transitions: a table mapping each observed
    (state, action) -> observed next state. This is the SUBSTRATE the coder /
    MDL pass later compresses; it reproduces every record by definition.

    Returns a JSON-serialisable dict: {"kind": "tabular", "table": {..}, "n": N}.
    Pure: no engine/network/global state.
    """
    table = {}
    n = 0
    for rec in records or []:
        prev, action, nxt = rec
        table[_key(prev, action)] = _canon(nxt)
        n += 1
    return {"kind": "tabular", "table": table, "n": n}


def mdl_refactor(model):
    """MDL/simplicity pass: replace a verbose transition table with the SHORTEST
    rule that still reproduces every recorded transition. Recognises two
    compressible regimes, otherwise returns the model unchanged:
      - identity : every next state equals its prev state.
      - constant : every transition maps to the SAME next state.
    predict_from_model answers identically on all recorded pairs after refactor,
    but the model is smaller (a better generalisation proxy). Pure.
    """
    if model.get("kind") != "tabular":
        return model
    table = model.get("table", {})
    if not table:
        return model
    items = []
    for k, nxt in table.items():
        prev, action = _json.loads(k)
        items.append((prev, action, nxt))
    if all(nxt == prev for prev, _a, nxt in items):
        return {"kind": "identity", "n": model.get("n", len(items))}
    first = items[0][2]
    if all(nxt == first for _p, _a, nxt in items):
        return {"kind": "constant", "value": first, "n": model.get("n", len(items))}
    return model


def predict_from_model(model, prev, action):
    """Pure predictor over any model from build_tabular_model / mdl_refactor.
    Returns the predicted next state, or None if the model has no applicable
    rule (an unseen (state, action) under a tabular model)."""
    kind = model.get("kind")
    if kind == "identity":
        return _canon(prev)
    if kind == "constant":
        return model.get("value")
    if kind == "tabular":
        return model.get("table", {}).get(_key(prev, action))
    return None


def wm_dir(base_dir, game):
    """Directory for one game's persisted model: <base>/brain/wm/<game>/."""
    return _os.path.join(base_dir, "brain", "wm", str(game))


def save_model(game_dir, model):
    """Persist a model dict to <game_dir>/model.json (atomic). Returns the path."""
    _os.makedirs(game_dir, exist_ok=True)
    path = _os.path.join(game_dir, MODEL_FILENAME)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        _json.dump(model, f, sort_keys=True, separators=(",", ":"))
    _os.replace(tmp, path)
    return path


def load_model(game_dir):
    """Load a persisted model dict, or None if absent/unreadable."""
    path = _os.path.join(game_dir, MODEL_FILENAME)
    try:
        with open(path) as f:
            return _json.load(f)
    except Exception:
        return None
