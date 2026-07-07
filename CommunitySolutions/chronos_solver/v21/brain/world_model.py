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
