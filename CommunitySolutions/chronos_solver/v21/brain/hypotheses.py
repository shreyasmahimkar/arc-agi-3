#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Hypothesis Manager (Epic B, phase B4)  [interface]
#
# Fixes the #1 documented failure of executable-world-model agents on
# ARC-AGI-3 (Rodionov 2026, §4): "tunnel vision" — the agent forms ONE early
# hypothesis about objects/goal and elaborates it instead of considering
# alternatives, so a wrong ontology on level 1 contaminates the whole game.
#
# The remedy is to keep a SET of competing world-model hypotheses, and to
# spend scored environment actions on the move that best DISCRIMINATES them
# (active inference / optimal-experiment design) rather than greedily. When an
# observation arrives, hypotheses that mispredicted it are falsified/down-
# weighted. This module ships the two pure decision cores; wiring them to real
# WorldModel objects (world_model.py) is a later B4 cycle.
# =====================================================================


def falsify(hypotheses, action, observed, predict, compare=None):
    """Drop hypotheses whose prediction for `action` disagrees with `observed`.

    Args:
      hypotheses: list of hypothesis objects (opaque here).
      predict(hyp, action) -> predicted observation.
      compare(a, b) -> bool (default ==).

    Returns the surviving hypotheses (order preserved). If EVERY hypothesis is
    falsified, returns [] — the caller must then propose new hypotheses rather
    than trust a dead set. Pure.
    """
    if compare is None:
        compare = lambda a, b: a == b
    survivors = []
    for h in (hypotheses or []):
        try:
            pred = predict(h, action)
        except Exception:
            pred = None
        if pred is not None and compare(pred, observed):
            survivors.append(h)
    return survivors


def most_discriminating_action(hypotheses, actions, predict, compare=None):
    """Pick the action whose predicted outcomes DISAGREE most across hypotheses.

    Information-gain proxy: for each candidate action, group the hypotheses by
    their predicted observation; the action that splits them into the most
    distinct predicted outcomes is the most informative to actually execute
    (one real step can then falsify the largest share of the set). Ties break
    toward the earliest action for determinism.

    Returns the chosen action, or None if there are no actions/hypotheses.
    Pure: no engine/network/global state.
    """
    if compare is None:
        compare = lambda a, b: a == b
    hyps = list(hypotheses or [])
    acts = list(actions or [])
    if not acts or len(hyps) < 2:
        return acts[0] if acts else None
    best_action, best_groups = None, -1
    for a in acts:
        buckets = []  # list of representative predictions
        for h in hyps:
            try:
                pred = predict(h, a)
            except Exception:
                pred = None
            placed = False
            for rep in buckets:
                if compare(rep, pred):
                    placed = True
                    break
            if not placed:
                buckets.append(pred)
        if len(buckets) > best_groups:
            best_groups, best_action = len(buckets), a
    return best_action
