#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Goal Induction (Epic B, phase B5)  [interface]
#
# ARC-AGI-3 gives NO stated goal — "goal acquisition" is one of the five named
# capabilities the benchmark tests (ARC-AGI-3 technical report, arXiv:2603.24621).
# The agent must INFER what winning means from observation: the score/level
# signal that rises, or a target configuration the environment rewards. The
# induced goal feeds world_model.goal_reached and planner.plan_in_model.
#
# This ships pure inducers over the score/level signal (the always-available,
# game-general reward channel); frame-configuration goal induction (e.g. "make
# the board match a target motif") is a later B5 cycle that will use
# perception.scene motifs + memory retrieval.
# =====================================================================


def induce_from_scores(scores):
    """Infer a goal descriptor from a sequence of observed level/score values.

    Args:
      scores: list of ints (levels_completed or score after each step so far).

    Returns a descriptor dict:
      {'kind': 'maximize_progress'|'unknown',
       'monotone': bool,          # progress only ever went up
       'last': int, 'best': int}
    'maximize_progress' is asserted once we have seen the signal increase at
    least once (evidence that some action advances the goal). Pure.
    """
    s = [int(x) for x in (scores or [])]
    if not s:
        return {"kind": "unknown", "monotone": True, "last": 0, "best": 0}
    increased = any(s[i] > s[i - 1] for i in range(1, len(s)))
    monotone = all(s[i] >= s[i - 1] for i in range(1, len(s)))
    return {
        "kind": "maximize_progress" if increased else "unknown",
        "monotone": monotone,
        "last": s[-1],
        "best": max(s),
    }


def goal_reached_by_progress(prev_completed, new_completed):
    """Level-completion goal test: the induced default goal is 'advance the
    level counter'. True iff `new_completed` exceeds `prev_completed`. Pure."""
    return int(new_completed) > int(prev_completed)
