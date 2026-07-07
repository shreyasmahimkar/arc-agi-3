# =====================================================================
# Chronos v21 — "brain" package (Cognitive layer, BACKLOG Epic B)
#
# A cognitively-inspired coordination layer stacked ON TOP of the existing
# v21 cascade (blitz -> BFS -> runtime_coder). It is NOT a neural brain; the
# name is a metaphor for the subsystems it wires together, each grounded in
# ARC-AGI-3 world-model research (see BRAIN_ARCHITECTURE.md):
#
#   perception   — object-centric scene graph from raw frames        [IMPLEMENTED]
#   world_model  — persistent, verified, refactorable executable WM   [interface]
#   hypotheses   — competing-hypothesis tracking (anti tunnel-vision) [interface]
#   planner      — plan-in-model -> execute-and-verify (MPC)          [interface]
#   memory       — cross-game concept/skill library (generalization)  [interface]
#   goal         — goal induction from observations                   [interface]
#
# Every module here is PURE / dependency-free at import (no arcengine, numpy,
# torch or network) so the offline self-test can import and exercise it. Engine
# adapters are lazy, mirroring blitz.py. Nothing in this package is on the
# default submission path yet: the cadence wires each subsystem behind an env
# flag only after a Mac cadence proves it, so the regression gate is never at
# risk. Build order and acceptance gates live in BACKLOG.md (Epic B).
# =====================================================================

__all__ = ["perception", "world_model", "hypotheses", "planner", "memory", "goal"]
