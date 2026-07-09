#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Perception-first coder digest (BACKLOG R6 + R8)
#
# Why: the runtime code-writer (runtime_coder.py) today feeds the coder a
# RAW SERIALIZED GRID (`_fmt` -> np.array2string). Two findings say that is
# the wrong representation for a small local model:
#   R6 (Symbolica *Arcgentica*): compress the exploration transcript into a
#       fixed-size structured digest so a long/deep wall never blows the
#       coder's context window.
#   R8 (arXiv:2512.21329, "Perception Bottleneck"): ~80% of VLM/LLM ARC
#       failures are PERCEPTION, not reasoning; a dedicated
#       perception -> natural-language stage before the reasoner adds
#       +11-13pp. Serialized raw grids are the hard format even for humans.
#
# So this module turns `observations` into a compact, DETERMINISTIC,
# bounded-length symbolic scene description built from brain.perception's
# connected-component objects + a tried-action -> outcome table. It replaces
# only the `{obs}` block of WM_PROMPT; the coder still writes code against the
# real `observations` dict (keys/indexing unchanged), so the runtime contract
# is untouched. Pure: imports only brain.perception (dep-free). Env-gated at
# the call site (V21_CODER_DIGEST, default OFF) — fully additive.
# =====================================================================

from brain import perception as _P

# Bounds keep the digest small even on a 64x64 many-object frame so the coder
# prompt never grows unbounded with exploration depth (R6).
_MAX_OBJECTS = 12
_MAX_CLICKS = 12
_MAX_TRANSITIONS = 24
_MAX_CHARS = 2000


def _frame_of(observations):
    """Extract the frame from a dict obs or treat obs itself as a frame."""
    if isinstance(observations, dict):
        return observations.get("frame")
    return observations


def digest(observations, max_objects=_MAX_OBJECTS, max_chars=_MAX_CHARS):
    """Perception-first structured scene digest for the coder prompt.

    Deterministic and length-bounded. Names every reported component by an
    id so the coder can reference objects by identity, and preserves a
    lossless action -> (changed, levels_completed) recall table from
    observations['transitions']. Returns a plain string. Never raises: any
    parse problem degrades to a minimal factual line (the caller also guards
    with a try/except and falls back to the raw formatter).
    """
    lines = []
    level = available = transitions = None
    if isinstance(observations, dict):
        level = observations.get("level")
        available = observations.get("available_actions")
        transitions = observations.get("transitions")

    if level is not None or available is not None:
        lines.append("level=%s available_actions=%s"
                     % (level, list(available) if available is not None else "?"))

    frame = _frame_of(observations)
    try:
        sc = _P.scene(frame)
    except Exception:
        sc = None
    if sc:
        H, W = sc["dims"]
        lines.append("scene: dims=%dx%d background=%s n_objects=%d"
                     % (H, W, sc["background"], sc["n_objects"]))
        objs = sc["objects"]
        # largest objects first: the salient, clickable/steerable pieces
        objs = sorted(objs, key=lambda o: (-o["size"], o["bbox"][0], o["bbox"][1]))
        shown = objs[:max_objects]
        if shown:
            lines.append("objects (largest first, %d of %d):" % (len(shown), len(objs)))
            for i, o in enumerate(shown):
                t, l, b, r = o["bbox"]
                cr, cc = o["centroid"]
                lines.append("  #%d color=%d size=%d bbox=(%d,%d,%d,%d) centroid=(row=%d,col=%d)"
                             % (i, o["color"], o["size"], t, l, b, r, cr, cc))
        try:
            cts = _P.click_targets(frame, limit=_MAX_CLICKS)
        except Exception:
            cts = []
        if cts:
            lines.append("click_targets (x=col,y=row): "
                         + ", ".join("(%d,%d)" % (c["x"], c["y"]) for c in cts))
    else:
        lines.append("scene: <no parseable frame>")

    if transitions:
        lines.append("action->outcome (each pressed once from start):")
        for tr in list(transitions)[:_MAX_TRANSITIONS]:
            if not isinstance(tr, dict):
                continue
            lines.append("  a%s -> changed=%s levels_completed=%s"
                         % (tr.get("action"), bool(tr.get("changed")),
                            tr.get("levels_completed")))

    out = "\n".join(lines)
    if len(out) > max_chars:
        out = out[:max_chars - 3] + "..."
    return out


def plan_failure_scene(start_frame, final_frame, max_objects=6, max_chars=600):
    """Perception-first description of where a FAILED plan ENDED (R6/R8 + R13).

    The Opus teacher's iterative retry (cadence_runner._opus_teacher_for_solver ->
    solve_wall_iterative) currently only feeds Opus a level-count report
    ("reached levels_completed=5 of goal 6"). Two findings say that's the wrong
    signal to hand a reasoner: R8 (perception is the real bottleneck) and R6
    (compress the transcript into a bounded structured digest). This turns the
    stuck END frame — and how it differs from the level start — into a compact
    symbolic note so the next round reasons over OBJECTS and DELTAS, not just a
    number. Bounded + deterministic + pure (imports only brain.perception).
    Never raises: any parse problem returns "" so the teach loop is unbroken.
    """
    try:
        sc = _P.scene(final_frame)
    except Exception:
        return ""
    if not sc:
        return ""
    H, W = sc["dims"]
    parts = ["final-frame scene: dims=%dx%d background=%s n_objects=%d"
             % (H, W, sc["background"], sc["n_objects"])]
    objs = sorted(sc["objects"], key=lambda o: (-o["size"], o["bbox"][0], o["bbox"][1]))
    shown = objs[:max_objects]
    if shown:
        parts.append("largest objects: " + "; ".join(
            "color=%d size=%d centroid=(row=%d,col=%d)"
            % (o["color"], o["size"], o["centroid"][0], o["centroid"][1])
            for o in shown))
    try:
        d = _P.diff(start_frame, final_frame)
        parts.append("delta vs level start: %d cells changed (%d appeared, %d disappeared, %d recolored)"
                     % (d["n_changed"], len(d["appeared"]),
                        len(d["disappeared"]), len(d["recolored"])))
    except Exception:
        pass
    out = "; ".join(parts)
    if len(out) > max_chars:
        out = out[:max_chars - 3] + "..."
    return out
