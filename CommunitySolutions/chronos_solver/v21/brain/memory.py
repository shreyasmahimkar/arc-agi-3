#!/usr/bin/env python3
# =====================================================================
# Chronos v21 brain — Concept / Skill Library (Epic B, phase B6)  [interface]
#
# The generalisation organ: a CROSS-GAME library of reusable abstractions so a
# regularity learned on one game seeds solving another (DreamCoder wake-sleep,
# LILO library learning, "Refactoring codebases through library design"). This
# is what turns the solver from a per-game search into something that
# accumulates transferable knowledge — the closest thing here to long-term
# memory. It stores three kinds of concept:
#   - action macros            (the existing v21_macro_bank.json)
#   - world-model code fragments (transition rules distilled from solved games)
#   - perceptual motifs         (object configurations that recur)
#
# Concepts are RETRIEVED by a perceptual key (below) so a new frame pulls the
# concepts seen in perceptually-similar situations, across games. The on-disk
# store + the wake-sleep refactor/compression pass live in later B6/B7 cycles;
# this ships the pure key + similarity retrieval core.
# =====================================================================


def perceptual_key(scene):
    """Compact, game-agnostic signature of a perception.scene(...) dict.

    Uses only structural features that transfer across games — grid shape,
    object count, and the multiset of (colour-agnostic) object sizes bucketed
    by magnitude — deliberately NOT absolute positions or a specific colour
    palette, so perceptually-similar scenes in DIFFERENT games map near each
    other. Returns a hashable tuple. Pure.
    """
    H, W = scene.get("dims", (0, 0))
    objs = scene.get("objects", []) or []
    def _bucket(n):
        b = 0
        while n > 1:
            n //= 2
            b += 1
        return b  # log2-ish size bucket
    size_sig = tuple(sorted(_bucket(o.get("size", 0)) for o in objs))
    return (H, W, len(objs), size_sig)


def key_similarity(k1, k2):
    """Similarity in [0,1] between two perceptual keys (1.0 == identical).
    Combines grid-shape match, object-count closeness, and size-signature
    overlap. Pure — used to rank library retrieval."""
    (h1, w1, n1, s1), (h2, w2, n2, s2) = k1, k2
    shape = 1.0 if (h1, w1) == (h2, w2) else 0.0
    count = 1.0 - (abs(n1 - n2) / max(1, n1 + n2))
    m1, m2 = list(s1), list(s2)
    inter = 0
    for v in set(m1):
        inter += min(m1.count(v), m2.count(v))
    denom = max(1, max(len(m1), len(m2)))
    sizes = inter / denom
    return round((shape + count + sizes) / 3.0, 6)


def retrieve(library, key, k=3, min_sim=0.0):
    """Top-k concepts from `library` (list of {'key', 'concept', ...}) most
    perceptually similar to `key`. Returns [(similarity, entry), ...] sorted
    high-to-low. Pure — the caller still verifies any plan a concept seeds."""
    scored = []
    for entry in (library or []):
        sim = key_similarity(key, entry.get("key"))
        if sim >= min_sim:
            scored.append((sim, entry))
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[:k]
