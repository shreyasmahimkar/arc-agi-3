# =====================================================================
# v21 brain/blackboard.py — the shared SCRATCHPAD (Epic C, teacher/student).
#
# The metaphor: every solver is a TEACHER sitting around a table, and a TODDLER
# (the learned intuitive prior) watches them all. They don't call each other —
# they leave LESSONS on a shared scratchpad (this blackboard), and read each
# other's lessons before acting. One shared "understanding of the world" per game.
#
#   Teachers WRITE lessons:
#     • search teachers (BFS / blitz / macro-planner / Go-Explore):
#         action_effects (a -> did it change/win), verified fragments (plans that
#         reached a subgoal — Go-Explore seeds), dead_ends (counterexamples),
#         novelty cells (downsampled frame -> shortest path seen).
#     • model teachers (runtime_coder / world_model): world_facts (learned dynamics).
#   Students READ hints:  hints(level) -> ranked actions + seed plans + avoid-list
#         + click targets.  The TODDLER (intuition prior) is DISTILLED from all of
#         this each cycle (consolidation / wake-sleep).
#
# PURE + offline: json + stdlib only (numpy optional for cell hashing). Persisted
# per game at brain/blackboard/<gid>.json so lessons compound across the 4h loop.
# Additive + env-gated (V21_BLACKBOARD); the proven cascade never depends on it.
# =====================================================================
import os, json, time, hashlib
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_BB_DIR = os.environ.get("V21_BLACKBOARD_DIR", os.path.join(_HERE, "blackboard"))
ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]


def _path(game):
    return os.path.join(_BB_DIR, f"{str(game).split('-')[0]}.json")


class Blackboard:
    """One shared, persistent scratchpad per game. Teachers `.teach(...)`; students
    `.hints(...)`. All entries carry provenance (`source`) so we can weight/trust
    teachers and audit what taught the toddler."""

    def __init__(self, game):
        self.game = str(game).split("-")[0]
        self.data = {"game": self.game, "action_effects": {}, "fragments": [],
                     "dead_ends": [], "cells": {}, "world_facts": [], "updated": 0}
        p = _path(self.game)
        if os.path.exists(p):
            try:
                self.data.update(json.load(open(p)))
            except Exception:
                pass

    # -- teachers write -----------------------------------------------------------
    def teach_action_effect(self, action, changed, won=False, source="?"):
        d = self.data["action_effects"].setdefault(str(int(action)),
                                                    {"changed": 0, "tried": 0, "won": 0})
        d["tried"] += 1; d["changed"] += int(bool(changed)); d["won"] += int(bool(won))

    def teach_fragment(self, plan, level, reached, source="?", features=None):
        """A verified plan (list of (action_id, data)) that reached `reached`
        levels_completed — a Go-Explore / macro seed for similar situations."""
        if not plan:
            return
        self.data["fragments"].append({"plan": _jsonable(plan), "level": int(level),
                                        "reached": int(reached), "len": len(plan),
                                        "features": features or {}, "source": source})

    def teach_dead_end(self, plan_prefix, source="?"):
        if plan_prefix:
            self.data["dead_ends"].append({"prefix": _jsonable(plan_prefix), "source": source})

    def teach_cell(self, cell_key, path_len, source="?"):
        """Go-Explore archive: keep the SHORTEST known path length to each cell."""
        cur = self.data["cells"].get(cell_key)
        if cur is None or path_len < cur.get("len", 1 << 30):
            self.data["cells"][cell_key] = {"len": int(path_len), "source": source}

    def teach_world_fact(self, text, source="?"):
        if text:
            self.data["world_facts"].append({"text": str(text)[:400], "source": source})

    # -- students read ------------------------------------------------------------
    def action_order(self, level=None):
        """Toddler's intuition: rank actions by observed effectiveness (win-weighted,
        then change-rate). Actions never seen fall back to the canonical order."""
        eff = self.data["action_effects"]
        def score(a):
            d = eff.get(str(a))
            if not d or d["tried"] == 0:
                return -1.0
            return 3.0 * d["won"] / d["tried"] + d["changed"] / d["tried"]
        return sorted(ALL_ACTIONS, key=lambda a: -score(a))

    def seed_plans(self, level, features=None, k=12):
        """Verified fragments to try FIRST (Go-Explore seeds), shortest-first,
        preferring same-level then feature-similar then any."""
        frs = self.data["fragments"]
        same = [f for f in frs if f["level"] == level]
        pool = same or frs
        pool = sorted(pool, key=lambda f: (f["len"], -f["reached"]))
        return [[tuple(s) for s in f["plan"]] for f in pool[:k]]

    def avoid_prefixes(self):
        return [[tuple(s) for s in de["prefix"]] for de in self.data["dead_ends"]]

    def hints(self, level, features=None):
        return {"action_order": self.action_order(level),
                "seed_plans": self.seed_plans(level, features),
                "avoid": self.avoid_prefixes(),
                "n_cells": len(self.data["cells"]),
                "n_fragments": len(self.data["fragments"])}

    # -- consolidation (the sleep teacher: compress/dedup, bound growth) ----------
    def consolidate(self, max_fragments=200, max_dead_ends=500):
        seen, uniq = set(), []
        for f in sorted(self.data["fragments"], key=lambda f: (f["level"], f["len"])):
            # json-serialize the plan so dedup works even when steps carry dict
            # data (e.g. ACTION6 click coords) — tuple(map(tuple, ...)) would
            # choke on the unhashable dict payload.
            key = (f["level"], json.dumps(f["plan"], sort_keys=True, default=str))
            if key not in seen:
                seen.add(key); uniq.append(f)
        self.data["fragments"] = uniq[:max_fragments]
        self.data["dead_ends"] = self.data["dead_ends"][-max_dead_ends:]
        return self

    def save(self):
        os.makedirs(_BB_DIR, exist_ok=True)
        self.data["updated"] = int(time.time())
        json.dump(self.data, open(_path(self.game), "w"))
        return self


def cell_key(frame, bins=8):
    """Downsampled-frame signature for the Go-Explore archive (coarse -> merges
    near-identical states). numpy if available, else a cheap stride sample."""
    try:
        import numpy as np
        f = np.asarray(frame)
        if f.ndim != 2:
            return hashlib.md5(str(frame).encode()).hexdigest()[:16]
        H, W = f.shape
        ys = np.linspace(0, H - 1, min(bins, H)).astype(int)
        xs = np.linspace(0, W - 1, min(bins, W)).astype(int)
        return hashlib.md5(f[np.ix_(ys, xs)].tobytes()).hexdigest()[:16]
    except Exception:
        return hashlib.md5(str(frame).encode()).hexdigest()[:16]


def _jsonable(plan):
    return [[a, (dict(d) if isinstance(d, dict) else d)] for a, d in plan]
