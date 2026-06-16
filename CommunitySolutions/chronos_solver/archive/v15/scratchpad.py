"""v15 — the GAME SCRATCHPAD: the shared memory between the three passes.

PASS 1 (offline, engine available): the v13 symbolic scout — pure BFS +
    probing — runs for a time budget and WRITES the scratchpad: solved
    levels with exact solutions, per-action effect measurements, where it
    got stuck, and human/LLM-readable notes.
PASS 2 (offline): the scratchpad is distilled into the PLM — solutions
    become chained expert replays (value/reward head food, via gen_data),
    action-effect notes become structured priors.
PASS 3 (everywhere, incl. hidden eval): the PLM plays with deep-think,
    maintaining its OWN live scratchpad learned in-episode (the only kind
    allowed at eval — the integrity line: offline scratchpads never ship
    as lookup tables, they ship as weights).

Format: one JSON per game — deliberately simple and readable, so any
future LLM pass can consume it as text.

{
  "game_id": "ar25",
  "created": "...", "pass1_budget_s": 600,
  "levels": 7, "stuck_at": 2,
  "solved": {"0": [[2, null], ...], "1": [[3, null], ...]},
  "action_effects": {"2": {"tries": 5, "changes": 5, "avg_px": 14.2}},
  "notes": ["L0 solved in 15 actions",
            "ACTION2 changes ~14px/use (5/5 tries)",
            "ACTION6 click (12,40) changed 230px"]
}
"""
import json
import os
import time


def new_scratchpad(game_id, budget_s=None):
    return {"game_id": game_id,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pass1_budget_s": budget_s,
            "levels": None, "stuck_at": None,
            "solved": {}, "action_effects": {}, "notes": []}


def add_solution(sp, level, actions):
    """actions: [[aid, data_or_None], ...] — v13 cache format, verbatim."""
    sp["solved"][str(level)] = actions
    sp["notes"].append(f"L{level} solved in {len(actions)} actions")


def add_action_effect(sp, aid, changed, n_px):
    e = sp["action_effects"].setdefault(str(aid),
                                        {"tries": 0, "changes": 0,
                                         "avg_px": 0.0})
    e["avg_px"] = (e["avg_px"] * e["changes"] + n_px) / max(e["changes"] + (1 if changed else 0), 1) \
        if changed else e["avg_px"]
    e["tries"] += 1
    e["changes"] += 1 if changed else 0


def add_note(sp, text):
    sp["notes"].append(text)


def save(sp, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, f"{sp['game_id']}.json")
    with open(p, "w") as f:
        json.dump(sp, f, indent=1)
    return p


def load(game_id, scratch_dir):
    p = os.path.join(scratch_dir, f"{game_id}.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    return None
