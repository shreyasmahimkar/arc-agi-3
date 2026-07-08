#!/usr/bin/env python3
"""
Pure ARC-AGI-3 run: claude-fable-5 (Anthropic's best model) plays a game
on three.arcprize.org via the raw REST API and gets an official scorecard.

Usage:
    python fable5_agent.py --game ls20 --max-actions 250

Requires in .env (same folder) or environment:
    ARC_API_KEY        - from https://three.arcprize.org
    ANTHROPIC_API_KEY  - from https://console.anthropic.com
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
except ImportError:
    pass

from anthropic import Anthropic

ROOT_URL = "https://three.arcprize.org"
HEX = "0123456789ABCDEF"


# ---------------------------------------------------------------- ARC API ---

class ArcClient:
    def __init__(self, api_key: str):
        self.s = requests.Session()  # keeps AWSALB cookies (session affinity)
        self.s.headers.update({"X-API-Key": api_key, "Accept": "application/json"})

    def _post(self, path: str, payload: dict) -> dict:
        r = self.s.post(f"{ROOT_URL}{path}", json=payload, timeout=60)
        if r.status_code != 200:
            raise RuntimeError(f"POST {path} -> {r.status_code}: {r.text[:300]}")
        return r.json()

    def open_scorecard(self, tags: list[str]) -> str:
        return self._post("/api/scorecard/open", {"tags": tags})["card_id"]

    def close_scorecard(self, card_id: str) -> dict:
        return self._post("/api/scorecard/close", {"card_id": card_id})

    def get_scorecard(self, card_id: str) -> dict:
        r = self.s.get(f"{ROOT_URL}/api/scorecard/{card_id}", timeout=60)
        return r.json() if r.status_code == 200 else {}

    def reset(self, game_id: str, card_id: str, guid: str | None = None) -> dict:
        payload = {"game_id": game_id, "card_id": card_id}
        if guid:
            payload["guid"] = guid
        return self._post("/api/cmd/RESET", payload)

    def act(self, action: str, game_id: str, card_id: str, guid: str,
            x: int | None = None, y: int | None = None, reasoning=None) -> dict:
        payload = {"game_id": game_id, "card_id": card_id, "guid": guid}
        if action == "ACTION6":
            payload["x"] = max(0, min(63, int(x or 0)))
            payload["y"] = max(0, min(63, int(y or 0)))
        if reasoning:
            payload["reasoning"] = reasoning
        return self._post(f"/api/cmd/{action}", payload)


# ------------------------------------------------------------- rendering ----

def render_grid(frame: list) -> str:
    """Frame is a list of 2D grids (usually 1). Cells 0-15 -> hex chars."""
    if not frame:
        return "(empty frame)"
    grid = frame[-1]  # last layer is the visible one
    lines = []
    for row in grid:
        lines.append("".join(HEX[v % 16] for v in row))
    return "\n".join(lines)


def diff_grids(prev: list, cur: list) -> str:
    if not prev or not cur:
        return "n/a"
    a, b = prev[-1], cur[-1]
    if len(a) != len(b):
        return "grid size changed"
    changes = []
    for y in range(len(b)):
        for x in range(len(b[y])):
            if a[y][x] != b[y][x]:
                changes.append((x, y, a[y][x], b[y][x]))
    if not changes:
        return "NO CHANGE (action had no visible effect)"
    if len(changes) > 60:
        xs = [c[0] for c in changes]
        ys = [c[1] for c in changes]
        return (f"{len(changes)} cells changed, in region "
                f"x[{min(xs)}-{max(xs)}] y[{min(ys)}-{max(ys)}]")
    return "; ".join(f"({x},{y}) {HEX[o % 16]}->{HEX[n % 16]}" for x, y, o, n in changes)


# 256-color ANSI approximations for the 16 ARC cell values
ANSI_COLORS = [16, 27, 196, 40, 226, 244, 201, 208, 45, 88,
               231, 99, 130, 22, 165, 51]


def print_grid_ansi(frame: list, label: str = "") -> None:
    """Draw the grid in the terminal with colored blocks (2 chars per cell)."""
    if not frame:
        return
    grid = frame[-1]
    out = [f"\n--- {label} ---"] if label else []
    for row in grid:
        out.append("".join(f"\x1b[48;5;{ANSI_COLORS[v % 16]}m  " for v in row)
                   + "\x1b[0m")
    print("\n".join(out))


def fmt_actions(available) -> str:
    if not available:
        return "RESET, ACTION1-ACTION6"
    names = []
    for a in available:
        if isinstance(a, int):
            names.append("RESET" if a == 0 else f"ACTION{a}")
        else:
            names.append(str(a))
    return ", ".join(names)


# ----------------------------------------------------------------- agent ----

SYSTEM_PROMPT = """\
You are an expert game-playing agent competing on ARC-AGI-3, an interactive \
reasoning benchmark. You are dropped into an unknown 64x64 grid game with NO \
instructions. You must discover the rules, objects, controls, and goal purely \
by experimenting and observing how the grid changes after each action.

Grid encoding: each character is one cell, values 0-15 shown as hex 0-F. \
Coordinates are (x, y): x = column 0-63 left to right, y = row 0-63 top to bottom.

Actions:
- ACTION1..ACTION4: often map to movement (up/down/left/right) but may differ
- ACTION5: often interact/select/confirm
- ACTION6: click a cell, requires x and y
- ACTION7: sometimes available (e.g. undo)
- RESET: restart level (use only when stuck or after GAME_OVER)

Strategy:
1. Early on, systematically test actions and watch the diff to learn controls.
2. Identify the player/avatar, walls, collectables, keys/locks, and the goal.
3. The game has multiple levels; completing a level increases your score.
4. If "NO CHANGE" repeats, that action is blocked there - try something else.
5. Build and refine a hypothesis of the rules. Exploit it to finish levels fast.
6. Keep your memory concise and factual (under 120 words): controls learned, \
map layout, goal, current plan. Never let it grow unbounded.

Respond ONLY with a JSON object:
{"observation": "<what changed and what it implies, 1-2 sentences>",
 "memory": "<UPDATED persistent notes: controls, rules, layout, current plan>",
 "action": "<RESET|ACTION1|ACTION2|ACTION3|ACTION4|ACTION5|ACTION6|ACTION7>",
 "x": <0-63, only for ACTION6>, "y": <0-63, only for ACTION6>}\
"""


class Fable5Agent:
    def __init__(self, model: str, history_turns: int = 6):
        self.client = Anthropic()  # uses ANTHROPIC_API_KEY
        self.model = model
        self.history: list[dict] = []      # rolling window of message dicts
        self.history_turns = history_turns
        self.memory = "(empty - first move)"
        self.tokens_in = 0
        self.tokens_out = 0

    def build_turn(self, frame: dict, prev_frame: dict | None, last_action: str,
                   action_no: int, max_actions: int) -> str:
        score = frame.get("score", frame.get("levels_completed", 0))
        win = frame.get("win_score", frame.get("win_levels", "?"))
        parts = [
            f"Action {action_no}/{max_actions} | state={frame.get('state')} "
            f"| levels completed: {score}/{win}",
            f"Last action taken: {last_action}",
            f"Diff vs previous frame: "
            f"{diff_grids(prev_frame.get('frame', []) if prev_frame else [], frame.get('frame', []))}",
            f"Available actions: {fmt_actions(frame.get('available_actions'))}",
            f"Your memory:\n{self.memory}",
            "Current grid:",
            render_grid(frame.get("frame", [])),
            "Choose the next action. JSON only.",
        ]
        return "\n".join(parts)

    def choose(self, frame: dict, prev_frame: dict | None, last_action: str,
               action_no: int, max_actions: int) -> dict:
        user_msg = {"role": "user",
                    "content": self.build_turn(frame, prev_frame, last_action,
                                               action_no, max_actions)}
        msgs = self.history[-(self.history_turns * 2):] + [user_msg]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=3000,
            system=SYSTEM_PROMPT,
            messages=msgs,
        )
        self.tokens_in += resp.usage.input_tokens
        self.tokens_out += resp.usage.output_tokens
        text = "".join(b.text for b in resp.content if b.type == "text")

        decision = self.parse(text)
        if decision.get("memory"):
            self.memory = decision["memory"]
        # store a compact version of the user turn in history (no full grid)
        compact = user_msg["content"].split("Current grid:")[0] + "(grid omitted)"
        self.history.append({"role": "user", "content": compact})
        self.history.append({"role": "assistant", "content": text})
        return decision

    @staticmethod
    def parse(text: str) -> dict:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                if str(d.get("action", "")).upper() in (
                        ["RESET"] + [f"ACTION{i}" for i in range(1, 8)]):
                    d["action"] = d["action"].upper()
                    return d
            except json.JSONDecodeError:
                pass
        m = re.search(r"(RESET|ACTION[1-7])", text.upper())
        if m:
            return {"action": m.group(1), "_fallback": True}
        return {"action": "ACTION5", "_fallback": True}


# ------------------------------------------------------------------ main ----

def main() -> None:
    ap = argparse.ArgumentParser(description="claude-fable-5 plays ARC-AGI-3")
    ap.add_argument("--game", default="ls20")
    ap.add_argument("--model", default=os.getenv("FABLE_MODEL", "claude-opus-4-8"))
    ap.add_argument("--max-actions", type=int, default=250)
    ap.add_argument("--history-turns", type=int, default=6)
    ap.add_argument("--render", action="store_true",
                    help="draw the colored game grid in the terminal each step")
    args = ap.parse_args()

    arc_key = os.getenv("ARC_API_KEY")
    if not arc_key or not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Set ARC_API_KEY and ANTHROPIC_API_KEY in fable5/.env")

    arc = ArcClient(arc_key)
    agent = Fable5Agent(args.model, args.history_turns)

    # resolve full game_id (API expects e.g. "ls20-xxxxxxxx" or accepts prefix)
    games = arc.s.get(f"{ROOT_URL}/api/games", timeout=60).json()
    game_id = next((g["game_id"] for g in games
                    if g["game_id"].startswith(args.game)), args.game)

    card_id = arc.open_scorecard(tags=["fable5", args.model])
    print(f"Scorecard: {card_id}")
    print(f"Game: {game_id} | Model: {args.model} | Max actions: {args.max_actions}\n")

    log = {"card_id": card_id, "game_id": game_id, "model": args.model,
           "started": datetime.now(timezone.utc).isoformat(), "steps": []}

    frame = arc.reset(game_id, card_id)
    guid = frame["guid"]
    if args.render:
        print_grid_ansi(frame.get("frame", []), "initial frame")
    prev_frame, last_action = None, "RESET (initial)"
    best = frame.get("score", frame.get("levels_completed", 0))

    try:
        for n in range(1, args.max_actions + 1):
            state = frame.get("state")
            if state == "WIN":
                print("\n*** WIN - all levels completed! ***")
                break
            if state == "GAME_OVER":
                print("GAME_OVER -> RESET")
                prev_frame, frame = frame, arc.reset(game_id, card_id, guid)
                guid = frame.get("guid", guid)
                last_action = "RESET (after game over)"
                continue

            d = agent.choose(frame, prev_frame, last_action, n, args.max_actions)
            if d.get("_fallback"):
                # model reply didn't parse as JSON - explore instead of looping
                import random
                avail = [a for a in (frame.get("available_actions") or [1, 2, 3, 4, 5])
                         if a != 0]
                d["action"] = f"ACTION{random.choice(avail)}" \
                    if isinstance(avail[0], int) else str(random.choice(avail))
                d["x"], d["y"] = random.randint(0, 63), random.randint(0, 63)
                print(f"      (warn: unparseable model reply, exploring with {d['action']})")
            action = d["action"]
            reasoning = {"observation": d.get("observation", ""),
                         "model": args.model}

            if action == "RESET":
                new = arc.reset(game_id, card_id, guid)
            else:
                new = arc.act(action, game_id, card_id, guid,
                              d.get("x"), d.get("y"), reasoning)
            prev_frame, frame = frame, new
            guid = frame.get("guid", guid)

            score = frame.get("score", frame.get("levels_completed", 0))
            if args.render:
                print_grid_ansi(frame.get("frame", []),
                                f"after {action} (step {n}, levels {score})")
            coord = f" ({d.get('x')},{d.get('y')})" if action == "ACTION6" else ""
            print(f"[{n:3d}] {action}{coord:10s} | levels: {score} "
                  f"| state: {frame.get('state')} | {d.get('observation', '')[:90]}")
            if score > best:
                best = score
                print(f"      >>> LEVEL UP! Now at {score} <<<")
            log["steps"].append({"n": n, "action": action,
                                 "x": d.get("x"), "y": d.get("y"),
                                 "score": score, "state": frame.get("state"),
                                 "observation": d.get("observation")})
    except KeyboardInterrupt:
        print("\nInterrupted - closing scorecard...")
    except Exception as e:
        print(f"\nRun aborted by error: {e}\nClosing scorecard and saving log...")

    final = arc.close_scorecard(card_id)
    detail = arc.get_scorecard(card_id)

    print("\n" + "=" * 60)
    print("FINAL SCORECARD")
    print(json.dumps(detail or final, indent=2)[:2000])
    print(f"\nLevels completed (best): {best}")
    print(f"Tokens: {agent.tokens_in:,} in / {agent.tokens_out:,} out")
    print(f"View online: {ROOT_URL}/scorecards/{card_id}")
    print(f"          or https://arcprize.org/scorecards/{card_id}")

    log["finished"] = datetime.now(timezone.utc).isoformat()
    log["final_scorecard"] = detail or final
    log["tokens"] = {"in": agent.tokens_in, "out": agent.tokens_out}
    out = Path(__file__).parent / f"run_{args.game}_{int(time.time())}.json"
    out.write_text(json.dumps(log, indent=2))
    print(f"Run log saved: {out.name}")


if __name__ == "__main__":
    main()
