#!/usr/bin/env python3
"""Watch the v19 agent solve a game LIVE with white-box BFS — or play it yourself.

  python play_bfs.py                   # ls20 L0, BFS solves it (the 0.22 path)
  python play_bfs.py ar25 1            # game ar25, level 1
  python play_bfs.py ls20 0 --render   # also print the start/solved frames
  python play_bfs.py ls20 0 --human    # PLAY IT YOURSELF (interactive)

Genuine solving: BFS searches the real engine and the solution is replay-verified
(levels_completed must increment). No cached answer involved.

--human controls: 1-7 = ACTION1..ACTION7 | w/a/s/d = ACTION1/3/2/4 |
                  6 (or click) prompts for "x y" | r = reset | u = undo (ACTION7) |
                  q = quit | ? = help
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import numpy as np
import combined_agent as ca
from arcengine import ActionInput, GameAction

NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT5", 6: "CLICK", 7: "UNDO"}
PAL = ".123456789ABCDEF"          # color value 0..15 -> single char
WASD = {"w": 1, "s": 2, "a": 3, "d": 4}


def _arr(frame):
    return np.array(frame, dtype=np.int64)[-1]


def grid(frame, compact=False):
    a = _arr(frame)
    if compact:
        return "\n".join("".join(PAL[c % 16] for c in row) for row in a)
    return "\n".join("".join(f"{c:2d}" for c in row) for row in a)


def _avail(r):
    aa = getattr(r, "available_actions", None) or []
    return sorted(a.value if hasattr(a, "value") else int(a) for a in aa)


def _load_game(game, level):
    src, cls = ca.find_game_source_and_class(game)
    if not src:
        print(f"no game source found for {game} (needs environment_files/{game}/.../{game}.py)")
        return None, None
    b = ca.BFSSolver(src, cls, scan_timeout=5, bfs_timeout=120, workers=1)
    assert b.load(), "failed to load game class"
    return b, src


def _fresh(b, level):
    g = b.game_cls(); g.set_level(level)
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    return g, r


def _step(g, aid, data=None):
    ai = ActionInput(id=GameAction.from_id(aid), data=data) if data else ActionInput(id=GameAction.from_id(aid))
    return g.perform_action(ai, raw=True)


def play_human(game, level):
    b, _ = _load_game(game, level)
    if b is None:
        return
    g, r = _fresh(b, level)
    print(f"\n=== PLAYING {game} L{level} ===")
    print("controls: 1-7 actions | w/a/s/d = ACT1/3/2/4 | 6=click(x y) | r reset | u undo | q quit | ? help")
    _show(r, game, level)
    while True:
        try:
            cmd = input("move > ").strip().lower()
        except EOFError:
            print(); break
        if cmd in ("q", "quit"):
            break
        if cmd in ("?", "h", "help"):
            print("  1-7 = ACTION1..7 | w/a/s/d = ACT1/3/2/4 | 6/click -> x y | r reset | u undo | q quit")
            continue
        if cmd in ("r", "reset"):
            g, r = _fresh(b, level); _show(r, game, level); continue
        if cmd in ("u", "undo"):
            r = _step(g, 7); _show(r, game, level); continue
        aid = WASD.get(cmd)
        if aid is None and cmd.split()[0:1] == ["click"]:
            cmd = "6 " + " ".join(cmd.split()[1:])
        parts = cmd.split()
        if aid is None and parts and parts[0].isdigit():
            aid = int(parts[0])
        if aid is None or not (1 <= aid <= 7):
            print("  ? unknown — type ? for help"); continue
        data = None
        if aid == 6:                               # click needs coordinates
            xy = parts[1:] if len(parts) >= 3 else input("  click x y > ").split()
            if len(xy) < 2 or not all(t.lstrip('-').isdigit() for t in xy[:2]):
                print("  click needs: x y (0-63)"); continue
            data = {"x": int(xy[0]), "y": int(xy[1])}
        r = _step(g, aid, data)
        _show(r, game, level)
        st = str(getattr(r, "state", ""))
        if "WIN" in st or getattr(r, "levels_completed", 0) > level:
            print(f"\n  *** LEVEL {level} COMPLETE ({st}) ***"); break


def _show(r, game, level):
    if r.frame:
        print(grid(r.frame, compact=True))
    print(f"[{game} L{level}] levels_completed={getattr(r,'levels_completed','?')} "
          f"state={getattr(r,'state','?')} avail={_avail(r)}")


def solve(game, level, render):
    b, src = _load_game(game, level)
    if b is None:
        return
    t = time.time(); sol = b.solve_level(level)
    if not sol:
        print(f"{game} L{level}: BFS found no solution in budget"); return
    print(f"{game} L{level} SOLVED LIVE by BFS in {len(sol)} actions ({time.time()-t:.1f}s)")
    print(" path:", " ".join(NAMES.get(a, str(a)) for a, _ in sol))
    g, r = _fresh(b, level)
    if render and r.frame:
        print("\n--- start frame ---\n" + grid(r.frame, compact=True))
    for a, d in sol:
        r = _step(g, a, d)
    if render and r.frame:
        print("\n--- solved frame ---\n" + grid(r.frame, compact=True))
    print(f"\n levels_completed after replay: {getattr(r,'levels_completed','?')} "
          f"| state: {getattr(r,'state','?')}")


def main():
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    pos = [a for a in sys.argv[1:] if not a.startswith("--")]
    game = pos[0] if pos else "ls20"
    level = int(pos[1]) if len(pos) > 1 else 0
    if "--human" in flags:
        play_human(game, level)
    else:
        solve(game, level, render="--render" in flags)


if __name__ == "__main__":
    main()
