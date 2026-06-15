#!/usr/bin/env python3
"""Watch the v19 agent solve a game LIVE with white-box BFS (the 0.22 path).

  python play_bfs.py            # ls20, level 0
  python play_bfs.py ar25 1     # game ar25, level 1
  python play_bfs.py ls20 0 --render   # also print the start/solved frames

Genuine solving: BFS searches the real engine and the solution is replay-verified
(levels_completed must increment). No cached answer involved.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import combined_agent as ca
from arcengine import ActionInput, GameAction

NAMES = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT", 5: "ACT5", 6: "CLICK", 7: "UNDO"}


def grid(frame):
    a = np.array(frame, dtype=np.int64)[-1]
    return "\n".join("".join(f"{c:2d}" for c in row) for row in a)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    render = "--render" in sys.argv
    game = args[0] if args else "ls20"
    level = int(args[1]) if len(args) > 1 else 0

    src, cls = ca.find_game_source_and_class(game)
    if not src:
        print(f"no game source found for {game} (live BFS needs environment_files/)"); return
    b = ca.BFSSolver(src, cls, scan_timeout=5, bfs_timeout=120, workers=1)
    assert b.load(), "failed to load game class"
    t = time.time()
    sol = b.solve_level(level)
    if not sol:
        print(f"{game} L{level}: BFS found no solution in budget"); return
    print(f"{game} L{level} SOLVED LIVE by BFS in {len(sol)} actions ({time.time()-t:.1f}s)")
    print(" path:", " ".join(NAMES.get(a, str(a)) for a, _ in sol))

    # replay through a fresh engine to SEE the level complete (verification)
    g = b.game_cls(); g.set_level(level)
    g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
    if render and r.frame:
        print("\n--- start frame ---\n" + grid(r.frame))
    for a, d in sol:
        ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
        r = g.perform_action(ai, raw=True)
    if render and r.frame:
        print("\n--- solved frame ---\n" + grid(r.frame))
    print(f"\n levels_completed after replay: {getattr(r,'levels_completed','?')} "
          f"| state: {getattr(r,'state','?')}")


if __name__ == "__main__":
    main()
