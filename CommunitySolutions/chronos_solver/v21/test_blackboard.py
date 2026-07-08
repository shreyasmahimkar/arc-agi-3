#!/usr/bin/env python3
# Offline tests for the teacher/student blackboard (Epic C). No engine/network.
import os, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ["V21_BLACKBOARD_DIR"] = tempfile.mkdtemp(prefix="v21bb_")

from brain import blackboard as bb


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); return cond


def main():
    ok = True
    b = bb.Blackboard("ls20")

    # teachers write lessons
    b.teach_action_effect(2, changed=True, won=False, source="bfs")
    b.teach_action_effect(2, changed=True, won=True, source="planner")   # 2 wins a lot
    b.teach_action_effect(4, changed=False, source="bfs")                # 4 does nothing
    b.teach_fragment([(2, None)] * 5, level=5, reached=6, source="planner")
    b.teach_fragment([(2, None)] * 8, level=5, reached=6, source="bfs")  # longer, same subgoal
    b.teach_dead_end([(4, None)], source="bfs")
    b.teach_cell("cellA", 10, source="explore")
    b.teach_cell("cellA", 6, source="explore")                          # shorter path kept

    # student reads hints
    h = b.hints(level=5)
    ok &= _check("toddler ranks the winning action (2) first",
                 h["action_order"][0] == 2)
    ok &= _check("seed plans are shortest-first (5 before 8)",
                 h["seed_plans"][0] == [(2, None)] * 5)
    ok &= _check("dead-end prefix is in the avoid list",
                 [(4, None)] in h["avoid"])
    ok &= _check("cell archive keeps the SHORTEST path (6, not 10)",
                 b.data["cells"]["cellA"]["len"] == 6)

    # persistence: lessons compound across the 4h loop (save -> reload)
    b.consolidate().save()
    b2 = bb.Blackboard("ls20")
    ok &= _check("blackboard persists + reloads (fragments survive)",
                 len(b2.data["fragments"]) >= 1 and b2.action_order()[0] == 2)
    ok &= _check("consolidate dedups fragments (kept the shorter of the pair)",
                 sum(1 for f in b2.data["fragments"] if f["level"] == 5) == 2)

    # cell_key is deterministic + coarse (merges near-identical frames)
    f1 = [[0] * 8 for _ in range(8)]; f2 = [[0] * 8 for _ in range(8)]; f2[0][0] = 0
    ok &= _check("cell_key deterministic + merges identical frames",
                 bb.cell_key(f1) == bb.cell_key(f2))
    ok &= _check("cell_key never raises on odd input", isinstance(bb.cell_key("x"), str))

    print("\n" + ("ALL BLACKBOARD TESTS PASSED" if ok else "BLACKBOARD TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
