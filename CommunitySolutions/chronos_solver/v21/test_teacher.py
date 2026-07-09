#!/usr/bin/env python3
# Offline tests for the Opus teacher's PURE parts (plan parsing, availability,
# exploit-refusal contract). No network / no real API call.
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from brain import teacher as T


def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); return cond


def main():
    ok = True
    # plan parsing: clean JSON
    ok &= _check("parse clean JSON plan",
                 T.parse_plan('{"plan": [[2, null], [6, {"x": 5, "y": 5}]]}')
                 == [(2, None), (6, {"x": 5, "y": 5})])
    # parsing tolerates code fences + prose
    ok &= _check("parse fenced + prose",
                 T.parse_plan('Here:\n```json\n{"plan": [[1,null],[2,null]]}\n```')
                 == [(1, None), (2, None)])
    # int-only steps
    ok &= _check("parse int-only steps", T.parse_plan('{"plan":[3,3,3]}') == [(3, None)] * 3)
    # junk -> None (never fabricates)
    ok &= _check("junk -> None", T.parse_plan("no json here") is None)
    ok &= _check("empty plan -> None", T.parse_plan('{"plan": []}') is None)

    # availability keys off the environment only
    os.environ.pop("ANTHROPIC_API_KEY", None); os.environ.pop("V21_OPUS_KEY", None)
    ok &= _check("no key -> not available", T.OpusTeacher().available() is False)
    ok &= _check("solve_wall no-ops without a key",
                 T.OpusTeacher().solve_wall("ls20", "x=1", 5, [1, 2, 3]) is None)
    os.environ["ANTHROPIC_API_KEY"] = "sk-test"
    ok &= _check("key present -> available", T.OpusTeacher().available() is True)

    # --- R7 teach-with-feedback (solve_wall_iterative) — pure, mocked _call ----------
    # _augment_notes: bounded + folds the failure in, keeps the base note
    aug = T._augment_notes("base note", 0, [(1, None), (2, None)], "reached lvl 4 of 6")
    ok &= _check("augment_notes keeps base", "base note" in aug)
    ok &= _check("augment_notes folds feedback", "reached lvl 4 of 6" in aug and "FAILED" in aug)
    ok &= _check("augment_notes bounded", len(aug) <= 1500)
    ok &= _check("augment_notes tolerates None feedback",
                 isinstance(T._augment_notes("", 1, None, None), str))

    # iterative loop: round 1 plan rejected, round 2 accepted -> returns round-2 plan,
    # and the 2nd prompt must carry the folded feedback (the gradient).
    tea = T.OpusTeacher()
    replies = ['{"plan": [[1, null]]}', '{"plan": [[2, null], [3, null]]}']
    seen_notes = []
    def fake_call(system, user):
        seen_notes.append(user)
        return replies[min(len(seen_notes) - 1, len(replies) - 1)]
    tea._call = fake_call  # no network
    tries = {"n": 0}
    def try_plan(plan):
        tries["n"] += 1
        return (plan == [(2, None), (3, None)], "stalled at action 0")
    got = tea.solve_wall_iterative("ls20", "src", 5, [1, 2, 3], try_plan,
                                   max_rounds=2, notes="local failed")
    ok &= _check("iterative returns verified round-2 plan", got == [(2, None), (3, None)])
    ok &= _check("iterative retried twice", tries["n"] == 2)
    ok &= _check("iterative fed feedback into round 2",
                 len(seen_notes) == 2 and "stalled at action 0" in seen_notes[1])

    # first-round success short-circuits (no wasted rounds)
    tea2 = T.OpusTeacher(); tea2._call = lambda s, u: '{"plan": [[4, null]]}'
    cnt = {"n": 0}
    def tp_ok(plan):
        cnt["n"] += 1; return True, "solved"
    ok &= _check("iterative short-circuits on first success",
                 tea2.solve_wall_iterative("ls20", "s", 2, [4], tp_ok, max_rounds=3) == [(4, None)]
                 and cnt["n"] == 1)

    # never solved -> None after exhausting rounds
    tea3 = T.OpusTeacher(); tea3._call = lambda s, u: '{"plan": [[1, null]]}'
    ok &= _check("iterative exhausts -> None",
                 tea3.solve_wall_iterative("ls20", "s", 2, [1],
                                           lambda p: (False, "no"), max_rounds=2) is None)

    # OBSERVABILITY: even when every round fails (result None), try_plan is invoked
    # once per round WITH each proposed plan — the per-round hook the runner uses to
    # log OPUS_TEACHER activity that would otherwise be invisible in the cron log.
    tea3b = T.OpusTeacher(); tea3b._call = lambda s, u: '{"plan": [[1, null]]}'
    seen = []
    def tp_seen(plan):
        seen.append(plan); return (False, "stalled")
    tea3b.solve_wall_iterative("ls20", "s", 2, [1], tp_seen, max_rounds=3)
    ok &= _check("iterative surfaces every failed round to the hook",
                 len(seen) == 3 and all(p == [(1, None)] for p in seen))

    # no key -> no-op (offline guard preserved)
    os.environ.pop("ANTHROPIC_API_KEY", None); os.environ.pop("V21_OPUS_KEY", None)
    ok &= _check("iterative no-ops without a key",
                 T.OpusTeacher().solve_wall_iterative("ls20", "s", 2, [1],
                                                      lambda p: (True, ""), max_rounds=2) is None)

    print("\n" + ("ALL TEACHER OFFLINE TESTS PASSED" if ok else "TEACHER TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
