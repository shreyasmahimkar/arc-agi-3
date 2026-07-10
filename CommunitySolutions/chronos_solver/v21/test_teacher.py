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

    # --- R13 robustness: transient network retry (_is_transient / _with_retries) ----
    import urllib.error, socket
    ok &= _check("URLError is transient",
                 T._is_transient(urllib.error.URLError("nodename nor servname")) is True)
    ok &= _check("socket timeout is transient", T._is_transient(socket.timeout()) is True)
    ok &= _check("HTTP 503 is transient",
                 T._is_transient(urllib.error.HTTPError("u", 503, "x", {}, None)) is True)
    ok &= _check("HTTP 429 is transient",
                 T._is_transient(urllib.error.HTTPError("u", 429, "x", {}, None)) is True)
    ok &= _check("HTTP 401 is NOT transient (bad key won't self-heal)",
                 T._is_transient(urllib.error.HTTPError("u", 401, "x", {}, None)) is False)
    ok &= _check("ValueError is NOT transient", T._is_transient(ValueError("x")) is False)

    # _with_retries: fail twice (DNS) then succeed -> returns the value; no real sleep
    naps = []
    calls = {"n": 0}
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.URLError("nodename nor servname provided")
        return "OK"
    ok &= _check("retries recover a transient blip",
                 T._with_retries(flaky, tries=3, base_backoff=0, sleep=naps.append) == "OK"
                 and calls["n"] == 3)
    # exhausts on a persistent transient error -> re-raises the last one
    persist = {"n": 0}
    def always_dns():
        persist["n"] += 1
        raise urllib.error.URLError("nodename nor servname provided")
    raised = False
    try:
        T._with_retries(always_dns, tries=3, base_backoff=0, sleep=naps.append)
    except urllib.error.URLError:
        raised = True
    ok &= _check("persistent transient exhausts then raises", raised and persist["n"] == 3)
    # a non-transient error is raised on the FIRST try (no wasted retries)
    hard = {"n": 0}
    def bad_request():
        hard["n"] += 1
        raise urllib.error.HTTPError("u", 400, "bad", {}, None)
    raised2 = False
    try:
        T._with_retries(bad_request, tries=3, base_backoff=0, sleep=naps.append)
    except urllib.error.HTTPError:
        raised2 = True
    ok &= _check("non-transient fails fast (1 attempt)", raised2 and hard["n"] == 1)

    # --- R14 grounding: the STATE block reaches Opus verbatim, additively -----------
    os.environ["ANTHROPIC_API_KEY"] = "sk-test"
    STATE = "scene: dims=64x64 background=4 n_objects=3\naction->outcome: a1 changed=True"
    # solve_wall: when state is passed, the user prompt carries the grounded block AND
    # the "trust this over your mental simulation" instruction; the source still ships.
    seen = {}
    teaG = T.OpusTeacher()
    teaG._call = lambda s, u: (seen.__setitem__("u", u), '{"plan": [[1, null]]}')[1]
    teaG.solve_wall("ls20", "SRC_CODE", 5, [1, 2, 3], notes="n", state=STATE)
    ok &= _check("grounded prompt carries the state digest", STATE in seen["u"])
    ok &= _check("grounded prompt flags it as ground truth",
                 "CURRENT OBSERVED STATE" in seen["u"] and "TRUST THIS" in seen["u"])
    ok &= _check("grounded prompt still ships the source", "SRC_CODE" in seen["u"])
    # empty/absent state -> byte-identical to the old ungrounded prompt (additive)
    seen2 = {}
    teaG._call = lambda s, u: (seen2.__setitem__("u", u), '{"plan": [[1, null]]}')[1]
    teaG.solve_wall("ls20", "SRC_CODE", 5, [1, 2, 3], notes="n")   # no state
    ok &= _check("no-state prompt omits the grounded block",
                 "CURRENT OBSERVED STATE" not in seen2["u"])
    # iterative threads state into EVERY round (constant ground truth across rounds)
    rounds_seen = []
    teaI = T.OpusTeacher()
    teaI._call = lambda s, u: (rounds_seen.append(u), '{"plan": [[9, null]]}')[1]
    teaI.solve_wall_iterative("ls20", "SRC", 5, [1], lambda p: (False, "no"),
                              max_rounds=2, notes="n", state=STATE)
    ok &= _check("iterative threads state into every round",
                 len(rounds_seen) == 2 and all(STATE in u for u in rounds_seen))

    # --- WM extraction robustness (_strip_module): the recurring ft09 L2 crash -------
    # the exact failure: Opus prepends a prose sentence, so the OLD startswith('```')
    # gate never fired and "Here is ..." reached compile() -> invalid syntax line 1.
    body = "class WorldModel:\n    def __init__(self, obs):\n        self.obs = obs\n"
    ok &= _check("strip: bare module passes through", T._strip_module(body).startswith("class WorldModel"))
    ok &= _check("strip: prose + fenced block -> code only",
                 T._strip_module("Here is the world model:\n```python\n" + body + "```") == body.strip())
    ok &= _check("strip: prose (no fence) -> drops preamble",
                 T._strip_module("Here is the world model:\n" + body) == body.strip())
    ok &= _check("strip: unclosed fence -> code after opening fence",
                 T._strip_module("```python\n" + body) == body.strip())
    ok &= _check("strip: plain ``` fence (no lang) still extracts",
                 T._strip_module("```\n" + body + "```") == body.strip())
    ok &= _check("strip: fence anywhere ignores trailing prose",
                 T._strip_module("Sure!\n```python\n" + body + "```\nHope that helps.") == body.strip())
    ok &= _check("strip: leading module docstring kept (not dropped as prose)",
                 T._strip_module('"""WM."""\n' + body).startswith('"""WM."""'))
    ok &= _check("strip: empty -> None", T._strip_module("   ") is None)
    ok &= _check("strip: None -> None", T._strip_module(None) is None)
    # extracted code must actually compile (the whole point — no stray prose/fence line)
    import ast as _ast
    _extracted = T._strip_module("Here you go:\n```python\n" + body + "```")
    try:
        _ast.parse(_extracted); _compiles = True
    except SyntaxError:
        _compiles = False
    ok &= _check("strip: extracted code compiles clean", _compiles)

    # no key -> no-op (offline guard preserved)
    os.environ.pop("ANTHROPIC_API_KEY", None); os.environ.pop("V21_OPUS_KEY", None)
    ok &= _check("iterative no-ops without a key",
                 T.OpusTeacher().solve_wall_iterative("ls20", "s", 2, [1],
                                                      lambda p: (True, ""), max_rounds=2) is None)

    print("\n" + ("ALL TEACHER OFFLINE TESTS PASSED" if ok else "TEACHER TESTS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
