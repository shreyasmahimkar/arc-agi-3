#!/usr/bin/env python3
# Offline self-test — no GPU, no arcengine, no network. The autonomous coder MUST
# run this green before committing any change. Exercises every module the loop
# depends on except the arcengine BFS (which only runs on the Mac).
import sys, os, importlib

def _check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    return cond

def main():
    ok = True
    import tempfile
    _HIST = os.path.join(tempfile.gettempdir(), "v21_test_evolution_history.jsonl")
    # 1) all modules import + compile
    for m in ("llm_backend", "runtime_coder", "intuition", "evolve", "blitz", "cadence_runner"):
        try:
            importlib.import_module(m); ok &= _check(f"import {m}", True)
        except Exception as e:
            ok &= _check(f"import {m}: {e}", False)

    import llm_backend as lb, runtime_coder as rc, intuition, evolve

    # 2) mock backend auto-selects offline
    ok &= _check("backend auto -> mock (offline)", lb.get_backend().name == "mock")

    # 3) runtime code-writer: write -> sandbox-exec -> plan -> win (ft09-like + ls20-like)
    coder = rc.RuntimeCoder(lb.get_backend())
    w1 = coder.solve_level({"note": "blind"}, lambda p: len(p) == 1 and p[0][0] == 6)
    ok &= _check("runtime_coder solves ft09-like (1xACTION6)", w1 == [(6, None)])
    w2 = coder.solve_level({"note": "maze"},
                           lambda p: len(p) >= 129 and all(a == 2 for a, _ in p[:129]), max_plans=64)
    ok &= _check("runtime_coder solves ls20-like (repeat ACTION2)", bool(w2))

    # 4) exploit refusal (R2.7)
    ok &= _check("refuses null-coord ACTION6 exploit",
                 rc._refuses_exploit([(6, {"x": None, "y": None})]) is True)

    # 5) sandbox blocks disallowed imports
    bad, err = rc._exec_world_model("import os\nclass WorldModel:\n def __init__(s,o):pass", {})
    ok &= _check("sandbox blocks `import os`", bad is None)

    # 6) intuition distill + order
    prior = intuition.distill("solutions", "intuition_prior.json")
    ip = intuition.IntuitionPrior("intuition_prior.json")
    ok &= _check("intuition order returns 7 actions", len(ip.order_actions("ls20")) == 7)

    # 7) evolve does not promote on a flat evaluator (no spurious promotion)
    champ, promoted = evolve.evolve_step(
        "champion.json", _HIST, [{"game": "ft09", "level": 2}],
        {"ft09": 1.0}, lb.get_backend(), lambda c, g: {x: 0.9 for x in g},
        ["ls20", "ft09", "vc33"], ["cn04", "sk48"], n=4)
    ok &= _check("evolve no-promote on flat eval", promoted is False)

    # 8) config-aware evaluator (BACKLOG #1)
    corpus_rhae = {"ft09": 0.5}
    walls = {"ft09": [{"game": "ft09", "level": 2, "baseline": 8}]}
    #   (a) no probe -> degrades to the config-insensitive corpus floor
    ev_floor = evolve.config_aware_eval_fn(corpus_rhae, walls, probe_fn=None)
    ok &= _check("cfg-eval no-probe -> corpus floor",
                 abs(ev_floor({"blitz_K": 200}, ["ft09"])["ft09"] - 0.5) < 1e-9)

    #   (b) config-SENSITIVE probe: a bigger blitz_K cracks the budget-gated wall
    def _mock_probe(config, gid, lvl):
        return 8 if int(config.get("blitz_K", 0)) >= 400 else None  # 8 == baseline -> RHAE 1
    ev = evolve.config_aware_eval_fn(corpus_rhae, walls, probe_fn=_mock_probe)
    lo = ev({"blitz_K": 200}, ["ft09"])["ft09"]
    hi = ev({"blitz_K": 500}, ["ft09"])["ft09"]
    ok &= _check("cfg-eval is config-sensitive (hi blitz_K scores higher)", hi > lo + 1e-6)
    ok &= _check("cfg-eval never drops below floor", lo >= 0.5 - 1e-9)

    #   (c) end-to-end: a raise-blitz_K challenger PROMOTES under the config-aware eval
    class _StubLLM:
        name = "stub"
        def complete(self, prompt, system=None, max_tokens=800, stop=None):
            return '[{"blitz_K": 500, "note": "raise budget"}]'
    _c0 = evolve.load_champion("champion.json")
    #   generalization-gated: the challenger must also crack a HELD-OUT wall (cn04)
    walls_ee = {"ft09": walls["ft09"],
                "cn04": [{"game": "cn04", "level": 3, "baseline": 8}]}
    champ2, promoted2 = evolve.evolve_step(
        "champion.json", _HIST, walls["ft09"],
        {"ft09": 0.5}, _StubLLM(),
        evolve.config_aware_eval_fn({"ft09": 0.5, "cn04": 0.5}, walls_ee, _mock_probe),
        ["ft09"], ["cn04"], n=2)
    ok &= _check("cfg-eval promotes a blitz_K challenger that cracks a wall", promoted2 is True)
    evolve.save_champion("champion.json", _c0)  # restore seed champion (no side effects)

    # 9) blitz Stage-0 (BACKLOG #2): pure cheap-win search over injected closures
    import blitz
    def _clone(s):
        return dict(s)
    #   (a) single-action win -> length-1 plan (action 3 wins immediately)
    def _play_single(s, step):
        aid, _ = step
        if aid == 3:
            s["won"] = True
        return 1 if s.get("won") else 0
    p_single = blitz.blitz_solve({}, 0, [1, 2, 3, 4], [], _clone, _play_single, repeat_K=50)
    ok &= _check("blitz solves single-action (length-1)", p_single == [(3, None)])

    #   (b) repeat-action win -> shortest k (ls20-like: repeat ACTION2 ×4)
    def _play_repeat(s, step):
        aid, _ = step
        if aid == 2:
            s["n"] = s.get("n", 0) + 1
        return 1 if s.get("n", 0) >= 4 else 0
    p_rep = blitz.blitz_solve({}, 0, [1, 2, 3], [], _clone, _play_repeat, repeat_K=50)
    ok &= _check("blitz solves repeat-action (shortest k=4)", p_rep == [(2, None)] * 4)

    #   (c) click-target win -> the correct ACTION6 coord (vc33-like)
    def _play_click(s, step):
        aid, data = step
        if aid == 6 and data == {"x": 5, "y": 5}:
            s["won"] = True
        return 1 if s.get("won") else 0
    p_click = blitz.blitz_solve({}, 0, [1, 2], [{"x": 1, "y": 1}, {"x": 5, "y": 5}],
                                _clone, _play_click)
    ok &= _check("blitz solves click-target (correct ACTION6 coord)",
                 p_click == [(6, {"x": 5, "y": 5})])

    #   (d) no cheap win -> None (never fabricates a plan)
    ok &= _check("blitz returns None when no cheap win exists",
                 blitz.blitz_solve({}, 0, [1, 2], [], _clone, lambda s, x: 0, repeat_K=10) is None)

    print("\n" + ("ALL OFFLINE TESTS PASSED" if ok else "OFFLINE TESTS FAILED"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
