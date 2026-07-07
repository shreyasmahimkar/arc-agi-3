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

    # 2b) OllamaBackend hard deadline: a hung request must RAISE (never stall the
    #     cadence) within ~the deadline, so the coder can fall back to safety nets.
    import time as _time
    _olla = lb.OllamaBackend()
    os.environ["V21_OLLAMA_DEADLINE"] = "5"  # 5s is the enforced floor
    _olla._complete_raw = lambda *a, **k: _time.sleep(60) or "never"  # simulate a stall
    _t0 = _time.time()
    try:
        _olla.complete("hi"); _raised = False
    except Exception:
        _raised = True
    _elapsed = _time.time() - _t0
    ok &= _check("ollama deadline raises on hang", _raised)
    ok &= _check("ollama deadline bounded (<10s, not the 60s stall)", _elapsed < 10.0)
    os.environ.pop("V21_OLLAMA_DEADLINE", None)

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

    # 5b) Stage-3.5 replay_wins (BACKLOG #3): pure fork-replay over injected
    #     clone/play closures — the try_plan_fn contract cadence_runner wires in.
    def _clone(s):
        return dict(s)
    def _play_maze(s, step):  # ls20-like: goal after 3 ACTION2 steps
        aid, _ = step
        if aid == 2:
            s["n"] = s.get("n", 0) + 1
        return 1 if s.get("n", 0) >= 3 else 0
    ok &= _check("replay_wins: winning plan reaches goal",
                 rc.replay_wins({}, [(2, None)] * 3, _clone, _play_maze, goal=1) is True)
    ok &= _check("replay_wins: short plan does NOT win",
                 rc.replay_wins({}, [(2, None)] * 2, _clone, _play_maze, goal=1) is False)
    ok &= _check("replay_wins: empty plan -> False",
                 rc.replay_wins({}, [], _clone, _play_maze, goal=1) is False)
    _start = {}
    rc.replay_wins(_start, [(2, None)] * 3, _clone, _play_maze, goal=1)
    ok &= _check("replay_wins: does not mutate start", _start == {})

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

    #   (e) click-REPEAT win (vc33-like): hammer ONE coord ×k -> shortest k on
    #       the RIGHT target (a different target repeated never wins)
    def _play_click_rep(s, step):
        aid, data = step
        if aid == 6 and data == {"x": 46, "y": 56}:
            s["c"] = s.get("c", 0) + 1
        return 1 if s.get("c", 0) >= 3 else 0
    p_crep = blitz.blitz_solve({}, 0, [1, 2],
                               [{"x": 12, "y": 56}, {"x": 46, "y": 56}],
                               _clone, _play_click_rep, repeat_K=20)
    ok &= _check("blitz solves click-REPEAT (shortest k=3 on right coord)",
                 p_crep == [(6, {"x": 46, "y": 56})] * 3)

    #   (f) prefers a shorter simple-action win over a longer click-repeat
    #       (exercises the len(best)-1 cap that short-circuits the click tier)
    def _play_mixed(s, step):
        aid, data = step
        if aid == 2:
            s["a"] = s.get("a", 0) + 1
        if aid == 6 and data == {"x": 5, "y": 5}:
            s["k"] = s.get("k", 0) + 1
        return 1 if (s.get("a", 0) >= 2 or s.get("k", 0) >= 5) else 0
    p_mixed = blitz.blitz_solve({}, 0, [2], [{"x": 5, "y": 5}],
                                _clone, _play_mixed, repeat_K=20)
    ok &= _check("blitz prefers shorter simple win over longer click-repeat",
                 p_mixed == [(2, None), (2, None)])

    # 9b) blitz MACRO replay (BACKLOG #4/#9): replay a solved sibling's plan as a
    #     Go-Explore seed; keep only the shortest winning prefix; never fabricate.
    def _play_seq(s, step):  # wins as soon as action 3 has been played
        aid, _ = step
        if aid == 3:
            s["won"] = True
        return 1 if s.get("won") else 0
    #   (a) a matching sibling macro wins
    p_macro = blitz.blitz_macros(
        {}, 0, [[(1, None), (2, None), (3, None)]], _clone, _play_seq)
    ok &= _check("blitz_macros replays a winning sibling plan",
                 p_macro == [(1, None), (2, None), (3, None)])
    #   (b) overshooting macro -> trimmed to shortest winning prefix
    p_trim = blitz.blitz_macros(
        {}, 0, [[(1, None), (2, None), (3, None), (4, None), (5, None)]],
        _clone, _play_seq)
    ok &= _check("blitz_macros trims to shortest winning prefix",
                 p_trim == [(1, None), (2, None), (3, None)])
    #   (c) prefers the shortest winner across several winning macros
    p_short = blitz.blitz_macros(
        {}, 0, [[(9, None), (9, None), (3, None)],
                [(3, None)]], _clone, _play_seq)
    ok &= _check("blitz_macros prefers shortest winning macro",
                 p_short == [(3, None)])
    #   (d) no macro wins -> None (never fabricates)
    ok &= _check("blitz_macros returns None when no macro wins",
                 blitz.blitz_macros({}, 0, [[(9, None), (8, None)]],
                                    _clone, _play_seq) is None)

    # 9c) B1 wiring: merge_click_targets fuses engine-scanned clicks with
    #     perception component centroids, deduped, scan-first, env-gated.
    _scan = [{"x": 1, "y": 1}, {"x": 5, "y": 5}]
    _perc = [{"x": 5, "y": 5}, {"x": 9, "y": 2}]  # (5,5) dup, (9,2) new
    #   (a) OFF -> scan clicks unchanged (deduped), no perception targets added
    off = blitz.merge_click_targets(_scan, [[0]], False,
                                    perception_fn=lambda f: _perc)
    ok &= _check("merge_click_targets OFF -> scan-only (no perception)",
                 off == _scan)
    #   (b) ON -> appends only NEW perception centroids after the scan targets
    on = blitz.merge_click_targets(_scan, [[0]], True,
                                   perception_fn=lambda f: _perc)
    ok &= _check("merge_click_targets ON -> adds new perception coord, dedups",
                 on == [{"x": 1, "y": 1}, {"x": 5, "y": 5}, {"x": 9, "y": 2}])
    #   (c) real perception_fn: two separate same-colour blobs -> both targets
    _grid_two = [
        [0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ]
    real = blitz.merge_click_targets([], _grid_two, True)
    ok &= _check("merge_click_targets ON -> real perception yields 2 blob coords",
                 len(real) == 2 and {t["x"] for t in real} == {1, 3})

    # 10) brain layer (BACKLOG Epic B) — cognitive subsystems, all pure/offline.
    from brain import perception as P
    from brain import world_model as WM
    from brain import hypotheses as HY
    from brain import planner as PL
    from brain import memory as MEM
    from brain import goal as GO
    ok &= _check("brain modules import", True)

    # 10a) perception: connected components on a grid with TWO separate blobs
    #      of the same colour (bg=0). Per-colour median would give one point
    #      between them; components give TWO distinct objects/click targets.
    grid = [
        [0, 0, 0, 0, 0],
        [0, 3, 0, 3, 0],
        [0, 3, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ]
    sc = P.scene(grid)
    ok &= _check("perception finds 2 components (same colour, separate blobs)",
                 sc["n_objects"] == 2 and sc["background"] == 0)
    ct = P.click_targets(grid)
    #   two distinct click targets at the two blob centroids (col=1 and col=3)
    ok &= _check("perception yields 2 distinct click targets",
                 len(ct) == 2 and {t["x"] for t in ct} == {1, 3}
                 and all(t["y"] == 2 for t in ct))  # median row of {1,2} == 2
    #   diff: turning one blob's top cell to bg registers exactly one change
    g2 = [row[:] for row in grid]; g2[1][1] = 0
    d = P.diff(grid, g2)
    ok &= _check("perception diff detects 1 disappeared cell",
                 d["n_changed"] == 1 and d["disappeared"] == [(1, 1)])
    #   numpy-free but numpy-compatible: a .tolist()-able object is accepted
    class _Arr:
        def __init__(self, g): self._g = g
        def tolist(self): return self._g
    ok &= _check("perception accepts array-like (.tolist)",
                 P.scene(_Arr(grid))["n_objects"] == 2)

    # 10b) world_model verifier: a correct predictor reproduces records; a
    #      wrong one is flagged and not trusted.
    recs = [("s0", "a", "s1"), ("s1", "a", "s2")]
    good = WM.verify_model(lambda prev, a: {"s0": "s1", "s1": "s2"}[prev], recs)
    ok &= _check("world_model verifier trusts a correct model",
                 good["accuracy"] == 1.0 and WM.is_trusted(good))
    bad = WM.verify_model(lambda prev, a: "WRONG", recs)
    ok &= _check("world_model verifier rejects a wrong model",
                 bad["accuracy"] == 0.0 and not WM.is_trusted(bad))

    # 10c) hypotheses: falsify drops mispredictors; discriminating action is the
    #      one whose predictions split the hypotheses the most.
    #   two hypotheses, only H_b predicts "X" for action 1 -> observing "X"
    #   keeps H_b, drops H_a.
    hyps = [{"name": "a"}, {"name": "b"}]
    def _pred(h, a):
        # H_a: action1->"P", action2->"Q";  H_b: action1->"X", action2->"Q"
        table = {"a": {1: "P", 2: "Q"}, "b": {1: "X", 2: "Q"}}
        return table[h["name"]][a]
    surv = HY.falsify(hyps, 1, "X", _pred)
    ok &= _check("hypotheses.falsify keeps only consistent hypothesis",
                 len(surv) == 1 and surv[0]["name"] == "b")
    #   action 1 splits the two hypotheses (P vs X); action 2 does not (Q vs Q)
    da = HY.most_discriminating_action(hyps, [2, 1], _pred)
    ok &= _check("hypotheses picks the discriminating action (1, not 2)", da == 1)

    # 10d) planner: BFS in a toy model + MPC executor abort-on-mismatch.
    #   model: integer state, action +1; goal at 3 -> plan is [+1,+1,+1]
    plan = PL.plan_in_model(0, [1], lambda s, a: s + a, lambda s: s == 3, max_depth=8)
    ok &= _check("planner.plan_in_model finds shortest plan in model",
                 plan == [1, 1, 1])
    #   executor: model predicts frames "1","2","3"; real diverges at step 2
    #   (returns "2","BAD") -> mismatch_at == 1, not won.
    seqp = iter(["1", "2", "3"])
    seqr = iter([1, 1, 1])          # levels_completed (never reaches goal=1? use 0)
    obs = iter(["1", "BAD"])
    res = PL.execute_and_verify(
        [1, 1, 1],
        real_play=lambda a: 0,
        model_predict=lambda a: next(seqp),
        observe=lambda: next(obs),
        goal_completed=1)
    ok &= _check("planner MPC aborts at first frame mismatch",
                 res["mismatch_at"] == 1 and res["won"] is False)

    # 10e) memory: a perceptual key transfers across games (same structure) and
    #      retrieval ranks the structurally-similar concept first.
    k_here = MEM.perceptual_key(sc)
    lib = [
        {"key": MEM.perceptual_key(P.scene(grid)), "concept": "same_shape"},
        {"key": (99, 99, 0, ()), "concept": "unrelated"},
    ]
    top = MEM.retrieve(lib, k_here, k=1)
    ok &= _check("memory retrieves the structurally-similar concept first",
                 top and top[0][1]["concept"] == "same_shape" and top[0][0] == 1.0)

    # 10f) goal induction from the score signal.
    gd = GO.induce_from_scores([0, 0, 1, 1, 2])
    ok &= _check("goal induces 'maximize_progress' from a rising signal",
                 gd["kind"] == "maximize_progress" and gd["monotone"] is True)
    ok &= _check("goal_reached_by_progress fires on a level advance",
                 GO.goal_reached_by_progress(1, 2) is True
                 and GO.goal_reached_by_progress(2, 2) is False)

    print("\n" + ("ALL OFFLINE TESTS PASSED" if ok else "OFFLINE TESTS FAILED"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
