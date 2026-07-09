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

    # 5a) numpy-style frame indexing must NOT crash candidate_plans. An LLM often
    #     writes `frame[y, x]`; on a Python-list frame that raises 'list indices
    #     must be integers or slices, not tuple' (ft09 L2 wall crash, cron line 34).
    #     _coerce_obs now hands the model a numpy frame so both index styles work.
    _np_code = ("class WorldModel:\n"
                " def __init__(self, o):\n"
                "  self.f = o.get('frame'); self.a = o.get('available_actions', [1])\n"
                " def candidate_plans(self, max_len):\n"
                "  _ = int(self.f[0, 0])  # numpy tuple-index; crashes on a list frame\n"
                "  return [[(a, None)] for a in self.a]\n")
    _wm, _e = rc._exec_world_model(_np_code, {"frame": [[1, 2], [3, 4]],
                                              "available_actions": [1, 2]})
    _plans = _wm.candidate_plans(10) if _wm is not None else None
    ok &= _check("runtime_coder: numpy frame[y,x] indexing does not crash",
                 _wm is not None and _plans == [[(1, None)], [(2, None)]])

    # 5a1) LLM-authored candidate_plans routinely calls str()/type()/exception
    #      types; these were missing from _SAFE_BUILTINS, so OPUS_WM on ls20 L5
    #      crashed "name 'str' is not defined" (cron 152556Z). The sandbox must
    #      now expose common pure value/type builtins (still no open/eval/exec).
    _str_code = ("class WorldModel:\n"
                 " def __init__(self, o):\n"
                 "  self.a = o.get('available_actions', [1])\n"
                 " def candidate_plans(self, max_len):\n"
                 "  key = str(self.a[0]) + ':' + repr(type(self.a).__name__)\n"
                 "  try:\n"
                 "   raise KeyError(key)\n"
                 "  except KeyError:\n"
                 "   return [[(a, None)] for a in self.a]\n")
    _wms, _es = rc._exec_world_model(_str_code, {"available_actions": [6, 1]})
    _ps = _wms.candidate_plans(10) if _wms is not None else None
    ok &= _check("runtime_coder: str/type/KeyError available in WM sandbox",
                 _wms is not None and _ps == [[(6, None)], [(1, None)]])
    # and the sandbox must STILL block dangerous builtins
    _open_code = ("class WorldModel:\n"
                  " def __init__(self, o): open('/etc/hostname')\n")
    _wo, _eo = rc._exec_world_model(_open_code, {})
    ok &= _check("runtime_coder: open() still blocked in WM sandbox", _wo is None)

    # 5a2) Perception-first coder digest (BACKLOG R6+R8): brain.summarize.digest
    #      replaces the raw-grid `{obs}` block behind V21_CODER_DIGEST. Must name
    #      each component, losslessly recall the action->outcome table, stay
    #      bounded, and be deterministic; the runtime `observations` contract is
    #      unchanged (only the prompt text differs).
    from brain import summarize as _SUM
    _obs = {
        "level": 2,
        "available_actions": [1, 2, 6],
        # two separate 1-cell blobs of colour 3 on a 0-background 4x4 frame
        "frame": [[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 3, 0]],
        "transitions": [
            {"action": 1, "levels_completed": 0, "changed": False},
            {"action": 6, "levels_completed": 0, "changed": True},
        ],
    }
    _dg = _SUM.digest(_obs)
    ok &= _check("digest reports scene + both components",
                 "n_objects=2" in _dg and "#0" in _dg and "#1" in _dg)
    ok &= _check("digest losslessly recalls action->outcome table",
                 "a1 -> changed=False" in _dg and "a6 -> changed=True" in _dg)
    ok &= _check("digest is length-bounded on a large many-object frame",
                 len(_SUM.digest({"frame": [[(r + c) % 15 for c in range(64)]
                                            for r in range(64)],
                                  "available_actions": [1, 2, 3, 4, 5, 6, 7]},
                                 max_chars=2000)) <= 2000)
    ok &= _check("digest deterministic (same input -> same output)",
                 _SUM.digest(_obs) == _dg)
    ok &= _check("digest never raises on a bare ndarray-less frame / empty obs",
                 isinstance(_SUM.digest({}), str)
                 and isinstance(_SUM.digest([[0, 1], [1, 0]]), str))
    # _obs_block honours the env flag and falls back to raw fmt when OFF/on error
    os.environ["V21_CODER_DIGEST"] = "1"
    ok &= _check("_obs_block uses digest when V21_CODER_DIGEST=1",
                 "n_objects" in rc._obs_block(_obs))
    os.environ["V21_CODER_DIGEST"] = "0"
    ok &= _check("_obs_block uses raw fmt when flag OFF",
                 "n_objects" not in rc._obs_block(_obs))
    os.environ.pop("V21_CODER_DIGEST", None)

    # 5a3) Perception-first teacher feedback (R6+R8+R13): summarize.plan_failure_scene
    #      turns a stuck END frame + its delta-from-start into the note the Opus
    #      teacher's next iterative round reads (cadence_runner._replay_feedback).
    #      Must name the surviving objects, report the delta, stay bounded, and
    #      never raise on a degenerate/None frame.
    _pstart = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    _pend = [[0, 0, 0, 0], [0, 3, 0, 0], [0, 0, 0, 0], [0, 0, 5, 0]]  # a 5 appeared
    _fb = _SUM.plan_failure_scene(_pstart, _pend)
    ok &= _check("plan_failure_scene names scene + objects",
                 "final-frame scene" in _fb and "n_objects=2" in _fb and "color=5" in _fb)
    ok &= _check("plan_failure_scene reports delta vs start",
                 "delta vs level start" in _fb and "1 appeared" in _fb)
    ok &= _check("plan_failure_scene is length-bounded",
                 len(_SUM.plan_failure_scene(
                     [[0] * 64 for _ in range(64)],
                     [[(r + c) % 15 for c in range(64)] for r in range(64)],
                     max_chars=600)) <= 600)
    ok &= _check("plan_failure_scene never raises -> str",
                 isinstance(_SUM.plan_failure_scene(None, None), str)
                 and isinstance(_SUM.plan_failure_scene([[1]], None), str))

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

    # 10b-2) C2 persistent executable world model substrate: a tabular model
    #        reproduces recorded transitions BY CONSTRUCTION; the MDL pass
    #        collapses an identity table to a compact rule; persistence to
    #        brain/wm/<game>/ round-trips so a model learned this run is reused
    #        next run (the C2 "seeds a solve next run" mechanism).
    import tempfile as _tf, os as _os2
    wm_recs = [([[0, 0]], (1, None), [[0, 1]]), ([[0, 1]], (1, None), [[0, 2]])]
    wm_model = WM.build_tabular_model(wm_recs)
    wm_rep = WM.verify_model(lambda p, a: WM.predict_from_model(wm_model, p, a), wm_recs)
    ok &= _check("C2 tabular WM reproduces all recorded transitions",
                 wm_rep["accuracy"] == 1.0 and WM.is_trusted(wm_rep))
    ok &= _check("C2 tabular WM returns None on an unseen state",
                 WM.predict_from_model(wm_model, [[9, 9]], (1, None)) is None)
    #   identity records -> MDL refactor collapses table to a shorter rule that
    #   still reproduces every record
    _id_recs = [("x", "a", "x"), ("y", "b", "y")]
    _id_model = WM.mdl_refactor(WM.build_tabular_model(_id_recs))
    _id_rep = WM.verify_model(lambda p, a: WM.predict_from_model(_id_model, p, a), _id_recs)
    ok &= _check("C2 MDL refactor detects identity + still reproduces",
                 _id_model["kind"] == "identity" and _id_rep["accuracy"] == 1.0)
    #   persist to brain/wm/<game>/ and reload -> same predictions next 'run'
    _wmbase = _tf.mkdtemp()
    _gd = WM.wm_dir(_wmbase, "ls20")
    WM.save_model(_gd, wm_model)
    _reloaded = WM.load_model(_gd)
    _rep2 = WM.verify_model(lambda p, a: WM.predict_from_model(_reloaded, p, a), wm_recs)
    ok &= _check("C2 persisted WM reloads + reproduces (cross-run reuse)",
                 _reloaded is not None and _rep2["accuracy"] == 1.0 and _os2.path.isdir(_gd))
    ok &= _check("C2 load_model returns None when a game has no saved model",
                 WM.load_model(WM.wm_dir(_wmbase, "nope")) is None)

    # 10b-3) C2 cadence_runner WIRING (pure helpers, engine-free): _wm_persist
    #        builds+refactors+saves a per-game model; _wm_reuse loads it next run
    #        and verifies it still reproduces freshly-captured records (the cross-
    #        run reuse signal); the gate + empty/absent paths degrade safely.
    import cadence_runner as CR
    _wmbase2 = _tf.mkdtemp()
    _gd2 = WM.wm_dir(_wmbase2, "ls20")          # same layout _wm_game_dir produces
    #   env gate: OFF by default, ON when the flag is set
    ok &= _check("C2 wiring: _wm_enabled OFF by default",
                 not CR._wm_enabled({}))
    ok &= _check("C2 wiring: _wm_enabled ON with V21_WORLD_MODEL=1",
                 CR._wm_enabled({"V21_WORLD_MODEL": "1"}))
    #   no model persisted yet -> reuse returns None (nothing to verify)
    ok &= _check("C2 wiring: _wm_reuse None before any model saved",
                 CR._wm_reuse(_gd2, wm_recs) is None)
    #   persist this 'run' -> model saved on disk
    _saved = CR._wm_persist(_gd2, wm_recs)
    ok &= _check("C2 wiring: _wm_persist saves a model to disk",
                 _saved is not None and _os2.path.isfile(_os2.path.join(_gd2, WM.MODEL_FILENAME)))
    #   next 'run' -> _wm_reuse loads it and it still reproduces the same records
    _rerep = CR._wm_reuse(_gd2, wm_recs)
    ok &= _check("C2 wiring: _wm_reuse trusts the reloaded model next run",
                 _rerep is not None and WM.is_trusted(_rerep) and _rerep["accuracy"] == 1.0)
    #   a DIFFERENT (unseen) record set is not reproduced by the tabular model
    _other = [([[7, 7]], (2, None), [[7, 8]])]
    _rep_other = CR._wm_reuse(_gd2, _other)
    ok &= _check("C2 wiring: _wm_reuse flags an unseen transition (not trusted)",
                 _rep_other is not None and not WM.is_trusted(_rep_other))
    #   empty records degrade safely (no crash, no bogus model)
    ok &= _check("C2 wiring: _wm_persist/_wm_reuse safe on empty records",
                 CR._wm_persist(_gd2, []) is None and CR._wm_reuse(_gd2, []) is None)

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
    #   plan_in_model_macro (B3, ls20 L5-L6 frontier): a long "corridor" where the
    #   goal sits at position 20. Single-step BFS would need depth 20; a MACRO edge
    #   (action 1 repeated until the state stops changing) collapses the corridor
    #   into one push. State = pos; action 1 = +1 up to a wall at 20 (then no change,
    #   so the macro terminates). goal=1 means levels_completed>=1, reached at pos==20.
    def _mk_corridor():
        # mutable state carried in a 1-elem list so clone/play match the engine style
        def clone(s): return [s[0]]
        def play(s, step):
            aid, _ = step
            if aid == 1 and s[0] < 20:
                s[0] += 1
            return 1 if s[0] == 20 else 0   # levels_completed
        def hfn(s): return s[0]
        return clone, play, hfn
    _cl, _pl, _hf = _mk_corridor()
    mplan = PL.plan_in_model_macro([0], [1], _cl, _pl, _hf, goal=1,
                                   max_states=5000, max_macro=64)
    ok &= _check("planner.plan_in_model_macro collapses a 20-step corridor via macro",
                 mplan is not None and len(mplan) == 20
                 and all(step == (1, None) for step in mplan))
    #   and it returns None when the goal is unreachable (wall before goal)
    def _mk_blocked():
        def clone(s): return [s[0]]
        def play(s, step):
            aid, _ = step
            if aid == 1 and s[0] < 10:      # wall at 10, goal needs 20 -> impossible
                s[0] += 1
            return 1 if s[0] == 20 else 0
        def hfn(s): return s[0]
        return clone, play, hfn
    _cl2, _pl2, _hf2 = _mk_blocked()
    mplan2 = PL.plan_in_model_macro([0], [1], _cl2, _pl2, _hf2, goal=1,
                                    max_states=5000, max_macro=64)
    ok &= _check("planner.plan_in_model_macro returns None when goal unreachable",
                 mplan2 is None)
    #   plan_in_model_goexplore (C1, ls20 L5-L6 lever): a 6x6 corridor+turn maze.
    #   COARSE cell_fn bins the grid 2x so near-identical cells MERGE — proving the
    #   archive stays small (Go-Explore) while still reaching the deep goal.
    def _mk_maze():
        def clone(s): return dict(s)
        def play(s, step):
            a, _ = step
            if a == 2 and s["x"] < 5: s["x"] += 1
            elif a == 3 and s["x"] >= 5 and s["y"] < 5: s["y"] += 1
            return 1 if (s["x"] >= 5 and s["y"] >= 5) else 0
        def cell(s): return (s["x"], s["y"] // 2)   # coarse in y -> merges neighbours
        return clone, play, cell
    _cl3, _pl3, _cell = _mk_maze()
    gplan = PL.plan_in_model_goexplore({"x": 0, "y": 0}, [2, 3, 4], _cl3, _pl3, _cell,
                                       goal=1, max_states=5000, max_macro=16)
    _s = {"x": 0, "y": 0}; _lc = 0
    for st in (gplan or []): _lc = _pl3(_s, st)
    ok &= _check("planner.plan_in_model_goexplore solves a corridor+turn maze",
                 bool(gplan) and _lc >= 1)
    #   action_order guidance is honoured (tries the given ids first, still wins)
    gplan_o = PL.plan_in_model_goexplore({"x": 0, "y": 0}, [2, 3, 4], _cl3, _pl3, _cell,
                                         goal=1, max_states=5000, max_macro=16,
                                         action_order=[3, 2, 4])
    ok &= _check("plan_in_model_goexplore honours action_order and still solves",
                 bool(gplan_o))
    #   seed_plans priming: hand it the winning plan as a fragment -> returns it fast
    gplan_s = PL.plan_in_model_goexplore({"x": 0, "y": 0}, [2, 3, 4], _cl3, _pl3, _cell,
                                         goal=1, max_states=50, max_macro=16,
                                         seed_plans=[gplan])
    ok &= _check("plan_in_model_goexplore replays a winning seed fragment first",
                 gplan_s is not None and len(gplan_s) == len(gplan))
    #   unreachable goal -> None (wall at x=3, goal needs x>=5)
    def _pl_block(s, step):
        a, _ = step
        if a == 2 and s["x"] < 3: s["x"] += 1
        elif a == 3 and s["x"] >= 5 and s["y"] < 5: s["y"] += 1
        return 1 if (s["x"] >= 5 and s["y"] >= 5) else 0
    gplan_n = PL.plan_in_model_goexplore({"x": 0, "y": 0}, [2, 3], _cl3, _pl_block, _cell,
                                         goal=1, max_states=2000, max_macro=16)
    ok &= _check("plan_in_model_goexplore returns None when goal unreachable",
                 gplan_n is None)
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

    # 11) budget-reserve gate: solved+verified corpus levels must NOT trigger a
    #     fresh BFS re-search by default (they'd steal the wall budget), but unsolved
    #     walls always do, and V21_RESOLVE_SOLVED=1 re-enables the optimality hunt.
    import cadence_runner as cr
    ok &= _check("resolve gate: unsolved level always re-solved",
                 cr._should_resolve(False, env={}) is True)
    ok &= _check("resolve gate: solved level skipped by default",
                 cr._should_resolve(True, env={}) is False)
    ok &= _check("resolve gate: V21_RESOLVE_SOLVED=1 re-solves solved level",
                 cr._should_resolve(True, env={"V21_RESOLVE_SOLVED": "1"}) is True)
    ok &= _check("resolve gate: unsolved level solved even with flag on",
                 cr._should_resolve(False, env={"V21_RESOLVE_SOLVED": "1"}) is True)

    # 12) Epic C0 blackboard read/write wiring (pure helpers — engine replay is
    #     _verify in solve_game, not exercised here). Uses a temp blackboard dir.
    _BBDIR = os.path.join(tempfile.gettempdir(), "v21_test_bb")
    os.environ["V21_BLACKBOARD_DIR"] = _BBDIR
    import shutil as _sh
    _sh.rmtree(_BBDIR, ignore_errors=True)
    ok &= _check("bb gate: OFF by default", cr._bb_enabled(env={}) is False)
    ok &= _check("bb gate: ON with V21_BLACKBOARD=1",
                 cr._bb_enabled(env={"V21_BLACKBOARD": "1"}) is True)
    ok &= _check("bb open: None when gated off", cr._bb_open("zz00", env={}) is None)
    _bb = cr._bb_open("zz00", env={"V21_BLACKBOARD": "1"})
    ok &= _check("bb open: Blackboard when gated on", _bb is not None)
    # WRITE a verified win, then READ it back as a seed candidate
    _plan = [(6, {"x": 1, "y": 2}), (2, {}), (6, {"x": 3, "y": 4})]
    cr._bb_record_solution(_bb, 0, _plan, source="test")
    _seeds = cr._bb_seed_candidates(_bb, 0)
    ok &= _check("bb write->read: taught plan is a seed candidate",
                 any(len(s) == 3 for s in _seeds))
    _eff = _bb.data["action_effects"]
    ok &= _check("bb write: terminal action recorded a win",
                 _eff.get("6", {}).get("won", 0) >= 1)
    ok &= _check("bb write: every plan action recorded as tried",
                 _eff.get("2", {}).get("tried", 0) >= 1)
    ok &= _check("bb read: won action ranks before non-won in action_order",
                 _bb.action_order(0).index(6) < _bb.action_order(0).index(2))
    ok &= _check("bb read: seed candidates never crash on empty level",
                 isinstance(cr._bb_seed_candidates(_bb, 9), list))
    ok &= _check("bb record: no-op on None bb / empty plan",
                 cr._bb_record_solution(None, 0, _plan) is None
                 and cr._bb_record_solution(_bb, 0, []) is _bb)
    _bb.consolidate().save()
    _bb2 = cr._bb_open("zz00", env={"V21_BLACKBOARD": "1"})
    ok &= _check("bb persist: fragments survive save/reload",
                 len(_bb2.data["fragments"]) >= 1)

    # 12b) R7(a) workspace counterexamples — failed teacher plans persist as
    #      dead_ends and feed back as a 'do NOT repeat' note next run (env-gated).
    ok &= _check("counterex gate: OFF by default", cr._counterex_enabled(env={}) is False)
    ok &= _check("counterex gate: ON with flag",
                 cr._counterex_enabled(env={"V21_WORKSPACE_COUNTEREX": "1"}) is True)
    ok &= _check("counterex open: None when gated off", cr._counterex_open("cx00", env={}) is None)
    _cx = cr._counterex_open("cx00", env={"V21_WORKSPACE_COUNTEREX": "1"})
    ok &= _check("counterex open: Blackboard when gated on", _cx is not None)
    ok &= _check("counterex notes: empty when no dead_ends", cr._counterex_notes(_cx, 5) == "")
    _fail = [(2, {}), (3, {}), (6, {"x": 7, "y": 8})]
    cr._counterex_record(_cx, 5, _fail, source="opus")
    _n = cr._counterex_notes(_cx, 5)
    ok &= _check("counterex notes: names failed action seq",
                 "2,3,6" in _n and "do NOT repeat" in _n)
    ok &= _check("counterex record: no-op on None bb / empty plan",
                 cr._counterex_record(None, 5, _fail) is None
                 and cr._counterex_record(_cx, 5, []) is _cx)
    _cx2 = cr._counterex_open("cx00", env={"V21_WORKSPACE_COUNTEREX": "1"})
    ok &= _check("counterex persist: dead_end survives save/reload",
                 len(_cx2.data["dead_ends"]) >= 1)

    # 13) Epic C3 toddler — intuitive action orderer (corpus prior + online
    #     action_effects, frame-aware) behind the fixed order_actions interface.
    from brain.toddler import Toddler
    from brain.blackboard import ALL_ACTIONS as _ALL

    class _StubPrior:
        p = {"global": {"3": 0.9}, "per_game": {}, "actions": _ALL}

    # (a) with no lessons + no prior, ordering is a NO-OP (canonical order)
    ok &= _check("toddler: empty -> canonical order (no-op)",
                 Toddler().order_actions() == list(_ALL))
    # (b) with only a corpus prior, the toddler leans on it (favours action 3)
    ok &= _check("toddler: unseen -> corpus prior leads",
                 Toddler(prior=_StubPrior()).order_actions(game="zz00")[0] == 3)
    # (c) ONLINE override: an action observed to win/change beats a prior-favoured
    #     action that never changes anything (self-supervised from effects)
    _t = Toddler(blackboard=cr._bb_open("zz00", env={"V21_BLACKBOARD": "1"}),
                 prior=_StubPrior())
    for _ in range(3):
        _t.observe(5, changed=True, won=True)     # effective action
        _t.observe(3, changed=False, won=False)   # prior-favoured but inert
    _o = _t.order_actions(game="zz00")
    ok &= _check("toddler: learned-effective action overrides corpus prior",
                 _o.index(5) < _o.index(3))
    # (d) FRAME-conditioning: different frames prefer different actions
    _tf = Toddler()
    for _ in range(2):
        _tf.observe(2, changed=True, won=True, frame="frameX")
        _tf.observe(4, changed=True, won=True, frame="frameY")
    ok &= _check("toddler: frame X prefers its effective action",
                 _tf.order_actions(frame="frameX")[0] == 2)
    ok &= _check("toddler: frame Y prefers its effective action",
                 _tf.order_actions(frame="frameY")[0] == 4)
    # (e) runner wiring: gate + candidate restriction + graceful degrade
    ok &= _check("toddler gate: OFF by default", cr._toddler_enabled(env={}) is False)
    ok &= _check("toddler gate: ON with V21_TODDLER=1",
                 cr._toddler_enabled(env={"V21_TODDLER": "1"}) is True)
    ok &= _check("toddler order: None when gated off",
                 cr._toddler_order(_bb, "zz00", 0, [1, 2, 3], env={}) is None)
    ok &= _check("toddler order: None when no blackboard",
                 cr._toddler_order(None, "zz00", 0, [1, 2, 3],
                                   env={"V21_TODDLER": "1"}) is None)
    _to = cr._toddler_order(_bb, "zz00", 0, [1, 2, 3, 4, 5],
                            env={"V21_TODDLER": "1"})
    ok &= _check("toddler order: returns only in-`avail` actions",
                 isinstance(_to, list) and set(_to) <= {1, 2, 3, 4, 5})

    _sh.rmtree(_BBDIR, ignore_errors=True)
    os.environ.pop("V21_BLACKBOARD_DIR", None)

    # frontier gate: only re-rootable walls get the paid cloud teacher/WM budget.
    ok &= _check("reroot gate: L0 always reachable",
                 cr._wall_reachable(0, {}) is True)
    ok &= _check("reroot gate: frontier wall reachable (all prior solved)",
                 cr._wall_reachable(5, {0: [1], 1: [1], 2: [1], 3: [1], 4: [1]}) is True)
    ok &= _check("reroot gate: wall behind an unsolved earlier wall is gated",
                 cr._wall_reachable(6, {0: [1], 1: [1], 2: [1], 3: [1], 4: [1]}) is False)
    ok &= _check("reroot gate: gap in prior levels gates the wall",
                 cr._wall_reachable(3, {0: [1], 2: [1]}) is False)
    ok &= _check("reroot gate: solver.solutions chain also re-roots",
                 cr._wall_reachable(2, {}, {0: [1], 1: [1]}) is True)
    ok &= _check("reroot gate: empty corpus gates any wall>0",
                 cr._wall_reachable(1, {}) is False)

    # R8/B1 teacher click-target GROUNDING: the level-start frame's perception
    # centroids are folded into the Opus-teacher FIRST-round prompt so its clicks
    # hit real objects (run 152556Z: vc33 L4 round 1 clicked empty space — first
    # no-op at action index 0). Pure + bounded + gate default-OFF.
    _gframe = [[0, 0, 0, 0, 0, 0],
               [0, 3, 3, 0, 0, 0],
               [0, 3, 3, 0, 5, 0],
               [0, 0, 0, 0, 5, 0],
               [0, 0, 0, 0, 0, 0]]
    _gnote = cr._teacher_click_note(_gframe)
    ok &= _check("teacher grounding: note names click targets",
                 "click targets" in _gnote and "(" in _gnote and ")" in _gnote)
    #   the 3-blob centroid is (col2,row2); the 5-blob centroid is (col4,row3)
    ok &= _check("teacher grounding: note carries real object centroids",
                 "(2,2)" in _gnote and "(4,3)" in _gnote)
    ok &= _check("teacher grounding: bounded to max_chars",
                 len(cr._teacher_click_note(_gframe, max_chars=60)) <= 60)
    ok &= _check("teacher grounding: empty/None frame -> '' (no crash)",
                 cr._teacher_click_note([]) == "" and cr._teacher_click_note(None) == "")
    ok &= _check("teacher grounding: gate OFF by default",
                 cr._teacher_ground_enabled({}) is False)
    ok &= _check("teacher grounding: gate ON when V21_TEACHER_GROUND=1",
                 cr._teacher_ground_enabled({"V21_TEACHER_GROUND": "1"}) is True)

    print("\n" + ("ALL OFFLINE TESTS PASSED" if ok else "OFFLINE TESTS FAILED"))
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
