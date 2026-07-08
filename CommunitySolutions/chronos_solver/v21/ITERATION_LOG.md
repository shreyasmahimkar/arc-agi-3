# v21 iteration log (append-only)

Each row = one coder cycle. Newest at top. The Opus 4.8 loop reads this before acting so it
builds on prior attempts instead of repeating them. Format:
`[UTC] item# — what changed — how verified — expected next-run effect`.

---

- [2026-07-08] **item C3 — TODDLER intuitive action orderer (brain/toddler.py, V21_TODDLER)** —
  HEALTH: runner HEALTHY/RUNNING — newest cron 120617Z last write ~75s before this check (14:01 UTC),
  no `cadence exit=` line, actively in ls20 BFS L5/L6 (16852 explored, 4495 unique @600s). `.cadence.lock`
  present = the live 120621Z run holding it (mtime 12:06 UTC), NOT stale. Last COMPLETE run 120621Z flat:
  ls20/ft09/vc33 all RHAE 1.000 on solved tiers, 0 walls newly cracked, evolve skipped (ollama backend).
  CODED: `brain/toddler.py::Toddler` — the C3 "toddler" behind the FIXED `order_actions(game, frame)`
  interface. Blends the corpus-frequency `IntuitionPrior` with the blackboard's ONLINE `action_effects`
  (win-weighted change-rate): unseen actions lean on the corpus prior, seen actions shift toward observed
  effectiveness (alpha=0.7). Frame-AWARE first (GPU-free) form: per-coarse-frame effect memory
  (`cell_key`) so a StochasticGoose/TRM net (R9/R11) can later drop in behind `_effect_score` with no
  interface change. Wired env-gated `V21_TODDLER` into `_goexplore_for_solver` via pure helpers
  `_toddler_enabled` / `_toddler_order` (Stage-3.45 action order; degrades to `bb.action_order`/canonical
  on off/None/failure — never invents actions, never raises). Exported `V21_TODDLER=1` in run_cadence.sh
  (per the R13 "coded-but-never-exported" lesson). VERIFIED: `py_compile` clean; `test_offline.py` GREEN
  with +10 new checks (no-op when empty; corpus-prior lead; learned-effect override of prior;
  frame-conditioning X vs Y; gate on/off; None on off/no-bb; avail restriction); `test_blackboard.py`
  GREEN. COMMIT BLOCKED: stale `.git/index.lock` (10:33 UTC) + `.git/HEAD.lock` (08:40 UTC) present and
  the Linux sandbox cannot unlink them ("Operation not permitted") — changes are on disk (toddler.py
  untracked, 3 files modified) awaiting a Mac-side commit. Expected next Mac run (locks cleared): with
  `V21_TODDLER=1` + `V21_GOEXPLORE=1`, ls20 L5 Go-Explore expands the effect-ranked action first instead
  of blind canonical order — watch for fewer states-to-first-new-cell and `levels_completed>=6`.

- [2026-07-08] **item C1 — cell-archive Go-Explore (ladder.go_explore, Stage-3.45 V21_GOEXPLORE)** —
  HEALTH: runner HEALTHY/RUNNING — newest cron 102145Z last write ~1.5m before this check (12:00 UTC),
  no `cadence exit=` line, actively in ls20 BFS L5/L6 then ft09 L2-L5 masked-space sweeps. `.cadence.lock`
  present but that's the live run holding it (not stale). Last COMPLETE run 102149Z flat: ls20/ft09/vc33
  all game_rhae 1.0, walls ls20 L5-L6 / ft09 L2-L5 / vc33 L4-L6 still UNSOLVED (no regression). CODE:
  upgraded `ladder.macro_bfs` → new pure `ladder.go_explore` cell-archive Go-Explore (one repr per COARSE
  cell, shortest-path-per-cell, return-to-promising-cell w/ over-visit cap, macro corridor-sweep that
  breadcrumbs every NEW cell with a patience stagnation-stop since coarse cells don't change each step).
  Wrapper `planner.plan_in_model_goexplore`; runner `_goexplore_for_solver` (cell_fn = `blackboard.cell_key`
  on status-bar-masked frame; steered by blackboard toddler `action_order`, primed by verified fragments)
  wired as Stage-3.45 for UNSOLVED walls; exported `V21_GOEXPLORE=1` + `V21_BLACKBOARD=1` in run_cadence.sh
  (R13 coded-but-never-exported lesson). VERIFIED green: py_compile (4 files) + `test_offline.py` (added 5
  checks: solve corridor+turn maze, action_order honoured, seed-fragment replay, unreachable→None) +
  `test_blackboard.py` + `test_ladder_mac.py --selftest`. verify+shortest-gated; corpus untouched.
  Commit <hash>. NEXT Mac run: expect a `GOEXPLORE solved` line if a coarse-cell archive cracks ls20 L5
  (`levels_completed>=6`); if flat, tune V21_GOEXPLORE_BINS / V21_PLANNER_STATES or seed from L4 end-state.
  NOTE: prior run 062043Z was SIGTERM-killed overrunning the 4h window — recommend BUDGET=300.

- [2026-07-08] **item C0 — wire the blackboard READ/WRITE into the cascade (V21_BLACKBOARD)** —
  HEALTH: runner RUNNING (new cadence 102145Z actively in BFS ls20 L5, last write 3m ago), but the
  PRIOR run (062043Z) overran the 4h window and was SIGTERM-killed (exit=143, launchd.err confirms
  caffeinate Terminated:15) — the exact "finish inside 4h" issue; recommend BUDGET=300. Walls
  unchanged: ls20 L5-L6, ft09 L2-L5, vc33 L4-L6 all UNSOLVED (last COMPLETE run 062047Z flat at
  RHAE 1.0). CODE: C0 substrate (brain/blackboard.py) was DONE but never wired; added pure helpers
  `_bb_enabled/_bb_open/_bb_record_solution/_bb_seed_candidates` in cadence_runner and wired
  solve_game: (READ) for still-UNSOLVED walls, replay the blackboard's verified fragments and keep
  the first that `_verify`s (the C0->C1 Go-Explore bridge, a sibling/prior lesson cracking a wall);
  (WRITE) every verified win teaches a fragment + per-action effects; consolidate().save() at pass
  end. All behind V21_BLACKBOARD (default OFF), verify+shortest-gated, corpus untouched. Also fixed
  a latent consolidate() crash (TypeError: unhashable dict) so it survives real ACTION6 click-plans
  — json-key dedup. VERIFY: py_compile green; test_offline.py green (+11 new C0 checks); 
  test_blackboard.py green. Commit: <hash>. Expected next Mac run with V21_BLACKBOARD=1: "BLACKBOARD
  seed solved" if a sibling fragment cracks a wall, and a growing brain/blackboard/<gid>.json.

- [2026-07-08] **item R6+R8 — perception-first coder digest (V21_CODER_DIGEST)** — CODE branch:
  newest COMPLETE Mac run 20260708T062047Z is the **4th consecutive FLAT** run (RHAE 1.0/1.0/1.0;
  ls20 L5/L6, ft09 L2-L5, vc33 L4-L6 all still UNSOLVED, improved=false on all 3 games). CRITICAL:
  062043Z is the FIRST run that actually includes BOTH the runner-wired brain-planner (7c6c743)
  AND the coerce_obs fix (ef38757), so the planner-fix is now VALIDATED and it did NOT crack a
  wall — the flatness is no longer excused by predating a fix. Log diagnosis: post-BFS
  wall-crackers only get real time on the SHALLOWEST unsolved wall per game (ls20 L5 got ~3.5min
  planner+~60s coder; ft09 L2 ~50s; vc33 L4 ~48s), while deeper walls (ls20 L6, ft09 L3-L5, vc33
  L5-L6) error out in <0.2s because `_make_start_state` can't chain past an unsolved prior level —
  correct behaviour for sequential games (crack L5 -> unlocks L6), so the real question is why the
  coder fails on the shallowest walls. Per R8 (arXiv:2512.21329) ~80% of coder failures on these
  are PERCEPTION, not reasoning, and today the coder is fed a RAW serialized grid (`_fmt` ->
  np.array2string). CHANGE: new pure `brain/summarize.py::digest()` builds a bounded, deterministic,
  perception-first scene description (B1 connected-component objects + click targets + a lossless
  action->outcome table from `transitions`); `runtime_coder._obs_block` swaps it into the `{obs}`
  prompt block behind `V21_CODER_DIGEST` (default OFF), fully guarded (any error -> raw `_fmt`), and
  the runtime `observations` contract is untouched (only prompt text changes). Verified: py_compile
  (runtime_coder/summarize/cadence_runner/test_offline) OK; `test_offline.py` GREEN with 7 new
  digest checks (names both components, lossless action->outcome recall, length-bounded on a 64x64
  frame, deterministic, never-raises on empty/bare-frame, env-flag on/off routing). No v19/solutions
  touched -> SKIP_LS20_GUARDRAIL=1. Commit cd0bb55. Expected next Mac run: default UNCHANGED; set
  `V21_CODER_DIGEST=1` (with `V21_RUNTIME_CODER=1` + local Qwen) so the coder reads objects-by-
  identity on ls20 L5 / ft09 L2 / vc33 L4 and can reference a scene it previously couldn't parse.

- [2026-07-08] **runtime_coder robustness — fix `frame[y, x]` candidate_plans crash (ft09
  wall)** — CODE branch: newest COMPLETE Mac run 20260708T043528Z (3rd FLAT: RHAE 1.0/1.0/1.0,
  all walls still UNSOLVED). NOTE: 043528Z started 00:35:24-04:00, exactly 68s BEFORE last
  cycle's brain-planner fix (7c6c743, committed 00:36:32-04:00) — so it predates the fix; the
  planner fix is still PENDING its first real test on the currently-running 062043Z run (log at
  ls20 L5 BFS as of this cycle, planner branch not yet reached). Did NOT re-touch the planner
  (already in place, awaiting validation). Instead fixed a distinct, verifiable defect visible in
  cron_043524Z.log line 34: `[coder] llm/candidate_plans failed: list indices must be integers or
  slices, not tuple` on ft09 L2 — the LLM WorldModel indexes the frame numpy-style (`frame[y,x]`)
  but the harness handed it a Python list, crashing the whole coder attempt. FIX: added
  `_coerce_obs()` to convert `observations['frame']` to `np.ndarray` before `WorldModel(...)` (numpy
  supports both `frame[y,x]` and `frame[y][x]`, so list-style code still works — strictly additive);
  updated WM_PROMPT to describe frame as a numpy array. Verified: `py_compile` OK on
  runtime_coder.py + cadence_runner.py + test_offline.py; `test_offline.py` green (36 checks, added
  "numpy frame[y,x] indexing does not crash"). No v19/solutions touched → SKIP_LS20_GUARDRAIL=1.
  Commit ef38757. Expected next-run effect: on ft09 L2–L5 (and any wall reaching runtime_coder),
  the LLM's candidate_plans no longer aborts on tuple-indexing, so its hypothesized plans actually
  get tested instead of falling back to the safety net alone.

- [2026-07-08] **item #4 (P1) — WIRE the Stage-3.4 brain-planner into the Mac runner
  (root-cause of the FLAT streak)** — Mac run 20260708T025330Z was the SECOND consecutive FLAT
  run (identical to 011103Z: RHAE 1.0/1.0/1.0, walls ls20 L5/L6 + ft09 L2-L5 + vc33 L4-L6 all
  still UNSOLVED). Root cause found: the Stage-3.4 macro-BFS planner committed last cycle (the
  diagnosed "macro REACH, not depth" fix for ls20's corridors) is env-gated `V21_BRAIN_PLANNER`
  default OFF and was NEVER exported in `run_cadence.sh` — so it never fired on the Mac (grep of
  cron_20260708T025326Z.log shows zero brain-planner lines; ls20 L5 just BFS-timed-out at 57k
  states then went straight to runtime_coder which failed in ~46s). Every other wall-cracker
  (V21_BLITZ / V21_EVOLVE_PROBE / V21_RUNTIME_CODER / V21_BRAIN_PERCEPTION) is defaulted ON in the
  runner; the planner was simply omitted. **Change:** added
  `export V21_BRAIN_PLANNER="${V21_BRAIN_PLANNER:-1}"` to `run_cadence.sh`'s wall-cracking block
  (before RUNTIME_CODER, so the pure white-box macro search gets first crack at the corridor before
  the LLM coder). No solver logic, no v19, no solutions touched — one-line runner wiring.
  *Verified:* `bash -n run_cadence.sh` OK; `python3 test_offline.py` green (planner already had 2
  offline checks: `plan_in_model_macro` collapses a 20-step corridor via macro edges); confirmed
  the plist ProgramArguments invokes this exact script. *Expected next Mac run:* on ls20 L5/L6 (and
  any UNSOLVED wall), after BFS times out the Stage-3.4 macro planner runs over the re-rooted engine
  and should register `levels_completed>=6` for ls20 if a corridor-collapsing macro path exists —
  breaking the FLAT streak. If still flat, the corridor needs the L4-end-state suffix-BFS seed
  (item #4 remaining) or the Opus teacher (R13). NOTE: deliberately did NOT set V21_RESOLVE_SOLVED=1
  (default OFF already reserves per-level budget for the walls by skipping re-BFS of L0-L4).

- [2026-07-08] **item #4 (P1) — Stage-3.4 brain-planner (Go-Explore / macro-BFS) for the
  ls20 L5-L6 corridor frontier** — Mac run 20260708T011103Z (budget=600s, qwen2.5-coder:7b) was
  FLAT vs 220316Z: same RHAE 1.0/1.0/1.0, no new wall (ls20 L5/L6, ft09 L2-L5, vc33 L4-L6 all
  still UNSOLVED). Diagnosis: BFS is exhausted on these walls, not budget-starved — ls20 L5 explored
  57k states/600s (117k/1200s prior) and still timed out; the corridors need macro reach, not depth.
  The 1200→600 budget drop (e663d4d) is intentional (faster CODE-branch signal). This cycle finished
  + verified the prior uncommitted brain-planner work: `brain/planner.plan_in_model_macro` (delegates
  to committed `ladder.macro_bfs` — corridor-collapsing macro edges + single-step precision) and
  `cadence_runner._brain_planner_for_solver` wiring it as Stage-3.4 over the white-box engine for
  UNSOLVED walls, env-gated `V21_BRAIN_PLANNER` (default OFF); also tightened `runtime_coder` WM
  prompt (explicit obs-dict schema, `.get` guidance). Added 2 offline checks (macro collapses a
  20-step corridor; returns None when unreachable). Verified: `py_compile` all changed .py +
  `test_offline.py` green (35 checks). No v19/solutions touched → committed with SKIP_LS20_GUARDRAIL=1.
  Expected next Mac run: with `V21_BRAIN_PLANNER=1` (+ `V21_RESOLVE_SOLVED` reserving budget), the
  macro-BFS should reach ls20 L5/L6 depth that plain BFS cannot, registering levels_completed>=6.

- [2026-07-07] **item #4 (P1) — reserve BFS budget for walls: stop re-solving corpus levels** —
  Mac run 20260707T220311Z (budget=1200s, model=qwen2.5-coder:7b-instruct-q4_K_M) IMPROVED over
  160000Z: ls20 L4 now SOLVED (43 actions, rhae 1.0; was a 600s timeout last run), also 1 shorter
  than the corpus L4 (44). This confirms the pending budget-raise (5d48138) + 7B model fix
  (0659668). BUT the run exposed the real ls20 L5/L6 blocker: `solve_game` ran a fresh full BFS on
  EVERY level incl. the already-solved L0–L4, which burned 3.4+25.4+163.8+448+1045.8 ≈ 1686s before
  L5 even started — so L5 (the wall) got ~0 budget. Since every solved ls20/ft09/vc33 level is
  already at RHAE 1.0 (BFS can't beat 1.0), re-deriving them each run is pure waste. Fix: added pure
  `cadence_runner._should_resolve(already_solved, env)` and gated the fresh `solver.solve_level`
  call on it — solved+verified corpus levels replay (verify only) and SKIP the re-BFS by default;
  unsolved walls always run BFS; `V21_RESOLVE_SOLVED=1` re-enables the optimality hunt (for a future
  sub-1.0 solve / R4 shortening). Corpus untouched (best stays the verified plan); one focused change
  in `v21/cadence_runner.py` only, no v19/v20 touched. Verified: `py_compile`
  (cadence_runner/test_offline) + `test_offline.py` GREEN with 4 new `_should_resolve` checks
  (unsolved→True, solved→False default, flag→True, unsolved+flag→True). Commit 572c4b5. ls20
  guardrail SKIPPED (`arc_agi` not importable in the Linux sandbox; change touches no v19 code or
  ls20 solutions). Expected next Mac run: L0–L4 replay in seconds, so ls20 BFS reaches L5 with the
  full per-level budget intact — first real shot at the L5 wall (pair with V21_RUNTIME_CODER=1 /
  Go-Explore item #4 for the depth L5 needs beyond plain BFS).

- [2026-07-07] **regression fix (P0) — hard wall-clock deadline on OllamaBackend.complete** —
  the 16:00Z (160000Z) Mac run STALLED: its cron log stops at `[coder] runtime backend=ollama`
  (ls20 L4, V21_RUNTIME_CODER on) and never advanced for 2.5h — no scorecard, no last_summary,
  ft09/vc33 never ran (a completion regression vs the 13:00Z run which finished all 3 games).
  Root cause: `urllib`'s `timeout=600` is a per-socket-op inactivity timeout, NOT a total
  deadline, so a swapping/OOM 7B model can hold the HTTP connection open indefinitely. Fix:
  extracted `_complete_raw` and ran it in a daemon watchdog thread with `join(deadline)`; on
  expiry `complete` RAISES (env `V21_OLLAMA_DEADLINE`, default 180s, 5s floor). RuntimeCoder
  already catches `llm.complete` exceptions → degrades to safety-net plans and the cadence
  continues to the next game. Edited only `v21/llm_backend.py` (+`test_offline.py`); no v19
  code or solutions touched. Verified: `py_compile` green + `test_offline.py` green with 2 new
  checks (hung backend raises; bounded <10s, not the 60s stall). Expected next Mac run: even if
  Ollama is unhealthy, the run COMPLETES and writes a scorecard for all 3 games again (corpus
  preserved) instead of hanging on ls20's coder step.

- [2026-07-07] **item B1 (Epic B) — wire perception into blitz** — connected the already-DONE
  `brain/perception.click_targets` scene-graph into the live click-selection path, env-gated OFF.
  New pure `blitz.merge_click_targets(scan_clicks, frame, use_perception, perception_fn=None)`:
  fuses v19's engine-scanned ACTION6 targets with perception's connected-component centroids,
  deduped by (x,y), scan-targets-first (proven default preserved), appending ONLY new component
  centroids the per-colour-median scan misses — the fix for vc33-style same-colour walls where
  several separate blobs of one colour collapse to a single median point landing on background.
  Called from `blitz_for_solver` behind `V21_BRAIN_PERCEPTION` (default OFF → default path byte-
  identical); any error falls through to scan-only clicks. Runs for UNSOLVED walls only, so zero
  cost on the solved corpus; every seeded click is still `verify_solution`+shortest-gated by the
  caller. Reason this cycle: all P0 items are CODED and only await a live Mac probe (blocked on
  env, not code); B1's remaining wiring is the highest-priority step advanceable purely offline
  and directly targets a live wall (vc33 L4–L6). Verified: `py_compile` (blitz/test_offline) +
  `test_offline.py` GREEN with 3 new checks (OFF→scan-only; ON→adds only new coord & dedups;
  ON→real perception_fn yields both blob coords). Commit 77b5e69 (ls20 guardrail SKIPPED —
  `arc_agi` not importable in the Linux sandbox; change touches no v19 code or solutions).
  Expected next Mac run: default behaviour UNCHANGED; set `V21_BRAIN_PERCEPTION=1` so on
  vc33 L4–L6 blitz probes a click ON each distinct component (not the median between them),
  giving the click-once / click-repeat tiers the right coords to hammer.

- [2026-07-07] **Epic B kickoff (brain layer)** — architected + scaffolded a cognitively-inspired
  layer on top of the blitz→BFS→runtime_coder cascade, grounded in ARC-AGI-3 world-model research
  (Rodionov 2026 executable world models; WorldCoder; DreamCoder/LILO; H-JEPA). New `BRAIN_
  ARCHITECTURE.md` (7 subsystems: perception, world_model, hypotheses, planner, memory, goal,
  consolidation — each mapped to existing code + a build phase). New pure/dependency-free `brain/`
  package: **perception.py FULLY implemented** (connected-component scene graph, frame-diff,
  ACTION6 `click_targets` — one target per component, fixing v19's per-colour-median clicks →
  directly aids vc33 L4–L6); interface cores for world_model (`verify_model`/`is_trusted` +
  MODEL_TEMPLATE), hypotheses (`falsify`, `most_discriminating_action` — anti tunnel-vision),
  planner (`plan_in_model` BFS-in-model + `execute_and_verify` MPC abort-on-mismatch), memory
  (`perceptual_key`/`retrieve` — game-agnostic cross-game retrieval), goal (`induce_from_scores`).
  BACKLOG gains **Epic B (B1–B8)** phased so the loop builds it; B1 done. Nothing wired into the
  default submission path — all env-gated OFF, additive to the proven cascade, corpus gate intact.
  Verified: `py_compile` (all 7 brain files + test_offline) + `test_offline.py` GREEN with 14 new
  brain checks (6 perception, 2 world_model, 2 hypotheses, 2 planner, 1 memory, 1 goal). Expected
  next Mac run: default behaviour UNCHANGED; the brain becomes live one env-flagged phase at a time
  as later cycles wire B1's `click_targets` into click selection, then B2+ onto the engine.

- [2026-07-07] **item #6 (P1)** — Blitz CLICK-REPEAT tier (vc33 walls). Added a 4th tier to
  pure `blitz.blitz_solve`: after the simple-action repeat tier, repeat a single ACTION6 click
  target ×K and keep the shortest winner (capped at `len(best)-1` so it only fires when it can
  beat any plan already found). Motivated by the vc33 verified corpus: its solutions end in long
  runs of ONE coord (L1 → 5× (0,44); L2 → 9× (46,56)), a pattern plain BFS times out on because
  it branches over every target at every depth, but that a fixed-coord line search cracks in ≤K
  probes/target. No adapter change — `blitz_for_solver` already passes the scanned click targets,
  so the tier is live on the next Mac cadence for UNSOLVED walls only (zero cost on the solved
  corpus); every plan is still `verify_solution` + shortest-gated by the caller. Verified:
  `py_compile` (blitz/cadence_runner/test_offline) + `test_offline.py` GREEN with 2 new checks
  (click-repeat picks shortest k on the RIGHT coord; prefers a shorter simple win over a longer
  click-repeat → exercises the cap). Commit e049348 (ls20 guardrail SKIPPED — `arc_agi` not
  importable in the Linux sandbox; change touches no v19 code or ls20 solutions). Expected next
  Mac run: default behavior unchanged on solved levels; on vc33 L4–L6 (and any click wall whose
  win is "hammer one component"), a cheap fixed-coord repeat can commit in seconds instead of BFS
  burning 180s; if the wall needs a mixed coord sequence, BFS runs as before.

- [2026-07-07] **item #4/#9 (P1/P2)** — Blitz MACRO-REPLAY tier (Go-Explore seed). Added pure
  `blitz.blitz_macros(start, target_level, macros, clone, play)`: replays each ALREADY-SOLVED
  sibling-level plan on a fresh fork of the wall level and returns the SHORTEST winning PREFIX
  (macros can overshoot the goal), or None — the cheapest Go-Explore seed there is, since a
  game's levels usually share mechanics so a sibling's verified solution can transfer verbatim
  with ZERO search. Wired into `blitz_for_solver` as Tier-0 (runs before the depth-1/repeat/click
  search): harvests macros from `solver.solutions` for every solved level ≠ the target, tries
  them first, falls through to `blitz_solve` if none win. Pure/offline-safe (injected closures);
  `blitz_solve` untouched (no regression to its 4 checks). Every returned plan is still routed
  through `verify_solution` + the shortest-plan corpus gate and `_refuses_exploit` by the caller —
  macros only PROPOSE. Runs only for UNSOLVED wall levels, so zero added cost on the solved corpus.
  Verified: `py_compile` (blitz/cadence_runner/test_offline) + `test_offline.py` GREEN with 4 new
  `blitz_macros` checks (sibling-plan win / overshoot→shortest-prefix trim / prefers-shortest /
  no-win→None). Commit 9f69b20 (ls20 guardrail SKIPPED — `arc_agi` not importable in the Linux
  sandbox; change touches no v19 code or ls20 solutions). Expected next Mac run: default behavior
  UNCHANGED on the solved corpus; for ft09 L2–L5 / vc33 L4–L6 / ls20 L5–L6, if a solved sibling's
  plan happens to win the wall, it commits in ~ms before BFS/blitz-search runs. Also NOTE: ls20 L1
  is already at 41 actions (RHAE 1.0) in the corpus from the last Mac run, so BACKLOG #7 is met.

- [2026-07-07] **item #3 (P0)** — Wire `runtime_coder` as cascade Stage-3.5. Added pure
  `runtime_coder.replay_wins(start, plan, clone, play, goal)` (fork-and-replay over injected
  closures; never mutates start) — the `try_plan_fn` contract the wiring needs. Added
  `cadence_runner._runtime_coder_for_solver(solver, lvl, llm, max_len)`: builds the level's
  TRUE chained start via `solver._make_start_state`, gathers observations (initial frame +
  one-step action→delta transitions), and hands them to `RuntimeCoder.solve_level` with a
  fork-replay `try_plan`; engine imports lazy (Mac-only). Wired into `solve_game` as the LAST
  cascade stage — runs ONLY when a wall level is still UNSOLVED after blitz + BFS. Backend built
  once via cached `_get_runtime_llm()` (local offline `get_backend`, honors `V21_RUNTIME_LLM`).
  OFF by default (loads a local model / adds wall-clock); opt-in via `V21_RUNTIME_CODER=1`
  (`V21_RUNTIME_MAXLEN`=200). Any error falls through; every returned plan still passes
  `verify_solution` + the shortest-plan corpus gate, and `_refuses_exploit` blocks null-coord
  ACTION6. Verified: `py_compile` (runtime_coder/cadence_runner/test_offline) + `test_offline.py`
  GREEN with 4 new `replay_wins` checks (win / short-plan-no-win / empty→False / no-mutation).
  Commit cc493d7 (ls20 guardrail SKIPPED — `arc_agi` not importable in the Linux sandbox; change
  touches no v19 code or ls20 solutions). Expected next Mac run: default behavior UNCHANGED; set
  `V21_RUNTIME_CODER=1` (with the local Qwen pulled) so BFS/blitz-blocked walls (ft09 L2–L5 /
  vc33 L4–L6 / ls20 L5–L6) get a generated-WorldModel plan committed if one wins.

- [2026-07-06] **item #2 (P0)** — Blitz Stage-0 pre-pass. New `blitz.py`: pure
  `blitz_solve(start, target_level, simple_actions, click_targets, clone, play, repeat_K)`
  that races the cheap shallow wins on a fork — (1) each simple action once, (2) each
  effective ACTION6 click once (both length-1), (3) repeat a single action ×K (shortest k).
  Adapter `blitz_for_solver(solver, lvl)` binds it to v19 `BFSSolver` (chained start via
  `_make_start_state`, click targets via `_scan_actions`; engine imports lazy so `import
  blitz` stays dep-free). Wired into `cadence_runner.solve_game` as Stage-0, run ONLY for
  UNSOLVED (wall) levels (zero added cost on the solved corpus), env `V21_BLITZ` (default on,
  `V21_BLITZ_K` repeat cap=200); any error falls through to BFS, and every blitz plan still
  passes `verify_solution` + the shortest-plan gate before it can enter the corpus. Verified:
  `py_compile` (blitz/cadence_runner/test_offline) + `test_offline.py` GREEN with 4 new
  checks (single-action, repeat-k shortest, click-target, no-fabrication→None). Commit
  a962f8c. NOTE: the v19 ls20 pre-commit benchmark was SKIPPED (`SKIP_LS20_GUARDRAIL=1`)
  because `arc_agi` isn't importable in the Linux cadence sandbox — the guard can't run
  here; the change touches no v19 code or ls20 solutions. Expected next Mac run: for
  ft09 L2–L5 / vc33 L4–L6, a cheap depth-1/repeat/click win (if one exists) commits in ~s
  instead of BFS burning 180s and returning nothing; if none exists, BFS runs as before.

- [2026-07-06] **item #1 (P0)** — config-aware evolve evaluator. Added
  `evolve.config_aware_eval_fn(corpus_rhae, walls_by_game, probe_fn)`: score = corpus
  floor + (1-floor)·mean(wall RHAE under the challenger's config), so a challenger that
  cracks a budget-gated wall scores strictly higher and can PROMOTE. Wired into
  `cadence_runner` with `_make_evolve_probe` (applies `blitz_K`→BFS `max_states` on the
  real engine); probe is opt-in via env `V21_EVOLVE_PROBE=1` (live rollout multiplies
  cadence wall-clock) and safely degrades to the corpus floor when off. Verified:
  py_compile + `test_offline.py` green with 4 new checks (floor fallback, config-
  sensitivity, floor-monotonicity, end-to-end generalization-gated promotion); tests now
  write history to a temp file so they don't pollute `evolution_history.jsonl`. Champion
  untouched (still v0). Expected next run: same behavior by default; set
  `V21_EVOLVE_PROBE=1` (with adequate `--bfs-timeout` budget) to let evolve actually
  promote a wall-cracking challenger.

- [2026-07-06] **bugfix** — added `solver.load()` + `solver.solutions[lvl]` chaining in
  `cadence_runner.solve_game`; fixed Ollama 404 (model-presence check + clear error).
  Verified: py_compile + inspected `combined_agent.BFSSolver` API. Effect: real BFS now runs
  on the Mac (was failing instantly with NoneType). — champion still v0.
- [2026-07-06] **scaffold** — built the loop: `runtime_coder` (on-the-fly WM writer, sandbox
  exec, tested), `evolve` (champion/challenger), `intuition` (corpus prior), `llm_backend`
  (ollama/hf/openai/mock). Verified: offline end-to-end test (ft09/ls20-like solves via mock).
- [2026-07-06] **seed** — corpus seeded from v20 (ls20 L0–L4, ft09 L0–L1, vc33 L0–L3); official
  baselines wired; RHAE floor measured (ls20 0.966, ft09 1.0, vc33 1.0; 9 wall levels).

## Next up: all of P0 (#1 evolve probe, #2 blitz, #3 runtime_coder) is now CODED +
## offline-verified — the loop is fully cascaded (blitz → BFS → runtime_coder) and only
## needs live Mac cadences (with the right env flags) to prove wall-cracking. The next
## between-rounds cycle should move to P2 optimality (#7: ls20 L1 45→≤41 suffix-trim — a
## pure-offline-safe win) or P1 wall analysis (#4/#5/#6), building on any new Mac scorecard.
## Blockers/pending Mac-side steps: (a) item #3 runtime_coder: set `V21_RUNTIME_CODER=1`
## with a local Qwen pulled to attack BFS/blitz-blocked walls; (b) item #2 blitz: live
## effect on ft09 L2–L5 / vc33 L4–L6 shows on the next Mac cadence; (c) item #1 evolve
## probe still needs `V21_EVOLVE_PROBE=1` (+ adequate `--bfs-timeout`) to PROMOTE; (d) ENV:
## Ollama returned HTTP 500 last run (model likely not pulled) — evolve stays skipped until
## the user pulls/fixes their local Ollama (or points backend at the local Qwen HF model).
