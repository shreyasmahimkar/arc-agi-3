# v21 iteration log (append-only)

Each row = one coder cycle. Newest at top. The Opus 4.8 loop reads this before acting so it
builds on prior attempts instead of repeating them. Format:
`[UTC] item# — what changed — how verified — expected next-run effect`.

---

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
