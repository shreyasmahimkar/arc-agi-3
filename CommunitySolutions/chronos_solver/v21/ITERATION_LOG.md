# v21 iteration log (append-only)

Each row = one coder cycle. Newest at top. The Opus 4.8 loop reads this before acting so it
builds on prior attempts instead of repeating them. Format:
`[UTC] item# — what changed — how verified — expected next-run effect`.

---

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

## Next up (from BACKLOG P0): item #3 — wire `runtime_coder` as cascade Stage-3.5 (call
## the local-LLM world-model writer on levels BFS/graph/blitz all fail, commit its
## shortest verified plan). Blockers/pending Mac-side steps: (a) item #2 blitz is coded +
## offline-verified — its live effect on ft09 L2–L5 / vc33 L4–L6 shows on the next Mac
## cadence; (b) item #1 evolve probe still needs `V21_EVOLVE_PROBE=1` on the Mac to
## PROMOTE; (c) ENV: Ollama returned HTTP 500 last run (model likely not pulled) — evolve
## stays skipped until the user fixes their local Ollama.
