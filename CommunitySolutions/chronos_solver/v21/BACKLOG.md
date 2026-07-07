# v21 autonomous-coder BACKLOG (worked top-down by the Opus 4.8 loop)

The between-rounds coder (Opus 4.8) picks the highest-priority **unblocked** item each
cycle, implements it in `v21/` only, compile+mock-tests it, commits, and logs the attempt
in `ITERATION_LOG.md`. Wall analysis (official baselines): solved = ls20 L0–L4, ft09 L0–L1,
vc33 L0–L3. **Walls to crack: ls20 L5–L6, ft09 L2–L5, vc33 L4–L6.**

Rules the coder follows: edit only under `v21/`; never touch `v19/`/`v20/`; always
`py_compile` + run `test_offline.py` before commit; never delete verified solutions
(append-shorter only); one item per cycle unless trivial.

## P0 — unblock the loop (do first)
1. **P2 config-aware evolve evaluator.** [CODED + offline-verified — live probe env-gated]
   `evolve.config_aware_eval_fn` scores corpus-floor + wall RHAE under the challenger's config;
   `cadence_runner._make_evolve_probe` applies `blitz_K`→BFS budget on the real engine.
   *Remaining:* run a Mac cadence with `V21_EVOLVE_PROBE=1` (+ enough `--bfs-timeout`) so a
   challenger that raises `blitz_K` actually solves a budget-gated wall and PROMOTES.
2. **Live blitz Stage-0.** [CODED + offline-verified — live effect on next Mac run]
   `blitz.py` (`blitz_solve` + `blitz_for_solver`) ports "race cheap wins on the fork first"
   (each action once, repeat-action×K, click-each-object); wired into `cadence_runner.solve_game`
   as Stage-0 for UNSOLVED levels only (verify + shortest-gated, env `V21_BLITZ`). Commit a962f8c.
   *Remaining:* a Mac cadence to confirm it commits a cheap win on ft09 L2–L5 / vc33 L4–L6.
3. **Wire `runtime_coder` as cascade Stage 3.5.** [CODED + offline-verified — live effect on
   next Mac run] `cadence_runner._runtime_coder_for_solver` builds obs from `_make_start_state`
   + one-step transitions and calls `RuntimeCoder.solve_level` with a fork-replay `try_plan`
   (`runtime_coder.replay_wins`); wired into `solve_game` as the LAST stage for UNSOLVED walls
   only, env `V21_RUNTIME_CODER` (default OFF). Verify + shortest-gate + exploit-refusal all
   preserved. Commit cc493d7. *Remaining:* a Mac cadence with `V21_RUNTIME_CODER=1` (local Qwen
   pulled) that solves a BFS/blitz-blocked wall end-to-end via generated code.

## P1 — crack the walls
4. **ls20 L5–L6 (LADDER / Go-Explore).** Variant re-root + TTRL suffix-BFS from the L4 end state.
   [PARTIAL — offline Go-Explore seed CODED] `blitz.blitz_macros` replays solved sibling-level
   plans (shortest winning prefix) as a Tier-0 wall seed, wired into `blitz_for_solver`. Commit
   9f69b20. *Remaining:* suffix-BFS from the L4 end state on the live engine; a Mac cadence to
   confirm a sibling macro (or seeded BFS) registers `levels_completed>=6`.
5. **ft09 L2–L5.** Investigate mechanics (these aren't blind like L0); deepen BFS/graph budget,
   add object-aware click targets. *Done when:* ≥1 of L2–L5 solved+verified.
6. **vc33 L4–L6.** Click-orchestration: better connected-component click-target selection in
   `graph_explore`. *Done when:* ≥1 of L4–L6 solved+verified.
   [PARTIAL — blitz click-REPEAT tier CODED] `blitz.blitz_solve` now repeats a single ACTION6
   coord ×K (shortest-gated), matching vc33's "hammer one component" endings (commit e049348).
   *Remaining:* a Mac cadence to confirm it commits a wall (L4–L6) whose win is a fixed-coord
   repeat; for mixed-coord walls, still needs better click-target ordering/selection.

## P2 — optimality & generalization
7. **ls20 L1 tighten 45→≤41** (only sub-1.0 solve). Masked/A* BFS or suffix trim. *Done when:* RHAE(L1)=1.0.
   [DONE] Corpus `solutions/ls20.json` L1 is 41 actions → RHAE(L1)=1.0 as of Mac run 20260706T194329Z.
8. **Trained intuition prior.** Replace corpus-frequency prior with a small policy net over
   frame features; keep the `order_actions` interface. *Done when:* held-out solve-rate improves.
9. **Cross-game macro retrieval (Stage 1b).** Use `intuition`/macro bank to seed BFS on a
   *similar* held-out game. *Done when:* a macro from one game solves a level of another.
   [PARTIAL — within-game macro replay CODED] `blitz.blitz_macros` replays a solved level's plan
   on another (wall) level of the SAME game (commit 9f69b20). *Remaining:* extend the macro source
   to `v21_macro_bank.json` / cross-GAME retrieval and seed BFS (not just full-plan replay).

## P3 — infra / submission
10. **Stall alarm.** Reporter pings if no cron_*.log in 8h.
11. **Kaggle offline notebook.** Bundle Qwen2.5-Coder as a dataset, `HF_HUB_OFFLINE=1`, embed
    agent+engine+cache; verify it runs network-off on a T4.
12. **Config-aware `MyAgent` load** of `champion.json` (blitz_K/action_order/heuristics).

# =====================================================================
# EPIC B — Cognitive ("brain") layer (game-general agent)
# Full design + rationale + research refs: BRAIN_ARCHITECTURE.md
# =====================================================================
# Goal: stack a cognitively-inspired layer (perception → executable world model
# → hypotheses → planner → memory → goal → consolidation) on top of the proven
# blitz→BFS→runtime_coder cascade, so the loop both cracks the remaining walls
# AND grows toward transfer to UNSEEN ARC-AGI-3 games. Spine = executable /
# program-synthesis world models (Rodionov 2026), NOT neural latent (that's B8).
# INVARIANT: the brain is additive — the proven cascade stays the fallback,
# every brain plan is still verify_solution + shortest-gated, each subsystem is
# wired live only behind its own env flag (default OFF) AFTER a Mac cadence
# proves it, and the offline submission guard is never disabled. All `brain/`
# code is pure/dependency-free at import and covered by test_offline.py.

## Epic B — phased build (each phase: green py_compile + test_offline, committed, env-gated OFF)
B1. **Perception scene-graph.** [DONE this session] `brain/perception.py`: connected-component
    objects (colour/size/bbox/centroid), frame-diff, and ACTION6 `click_targets` (one per
    component — fixes v19's per-colour-median clicks). 6 offline checks. *Remaining:* wire
    `click_targets` into `blitz`/BFS click selection behind `V21_BRAIN_PERCEPTION` on a Mac
    cadence (helps vc33 L4–L6).
B2. **Executable world-model persistence + verifier.** [PARTIAL — `brain/world_model.py` verifier
    core + template DONE] Generalise `runtime_coder` to a per-game model on disk
    (`brain/wm/<game>/`) that must reproduce recorded transitions (`verify_model`, `is_trusted`);
    add an MDL refactor pass. *Done when:* a persisted model reproduces a solved level's
    transitions offline and the loop reuses it next run.
B3. **MPC plan-executor.** [PARTIAL — `brain/planner.py` `plan_in_model` + `execute_and_verify`
    cores DONE] Wire to the real engine: plan inside the trusted model (unscored), execute with
    step-wise frame-mismatch abort; scored actions only on verified plans. *Done when:* a level is
    solved via model-planned + executor-verified actions on a Mac cadence.
B4. **Hypothesis manager (anti tunnel-vision).** [PARTIAL — `brain/hypotheses.py` `falsify` +
    `most_discriminating_action` cores DONE] Seed 2–3 competing WorldModels; spend scored actions
    on the most-discriminating move; falsify on mismatch. *Done when:* on a wall, discriminating
    exploration reaches a trusted model in fewer scored actions than greedy.
B5. **Goal induction.** [PARTIAL — `brain/goal.py` score-signal inducers DONE] Add frame-motif
    goal induction (perception motif + memory). *Done when:* induced goal drives a solve with no
    hand-coded goal.
B6. **Cross-game concept library.** [PARTIAL — `brain/memory.py` perceptual key + retrieval DONE]
    Persist macros + WM fragments + motifs to a bank keyed by perceptual signature; retrieve to
    seed a DIFFERENT game's search. *Done when:* a concept learned on one game solves a level of
    another (the Epic-B success metric). Subsumes/extends legacy #9.
B7. **Wake-sleep consolidation.** Extend `evolve`: replay solved trajectories, compress/refactor
    the library, re-distil the intuition prior. *Done when:* held-out solve-rate improves after a
    consolidation pass.
B8. **(Optional, far) Neural latent world model.** H-JEPA/Dreamer-style latent predictive model
    behind the same planner/goal interfaces. Blocked on a GPU training path; not offline-verifiable
    in 4h increments — do NOT start until B2–B7 are solid and a training route exists.

## Stop condition
All 20 levels across the 3 games solved + verified at RHAE 1.0 (or the highest reachable),
and the offline Kaggle notebook reproduces them. Then freeze and submit. Epic B has its OWN
success metric — held-out generalisation (a concept from one game solving another, B6) — pursued
in parallel without ever risking the verified corpus.
