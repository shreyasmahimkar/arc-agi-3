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
   [BUDGET-RESERVE DONE 2026-07-07] `cadence_runner._should_resolve` now skips re-BFS of
   already-solved+verified corpus levels (env `V21_RESOLVE_SOLVED`, default OFF), so L0–L4 replay
   in seconds and the ls20 BFS reaches L5 with the full per-level budget instead of ~0 (run
   220311Z burned ~1686s re-solving L0–L4 before L5 even started). See ITERATION_LOG for commit.
   *Still needed for the win:* L5 depth exceeds plain BFS — pair the reserved budget with
   `V21_RUNTIME_CODER=1` (Go-Explore/coder) or a seeded suffix-BFS from the L4 end state.
   [GO-EXPLORE STAGE-3.4 DONE 2026-07-08] `brain/planner.plan_in_model_macro` (→ committed
   `ladder.macro_bfs`: corridor-collapsing macro edges + single-step precision) wired as
   `cadence_runner._brain_planner_for_solver` Stage-3.4 over the white-box engine for UNSOLVED
   walls, env-gated `V21_BRAIN_PLANNER` (default OFF), verify+shortest-gated. +2 offline checks.
   Confirmed by run 011103Z: plain BFS timed out on ls20 L5 (57k states/600s) — macro reach is
   the missing ingredient, not depth.
   [RUNNER-WIRED 2026-07-08] Root-caused the 2nd FLAT run (025330Z): the Stage-3.4 planner was
   coded but `V21_BRAIN_PLANNER` was never exported in `run_cadence.sh`, so it never fired on the
   Mac. Added `export V21_BRAIN_PLANNER=1` to the runner's wall-cracking block (before RUNTIME_CODER).
   Budget-reserve (skip-re-BFS of solved L0-L4) is already active by default, so V21_RESOLVE_SOLVED
   is intentionally left OFF. *Still needed for the win:* the NEXT Mac cadence should now actually
   run the macro planner on ls20 L5/L6 and register `levels_completed>=6`; if still flat, add the
   L4-end-state suffix-BFS seed or the R13 Opus teacher.
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
   [ELEVATED by user steer 2026-07-08 → see R9 (TRM/HRM tiny recursive scorer) and R11 (Tufa
   StochasticGoose CNN frame-change head) for the concrete open-model implementations.]
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
    component — fixes v19's per-colour-median clicks). 6 offline checks. [WIRED — env-gated,
    offline-verified] `blitz.merge_click_targets` fuses perception centroids with the scanned
    clicks (scan-first, deduped) in `blitz_for_solver` behind `V21_BRAIN_PERCEPTION` (default OFF);
    +3 offline checks; commit 77b5e69. *Remaining:* a Mac cadence with `V21_BRAIN_PERCEPTION=1` to
    confirm per-component click coords crack a vc33 L4–L6 wall the median-scan misses.
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

# =====================================================================
# RESEARCH FEED — R1–R5 integrated 2026-07-07; R6 added (RESEARCH-2); R7–R8 added (RESEARCH-3)
# Latest ARC-AGI-3 literature scanned (arXiv / ARC Prize / Kaggle). Each item is
# ADDITIVE, env-gated when shipped, and slots into the proven cascade + brain.
# These VALIDATE the current direction (executable world models + explore-first)
# and add concrete, prioritized mechanisms. Do not duplicate B2–B6 — these refine
# them with specific techniques from the papers.
# =====================================================================
R1. **Explore→Verify→Plan with a belief-entropy COMMIT GATE** (AERA, arXiv:2605.25931,
    "Explore Before You Solve"). Strongest single finding: what enables non-zero RHAE on
    hidden-rule games is maintaining explicit world-model HYPOTHESES and *gating the switch from
    exploration to planning on a proxy for belief entropy* (uncertainty over models). Also gives a
    concrete budget heuristic: spend ≈40% of the human baseline on exploration before committing.
    Their public-set taxonomy explicitly places our walls — ft09 = blind-ACTION6 reflex; ls20 =
    budget-constrained repeated-action (50–200 steps); vc33 = probe-then-ACTION6. *Action:* sharpen
    B4 into a real commit gate — in `brain/hypotheses.py`, don't hand off to the planner until the
    surviving-hypothesis set collapses (entropy proxy below threshold) OR the ≈40%-of-baseline
    explore budget is spent; expose the budget as an env knob. Pure/offline-testable (entropy proxy
    + gate over injected hypotheses). *Done when:* on a wall, the gate reaches a single trusted model
    in fewer scored actions than greedy BFS. NOTE: their 55-game code-track entry is "BFS + offline
    pre-solve cache" at RHAE 0.30 — i.e. OUR architecture — good external validation.
R2. **Verify → MDL-refactor → plan-through-model** (Executable World Models, arXiv:2605.05138).
    The verifier-driven executable-WM loop (verify against observations → refactor toward SIMPLER
    abstractions as an MDL proxy → plan through the model before acting) is exactly B2/B3; the paper
    reports 15/25 games solved at RHAE 58% with a strong coder model. Two concrete adds: (a)
    prioritize the **MDL refactor pass** in `brain/world_model.py` (shorter program that still
    reproduces all recorded transitions → better generalization), and (b) the paper flags that WM
    quality "varies substantially across runs" → add **best-of-N / multi-hypothesis** WM synthesis
    (ties into R1's competing hypotheses). *Action:* bump B2's MDL-refactor to the next CODE-branch
    item once a Mac run gives runtime_coder live signal; keep it env-gated + verifier-gated.
R3. **Graph-based level explorer + frame processor** (arXiv:2512.24156). Method = Frame Processor
    (image segmentation → status-bar detection & MASKING → priority-based action grouping → STATE
    HASHING) feeding a Level Graph Explorer (explicit state-graph, action-selection strategy,
    FRONTIER MANAGEMENT). We already have transient/status-bar masking and connected-component
    perception (B1); the new, directly-usable pieces are **state hashing for dedup** and **frontier
    management** to make BFS explore unique states instead of re-expanding, plus **priority-based
    action grouping** to order ACTION6 targets. *Action:* fold state-hash dedup + priority action
    grouping into the BFS/planner for vc33 L4–L6 (item #6); offline-testable on a mock state graph.
R4. **Speed–Depth / RHAE is quadratic** (2605.25931, §3). RHAE = (human/AI actions)², so a solve
    that uses 2× human actions earns only 25% credit; budget-constrained repeated-action wins
    (ls20-style) are penalized hard for length. *Action:* keep the shortest-plan gate strict and add
    a suffix-trim/A* optimality pass for repeat-heavy solves (extends legacy #7/#8) — a solved-but-
    long wall should be revisited to SHORTEN, not just left at RHAE<1.
R5. **Test-time training on a tiny model** (NVARC 2025 winner: Qwen-4B + TTT + synthetic data,
    24% on ARC-AGI-2, ARC Prize 2025 report arXiv:2601.10904; TRM test-time adaptation
    arXiv:2511.02886). This is the static-grid (v1/v2) recipe; less direct for interactive v3 but
    relevant to a future learned intuition prior (item #8 / B8). *Action:* park as a B8 reference —
    do NOT start (needs a GPU training route); revisit only after B2–B4 are live.
R6. **Orchestrator + subagents with COMPRESSED-summary context control** (Symbolica *Arcgentica*,
    open-source `symbolica-ai/arcgentica` + `symbolica-ai/ARC-AGI-3-Agents`; blog
    https://www.symbolica.ai/blog/arc-agi-3 ; scores 36.08% on the 25-game public set = 113/182
    levels, 7/25 games, and solves all 3 public envs incl. ours). Integrated 2026-07-07 RESEARCH-2
    cycle — NEW vs R1–R5. Key mechanism: a top-level orchestrator never touches the environment; it
    delegates to specialised subagents that return **compressed textual summaries**, which caps
    context growth so a long exploration transcript never blows the coder's context window. This is
    directly relevant to our `runtime_coder` Stage-3.5 and the brain planner on LONG walls (ls20
    L5–L6, ft09 L2–L5): today we feed raw obs/one-step transitions into the coder model, which grows
    unbounded as exploration deepens. *Action:* add a pure `brain/summarize.py` (or a helper in
    `runtime_coder`) that compresses recorded transitions into a fixed-size structured digest
    (object/scene deltas + tried-action → outcome table) BEFORE the coder prompt, env-gated
    `V21_CODER_DIGEST` (default OFF); slots between obs-build and `OllamaBackend.complete`. Keep it
    additive + offline-testable (digest is deterministic over a mock transition log; assert bounded
    length + lossless action→outcome recall). *Done when:* a Mac cadence shows the coder step
    completing (not stalling/OOM) on a deep wall where it previously timed out — ties into the
    160000Z stall root-cause. Do NOT adopt their multi-agent SDK wholesale (heavy dep, network);
    port only the summary-compression idea into the existing single-coder path.
R7. **Workspace optimization: evolve the persistent substrate, not the weights** (NVIDIA/Technion
    *DREAMTEAM*, arXiv:2605.09650, "Workspace Optimization: How to Train Your Agent"; abs
    https://arxiv.org/abs/2605.09650 ). Integrated 2026-07-07 RESEARCH-3 cycle — NEW vs R1–R6, and
    currently the highest public-set score we've seen: **38.4% on the 25-game set (up from Symbolica's
    36.08%) using 31% FEWER environment actions per game.** Core idea maps almost 1:1 onto our loop:
    since a frozen frontier/coder model can't be weight-trained, treat the agent's *workspace* (the
    structured files it reads/writes/tests) as the trainable object, mirroring weight-space training —
    **artifacts↔parameters, evidence↔data, counterexamples↔losses, textual feedback↔gradients.** Our
    corpus/`champion.json`/`intuition_prior.json`/macro-bank ARE the artifacts, the scorecard +
    recorded transitions ARE the evidence, the regression gate + UNSOLVED walls ARE the
    counterexamples, and ITERATION_LOG/cron notes ARE the textual-feedback "gradient." *Action (two
    concrete, additive, env-gated adds):* (a) after each Mac run, `evolve` writes failed-wall
    transcripts as **counterexample artifacts** (`brain/wm/<game>/counterexamples.jsonl`: the wall,
    the tried plan, why it failed) that the NEXT cycle's `runtime_coder`/planner reads as
    negative-constraint context ("don't repeat these dead ends") — env `V21_WORKSPACE_COUNTEREX`
    (default OFF); pure + offline-testable (assert the coder prompt excludes a recorded dead-end
    action). (b) Adopt their **action-frugality objective**: in the shortest-plan/evolve scoring,
    tie-break challengers by env-actions-per-solve so the workspace evolves toward the 31%-fewer-
    actions regime (extends R4's quadratic-RHAE pressure). Slots into `evolve` + `runtime_coder`
    Stage-3.5 + brain `memory`; do NOT pull the DREAMTEAM multi-agent harness (network/heavy) — port
    only the workspace-as-trainable-substrate discipline. *Done when:* a Mac cadence shows a wall the
    coder previously failed is not re-tried down the same dead end (counterexample recall), or a
    challenger promotes on equal-RHAE-but-fewer-actions.
R8. **Perception is the real bottleneck — feed the coder a symbolic scene description, not raw grids**
    (CMU/UMich/UCSD/UIUC, arXiv:2512.21329, "Your Reasoning Benchmark May Not Test Reasoning:
    Revealing Perception Bottleneck in Abstract Reasoning Benchmarks"; abs
    https://arxiv.org/abs/2512.21329 ). Integrated 2026-07-07 RESEARCH-3 cycle — NEW vs R1–R7.
    Controlled two-stage experiments show **~80% of VLM ARC failures are PERCEPTION errors, not
    reasoning**, and inserting a dedicated perception→natural-language stage before the reasoning
    model gives +11–13pp (up to 2.5× on Mini-ARC). Strong external VALIDATION of our design choice:
    v21 already parses frames with exact connected-component perception (B1) instead of a VLM, so we
    *sidestep* the perception bottleneck entirely — the paper argues that's exactly where the wins
    are. *Action:* make R6's `V21_CODER_DIGEST` digest **perception-first** — the digest the coder
    reads should be a structured scene description built from B1's `perception.py` objects
    (per-object colour/size/bbox/centroid + frame-diff deltas + tried-action→object-change table),
    NOT a serialized raw grid (the paper shows serialized grids are the hard format even for humans).
    This is a refinement of R6 (same env flag `V21_CODER_DIGEST`), not a new subsystem: swap the
    digest's representation from raw-grid to perception-object schema. Pure + offline-testable (assert
    the digest names each component and its post-action delta over a mock transition log). *Done when:*
    on a wall, the perception-object digest lets the coder reference objects by identity and the
    coder step yields a valid plan where the raw-grid digest did not.

# ---------------------------------------------------------------------
# STRATEGIC STEER (2026-07-08, user): raw BFS provably cannot crack ls20 L5/L6
# (117k states, 30k unique, still timed out at 1200s) — budget lowered 1200->600
# so passes turn over faster. MAIN FOCUS moves OFF search depth and ONTO: latest
# world-model research, small/intuitive OPEN models (TRM-style), Tufa Labs'
# interactive-v3 work, Opus-4.8 as a teacher on the network-enabled cadence box,
# and (long-horizon) an ensemble that combines them. Items R9–R14. All additive +
# env-gated; the offline submission corpus is never risked.
# ---------------------------------------------------------------------
R9. **Tiny Recursive Models (TRM) / HRM as the learned intuition prior** (elevates item #8; the
    user's "small intuitive open models" steer). TRM = 7M-param single tiny net that recurses on a
    latent "scratchpad" and backprops through ALL recursive steps; 45% ARC-AGI-1 / 8% ARC-AGI-2,
    beating DeepSeek-R1/o3-mini/Gemini-2.5-Pro at <0.01% of their params (Jolicoeur-Martineau,
    "Less is More", arXiv:2510.04871). Predecessor HRM (27M, 40.3% ARC-AGI-1). OPEN WEIGHTS on HF:
    `wtfmahe/Samsung-TRM`, `SamsungSAILMontreal/TinyRecursiveModels`, `domus-magna/trm-repro`.
    Refinements: TTT-of-TRM (arXiv:2511.02886), Mamba-2 hybrid (arXiv:2602.12078), identity-cond +
    test-time compute (arXiv:2512.11847). *Action:* replace the corpus-frequency `intuition_prior`
    with a TRM-style tiny recursive scorer behind the existing `order_actions` interface, env
    `V21_TRM_PRIOR` (OFF). CAVEAT: TRM is a STATIC-grid solver — for interactive v3 reframe it as a
    per-frame "rank next action / predict frame-delta" scorer, not a puzzle solver. *Done when:* the
    TRM prior beats the frequency prior on held-out action ordering offline. **BLOCKER: GPU
    training/TTT route** (not offline-verifiable in a 4h sandbox); bundle a quantized checkpoint for
    offline inference once trained.
R10. **Mine the now-PUBLIC ARC-AGI-3 reference code** (validates R2/R3/B2, gives real impls to port).
    (a) Executable World Models baseline `github.com/astroseger/arc-3-agents-baseline1` (Rodionov
    2605.05138: coder builds+verifies+MDL-refactors a Python WM, plans through it; 15/25 games, RHAE
    58%) → seeds B2/B3 authoring loop. (b) Graph explorer `github.com/dolphin-in-a-coma/arc-agi-3-just-explore`
    (frame processor + state hashing + frontier mgmt) → seeds R3/#6. (c) Interactive agent
    `github.com/ssppsy/arc-agi-3`. (d) Official API/toolkit `github.com/arcprize/arc-agi`. *Action:*
    port the WM-authoring loop (a) and frontier-managed state-graph (b) into `brain/world_model.py`/
    planner + BFS, pure/offline + env-gated; do NOT vendor network/agent-SDK deps. *Done when:* a
    ported WM-authoring or frontier-dedup step reproduces a solved level offline.
R11. **Tufa Labs "StochasticGoose" — CNN+RL frame-change predictor** (won the ARC-AGI-3 Agent
    Preview at 12.58%, 18 levels: a 4-layer conv over 64×64 frames predicting which actions cause a
    frame change; ARC Prize 2025 report arXiv:2601.10904 + Tufa preview writeup). INTERACTIVE-v3-
    NATIVE and the cleanest attack on **ft09** (blind-ACTION6 reflex): a small conv "will this action
    change the frame?" head is exactly ft09's missing intuition. Ties to R9 (both are the learned
    prior). *Action:* prototype a tiny frame-change-predictor head (numpy inference, bundled weights)
    that reorders/gates ACTION6 candidates, env `V21_FRAMECHANGE_PRIOR` (OFF). *Done when:* it cuts
    scored actions on ft09 vs the blind scan offline. **BLOCKER: GPU training route** (as R9).
R12. **Tufa Labs LADDER + TTRL — recursive variant self-improvement** (arXiv:2503.00735). Generate
    progressively SIMPLER variants of a hard level, solve those first, RL up to the hard one on a
    verifiable reward; TTRL adds test-time RL. This is the named/cited method behind item #4's ls20
    "variant re-root / Go-Explore", and it strengthens `evolve`/consolidation (B7). *Action:* fold a
    LADDER-style variant-decomposition seed into `blitz.blitz_macros` / the evolve loop for ls20
    L5–L6 — solve a shortened/rooted sub-problem, replay its winning prefix as a Go-Explore seed.
    Pure/offline-testable on a mock variant ladder. *Done when:* a variant-seed prefix registers
    progress on ls20 L5 that cold BFS does not.
R13. **Opus-4.8 as a TEACHER/coder on the network-enabled cadence box** (user obs: Opus-4.8 solves
    ls20 levels as-is). The Mac cadence already runs `--allow-network`; add an env-gated
    `V21_LLM_BACKEND=claude` coder behind the existing `llm_backend` interface (external network +
    key ONLY on the Mac) so `runtime_coder`/`evolve` can use a frontier model to actually crack a
    wall, then DISTILL the `verify_solution`-passed plan into the offline corpus. INVARIANT: the
    teacher runs only during Mac *solving*; the committed corpus is a model-free verified action list,
    so the OFFLINE Kaggle guard is untouched (Kaggle never calls the teacher). Also capture Opus's
    ls20 reasoning traces as macros / to distil the intuition prior (feeds R9). *Done when:* an
    Opus-authored verified plan solves ls20 L5 or L6 and lands shortest-gated in the corpus.
    **BLOCKER: Claude/Opus API key + external network on the Mac** (localhost-only today) — user action.
R14. **(Long-horizon integration target) The ensemble the user is imagining: open world model +
    massively-parallel multi-agent + VLM + neuro-symbolic brain-mimicking nets, combined to crack
    ls20 L5/L6.** This is the north star, and much of it already has homes — don't build it as one
    monolith; assemble it from the env-gated pieces: (i) **open world model** = R2/R10/B2 executable
    WM; (ii) **"100 agents thinking in parallel"** = best-of-N / competing-hypothesis WM synthesis
    (R1 belief-entropy gate + R2 best-of-N + B4 hypotheses) — on the Mac this is sequential-batched
    N, not literally 100 processes (memory-bound alongside BFS), but the DISCRIMINATION logic is the
    same: spawn N candidate models, spend scored actions on the most-discriminating move, keep the
    survivor; (iii) **VLM perception** — we deliberately use exact connected-component perception (B1)
    INSTEAD of a VLM because R8 shows ~80% of VLM ARC failures are perception errors; keep symbolic
    perception as the front-end and reserve a VLM only for a fallback describe-the-scene path; (iv)
    **neuro-symbolic / brain-mimicking nets** = the TRM/HRM/StochasticGoose learned priors (R9/R11)
    feeding the symbolic BFS/WM cascade — that IS the neuro-symbolic combo (tiny net for intuition,
    program-synthesis WM for verified planning). *Action:* treat R14 as the assembly spec, not a
    ticket — each cycle ship ONE of R9–R13 env-gated, and the "ensemble" emerges as the cascade
    `blitz → BFS → [TRM/framechange prior orders actions] → best-of-N executable WM (R1 gate) →
    Opus-teacher fallback`. *Done when:* the combined env-gated stack solves ls20 L5 or L6 end-to-end.
    **BLOCKER: aggregate — GPU training route (R9/R11) + Mac API access (R13) + more RAM for real
    best-of-N.** Sequenced behind R9–R13; do not attempt as a single big-bang change.

## Stop condition
All 20 levels across the 3 games solved + verified at RHAE 1.0 (or the highest reachable),
and the offline Kaggle notebook reproduces them. Then freeze and submit. Epic B has its OWN
success metric — held-out generalisation (a concept from one game solving another, B6) — pursued
in parallel without ever risking the verified corpus.
