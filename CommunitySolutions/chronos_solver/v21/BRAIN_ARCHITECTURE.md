# Chronos v21 — Cognitive ("Brain") Architecture

*A cognitively-inspired layer stacked on the existing v21 solver so the 4-hourly
loop keeps cracking specific wall levels **and** grows toward a game-general
agent that transfers to unseen ARC-AGI-3 games.*

This is not a literal neural brain. "Brain" is a metaphor for a set of
subsystems — perception, prediction, hypothesis testing, planning, memory, goal
inference, and consolidation — that together produce the behaviour ARC-AGI-3
demands. Every design choice below is grounded in current research and mapped
onto code that already exists in `v21/`.

---

## 1. Why this, why now

ARC-AGI-3 is an *interactive* reasoning benchmark: an agent is dropped into an
unfamiliar turn-based game with no instructions, no stated goal, and no rules,
and must **explore, infer the goal, model the dynamics, plan, and act
efficiently** (score = Relative Human Action Efficiency). At launch (March 2026)
humans solved 100% of environments; frontier models scored ~0.5%.

The current Chronos stack is a strong *search* engine: `blitz` races cheap wins,
`BFS` finds shortest verified action plans, `runtime_coder` writes a one-shot
world model, `evolve`/`intuition` tune configs and priors. It reaches RHAE 1.0
on the levels it can search — but it does not **model** a game, does not
**transfer** knowledge between games, and stalls on walls where blind search
explodes (vc33 L4 timed out at ~101k states/180s).

The most relevant result is Rodionov 2026, *Executable World Models for
ARC-AGI-3*: a coding agent that maintains a **verified, refactorable Python
world model**, plans **inside** it (unscored), and only spends scored
environment actions on plans that pass an online frame-by-frame check. It fully
solved 7/25 public games (mean RHAE 32.6%). Its two documented failure modes
tell us exactly what a "brain" must add:

1. **Tunnel vision** — the agent commits to one early ontology/goal and
   elaborates it instead of considering alternatives; a wrong hypothesis on
   level 1 contaminates the whole game.
2. **Weak planning over a correct model** — even with good dynamics, a naive
   planner can't search the induced state space.

So the brain layer = **executable world model** (spine) + **competing-hypothesis
management** (fixes #1) + **strong planner skills** (fixes #2) + **object-centric
perception** (fixes the upstream "perception bottleneck", arXiv:2512.21329) +
**cross-game memory** (the generalisation organ) + **goal induction** +
**wake-sleep consolidation** (the self-improvement flywheel — i.e. the 4-hourly
loop itself).

Chosen spine: **executable / program-synthesis world models**, not neural latent
(H-JEPA/Dreamer). Rationale: it is offline-verifiable in small increments (fits
the loop's hard test gate), needs no GPU training, and builds directly on
`runtime_coder`. A neural latent model is kept as an optional far-future slot
(phase B8), behind the same planner/goal interfaces.

---

## 2. The seven subsystems (and their brain metaphor + code home)

| Subsystem | Brain metaphor | Job | Code | Status |
|---|---|---|---|---|
| **Perception** | visual cortex | raw frame → object-centric scene graph, frame-diff, click targets | `brain/perception.py` | **implemented** |
| **World Model** | predictive cortex / mental simulation | persistent, verified, refactorable executable dynamics per game | `brain/world_model.py` (+ generalises `runtime_coder`) | verifier core done; authoring/persistence next |
| **Hypotheses** | prefrontal / scientific reasoning | keep competing world models; act to *discriminate* them; falsify on mismatch | `brain/hypotheses.py` | decision cores done |
| **Planner** | motor cortex / basal ganglia | plan **in** the model (unscored), then execute-and-verify (MPC) with abort-on-mismatch | `brain/planner.py` (+ reuses `blitz`/`BFS` as skills) | pure cores done |
| **Memory** | hippocampus → neocortex | cross-game library of macros, world-model fragments, perceptual motifs; retrieval by perceptual key | `brain/memory.py` (+ `v21_macro_bank.json`) | key + retrieval done |
| **Goal** | limbic / reward | induce the unstated goal from score/frame signal | `brain/goal.py` | score-signal inducers done |
| **Consolidation** | sleep / wake-sleep replay | between runs: distil priors, refactor the library, tune configs | extends `evolve.py` + `intuition.py` | seed exists (`evolve`,`intuition`) |

All of `brain/` is pure/dependency-free at import (no arcengine/numpy/torch/
network), exactly like `blitz.py`, so the offline self-test exercises it.

---

## 3. The cognitive control loop

The coordinator is an upgraded `cadence_runner.solve_game` — the "thalamus" that
routes between subsystems. For each UNSOLVED level:

```
observe frame ─▶ PERCEIVE (scene graph, click targets)
              ─▶ RECALL   (memory.retrieve by perceptual key → macros/WM fragments)
              ─▶ HYPOTHESIZE (seed/verify competing WorldModels; verify_model)
              ─▶ PLAN-IN-MODEL (planner.plan_in_model on the trusted WM; unscored)
                    │  (if no trusted model yet: pick most-discriminating action
                    │   to cheaply falsify hypotheses instead of guessing)
              ─▶ ACT-AND-VERIFY (planner.execute_and_verify: step real env +
                    model in lockstep; abort on first frame mismatch)
              ─▶ UPDATE (falsify wrong hypotheses; refine WM; record transitions)
              ─▶ on LEVEL_COMPLETED → CONSOLIDATE (add macro/fragment to memory,
                    distil prior) and continue.
```

**Safety invariant.** The brain is *additive*. The proven cascade
(`blitz → BFS → runtime_coder`) remains the fallback, and every plan the brain
proposes still passes `verify_solution` + the shortest-plan corpus gate before
it can enter the corpus. Each subsystem is wired into the live path only behind
its own env flag (e.g. `V21_BRAIN_PERCEPTION`, `V21_BRAIN_WM`), default OFF, and
only after a Mac cadence proves it. The offline submission guard is never
disabled. This is why the brain can be built incrementally without ever risking
the verified corpus (the regression gate protects it).

---

## 4. How each piece attacks the current walls

- **vc33 L4–L6 (click orchestration).** `perception.click_targets` gives one
  target per *connected component* (v19 used one median per colour, which can
  land between blobs). Fewer, better-placed targets shrink BFS branching; the
  new `blitz` click-repeat tier hammers a single component. → phase B1 already
  usable.
- **ft09 L2–L5 (non-blind reflex).** A verified world model lets the planner
  reason about *why* an ACTION6 works instead of brute-forcing; MPC abort
  catches a wrong model in one step.
- **ls20 L5–L6 (keyboard maze).** `plan_in_model` runs suffix search from the
  L4 end-state inside the model (unscored), and `memory` replays sibling/other-
  game maze macros as Go-Explore seeds.
- **Generalisation to unseen games (the real prize).** `memory.perceptual_key`
  is deliberately game-agnostic (grid shape, object count, size signature — not
  colours or absolute positions), so a concept learned on one game is retrieved
  for a perceptually-similar situation in another.

---

## 5. Build order for the 4-hourly loop (see BACKLOG Epic B)

Each phase is one or a few loop cycles, each ending green on `py_compile` +
`test_offline.py`, committed, env-gated OFF until a Mac cadence proves it:

- **B1 Perception** *(done this session)* — scene graph + component click
  targets + frame-diff. Immediately consumable by `blitz`/BFS click selection.
- **B2 World-model persistence + verifier** — generalise `runtime_coder` to a
  per-game model on disk (`brain/wm/<game>/`) that must reproduce recorded
  transitions (`verify_model`); refactor pass for MDL simplicity.
- **B3 MPC plan-executor** — `execute_and_verify` wired to the real engine;
  scored actions only on model-verified plans.
- **B4 Hypothesis manager** — seed 2–3 competing models; spend actions on the
  most-discriminating move; falsify on mismatch (kills tunnel vision).
- **B5 Goal induction** — from score/level signal now; frame-motif goals later.
- **B6 Cross-game concept library** — persist macros + WM fragments + motifs;
  retrieve by perceptual key to seed a *different* game's search.
- **B7 Wake-sleep consolidation** — in `evolve`: replay solved trajectories,
  compress/refactor the library, re-distil the intuition prior.
- **B8 (optional, far)** — neural latent world model (H-JEPA/Dreamer) behind the
  same planner/goal interfaces, if/when a GPU training path exists.

**Stop condition unchanged:** all 20 target levels verified at best-reachable
RHAE *and* the offline Kaggle notebook reproduces them — then freeze/submit. The
brain layer's own success metric is **held-out generalisation**: a concept
learned on one game solving a level of another.

---

## 5b. Literature validation (research feed, 2026-07-07)

A RESEARCH-branch scan confirmed the brain's direction and added three concrete
mechanisms (tracked as BACKLOG R1–R3):

- **Explore→Verify→Plan + belief-entropy commit gate** (AERA, arXiv:2605.25931).
  The mechanism that produces non-zero RHAE on hidden-rule games is an explicit
  world-model *hypothesis* plus a gate that only hands off to the planner once
  belief entropy drops (or a ≈40%-of-human-baseline explore budget is spent).
  This is the missing piece for subsystem 3 (hypotheses) → 4 (planner): make the
  explore→plan transition a real, entropy-gated commitment rather than a fixed
  cascade order. Their 55-game code-track entry is "BFS + offline pre-solve
  cache" (RHAE 0.30) — i.e. our exact spine — external validation of the design.
- **Verify → MDL-refactor → plan-through-model** (Rodionov, arXiv:2605.05138):
  reinforces subsystems 2–3; prioritize the MDL refactor pass (shorter program
  reproducing all transitions) and best-of-N WM synthesis (runs vary a lot).
- **Graph-based level explorer** (arXiv:2512.24156): frame processor (segmentation
  → status-bar masking → priority action grouping → **state hashing**) → graph
  explorer with **frontier management**. State-hash dedup + frontier management
  drop straight into the BFS/planner; priority action grouping orders ACTION6
  targets — all directly useful for vc33 L4–L6.

Speed–Depth note (RHAE is quadratic in action count): keep the shortest-plan gate
strict; long repeat-heavy wins should be revisited to shorten, not left at RHAE<1.

A second RESEARCH-branch scan (2026-07-07, RESEARCH-2) added one new mechanism
(BACKLOG R6), distinct from R1–R5:

- **Orchestrator + compressed-summary subagents** (Symbolica *Arcgentica*,
  open-source, 36.08% / 113-of-182 levels on the 25-game public set, solves all
  3 public envs). The transferable idea is context control: subagents return
  **compressed textual summaries** rather than raw transcripts, so the model's
  context stays bounded as exploration deepens. For us this is a single-coder
  optimisation, not a multi-agent rewrite: compress recorded transitions into a
  fixed-size structured digest (scene deltas + tried-action→outcome table) before
  the `runtime_coder`/planner prompt. This directly targets the 160000Z-style
  stall where the coder step times out/OOMs on a deep wall's growing context.

A third RESEARCH-branch scan (2026-07-07, RESEARCH-3) added two mechanisms
(BACKLOG R7–R8), distinct from R1–R6:

- **Workspace optimization** (NVIDIA/Technion *DREAMTEAM*, arXiv:2605.09650;
  38.4% on the 25-game public set — new SOTA over Symbolica's 36.08% — with 31%
  fewer env actions). Since the frontier/coder weights are frozen, the *workspace*
  (the files the agent reads/writes/tests) is the trainable object:
  artifacts↔parameters, evidence↔data, counterexamples↔losses, feedback↔gradients.
  Our corpus/champion/intuition/macro-bank already ARE that workspace; the concrete
  add is persisting **failed-wall counterexamples** the next cycle's coder reads as
  negative constraints, plus an **actions-per-solve** tie-breaker in evolve (R7).
- **Perception bottleneck** (CMU et al., arXiv:2512.21329): ~80% of VLM ARC
  failures are perception, not reasoning; a perception→description stage adds
  +11–13pp. VALIDATES our exact connected-component perception (we sidestep the
  VLM bottleneck) and motivates making the R6 coder digest **perception-first** — a
  structured object/scene schema, not a serialized raw grid (R8).

---

## 6. Key references

- ARC Prize Foundation (2026). *ARC-AGI-3: A New Challenge for Frontier Agentic
  Intelligence.* arXiv:2603.24621.
- Liew, K.H. (2026). *Explore Before You Solve: The Speed–Depth Trade-off in
  Epistemic Agents for ARC-AGI-3.* arXiv:2605.25931. (EXPLORE/VERIFY/PLAN +
  belief-entropy commit gate; ≈40%-of-baseline explore budget; RHAE-as-Pareto.)
- Rudakov, Shock, Cowley (2026). *Graph-Based Exploration for ARC-AGI-3
  Interactive Reasoning Tasks.* arXiv:2512.24156. (Frame processor + state hashing
  + frontier-managed graph explorer.)
- ARC Prize Foundation (2026). *ARC Prize 2025: Technical Report.* arXiv:2601.10904.
  (NVARC winner: Qwen-4B + test-time training + synthetic data.) McGovern, R.
  (2025). *Test-time Adaptation of Tiny Recursive Models.* arXiv:2511.02886.
- Rodionov, S. (2026). *Executable World Models for ARC-AGI-3 in the Era of
  Coding Agents.* arXiv:2605.05138. (Verifier-driven executable WM; tunnel-vision
  + weak-planner failure modes.)
- Tang, Key, Ellis (2024). *WorldCoder: building world models by writing code.*
  arXiv:2402.12275.
- Ellis et al. (2020). *DreamCoder: wake-sleep Bayesian program learning.*
  arXiv:2006.08381. Grand et al. (2023). *LILO: learning interpretable
  libraries.* arXiv:2310.19791. (Library learning / MDL refactor.)
- LeCun / H-JEPA and follow-ups (2026): hierarchical latent predictive world
  models & planning in representation space — the neural-latent B8 slot.
- Wang, X. et al. (2026). *Your Reasoning Benchmark May Not Test Reasoning:
  Revealing Perception Bottleneck in Abstract Reasoning Benchmarks.*
  arXiv:2512.21329. (~80% of ARC failures are perception; perception→description
  stage adds +11–13pp — validates symbolic connected-component perception — R8.)
- NVIDIA / Technion (2026). *Workspace Optimization: How to Train Your Agent
  (DREAMTEAM).* arXiv:2605.09650. (Evolve the read/write/test workspace as the
  trainable substrate; 38.4% public-set SOTA, 31% fewer env actions — R7.)
- Symbolica AI (2026). *Arcgentica: REPL/orchestrator agents for ARC-AGI-3.*
  Open-source `symbolica-ai/arcgentica` + `symbolica-ai/ARC-AGI-3-Agents`; blog
  https://www.symbolica.ai/blog/arc-agi-3 . (Orchestrator + compressed-summary
  subagents for bounded context; 36.08% on the 25-game public set — R6.)
