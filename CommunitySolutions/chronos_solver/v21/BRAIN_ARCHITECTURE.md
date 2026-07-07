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

## 6. Key references

- ARC Prize Foundation (2026). *ARC-AGI-3: A New Challenge for Frontier Agentic
  Intelligence.* arXiv:2603.24621.
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
- *Perception bottleneck in abstract-reasoning benchmarks*, arXiv:2512.21329.
