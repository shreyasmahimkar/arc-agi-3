# v19 — Technical Case Study (for a Data Science / AI audience)

A deep, honest walkthrough of the v19 agent for **ARC-AGI-3**, written to surface
the *data-science craft* behind it: problem framing, experiment design, honest
generalisation measurement, model-based RL, and the MLOps that keeps it running.

---

## 0. TL;DR (the elevator version)

ARC-AGI-3 is an interactive benchmark where an agent must learn a *novel* game's
rules from pixels with no instructions and a sparse reward — frontier LLMs score
**< 1%**, humans 100%. v19 is an **Expert-Iteration (ExIt) system**: a white-box
search engine *solves* games and **distills those solutions into a learned world
model** that is meant to generalise to *unseen* games. The whole thing runs as a
self-improving flywheel — `solve → harvest → train → repeat`.

The headline data-science story isn't "it hit SOTA" — it's the **disciplined
diagnosis**: I built honest held-out generalisation metrics, ran controlled
ablations (including ones that *failed*), and used a **data-vs-capacity-vs-
architecture decision tree** to decide where compute is worth spending. That
diagnostic rigour — knowing *why* a number is stuck and refusing to throw a GPU
at an architecture problem — is the transferable skill.

---

## 1. The problem, in data-science terms

| property | implication for modelling |
|---|---|
| **Interactive, sequential** (act → observe frame → act) | it's RL / sequential decision-making, not supervised i.i.d. |
| **Sparse reward** (only "level complete" fires) | credit assignment is brutal; naive RL gets ~0 signal |
| **Novel rules per game, no instructions** | the test is **generalisation to unseen tasks**, not fitting one |
| **64×64 categorical "pixels" (16 colours)** | structured spatial input; colours are *arbitrary labels* |
| **Scored on action-efficiency (RHAE: `(human/ai_actions)²`)** | wasted actions are quadratically punished → sample-efficiency is the objective |

The core ML question is therefore **transfer**: can a model learn game *mechanics*
from a training distribution of games and apply them to a held-out distribution?
That reframes a "game-solving" task into a **representation-learning + generalisation**
problem — which is where the data-science work lives.

---

## 2. Architecture: three pillars + a flywheel

```
                ┌─────────────────── ExIt FLYWHEEL (exit_cycle.sh) ───────────────────┐
                │                                                                      │
  (1) SOLVER ───┼──► corpus (solutions/*.json) ──► (2) HARVEST ──► transitions ──► (3) WORLD MODEL
  BFS ladder    │     "expert demonstrations"        replay through        (next-frame + reward predictor)
  combined_agent│                                    real engine            train_wm_v19.py  ─┐
                │                                                                              │
                └──────────────────────── (4) WM-IMAGINATION PLANNER ◄────────────────────────┘
                                            wm_planner.py — plan in the model, verify on the engine
```

**Pillar 1 — the white-box solver (`combined_agent.py`).**
A fusion of the repo's lineage: a **BFS search "ladder"** (`bfs → sense → IW1/IW2 →
EHC → waypoint → A* → greedy → rescues`). In ML terms it's a **near-optimal expert**
that generates ground-truth `(state → action)` demonstrations. IW = *Iterated
Width* (novelty-pruned BFS); waypoint = *sub-goal decomposition*; greedy = *progress
shaping*. It's white-box (it can fork the simulator), so it's only used **offline**
on games whose source is reachable — exactly the role of an expert in ExIt /
AlphaZero. A flagged solution cache (`V19_STORE_SOLUTIONS`) makes runs resumable.

**Pillar 2 — the black-box agent (`forge_agent.py`).**
The *deployable* agent, modelled on the two ARC-AGI-3 preview winners:
- **ChangeNet** (a CNN predicting "will this action change the frame") — the
  *StochasticGoose* idea (1st place, 12.6%); used as an action prior.
- **Transition graph + frontier exploration** — the *Blind Squirrel* idea (2nd,
  6.7%): hash states, prefer untried actions, navigate to the nearest unexplored
  node. RHAE-aware ("never knowingly repeat a transition").
- **Macro-actions** (move-until-the-frame-stops-changing) — borrowed from the
  search lineage to collapse corridors.
It sees **only the public observation** (frame, `levels_completed`, available
actions) — no engine internals — so it's the honest stand-in for the hidden
leaderboard games.

**Pillar 3 — the world model (`train_wm_v19.py`).**
A convolutional **forward dynamics model**: `(frame, action) → (next_frame, reward)`.
Next-frame is a 16-way per-pixel classification; reward is a binary head with
**optimistic up-weighting** (rare "level-complete" steps weighted ×20 so the sparse
signal isn't drowned). This is the *learned* substitute for the white-box simulator
— v15's "knowledge in weights, not stored answers." A **WM-imagination MPC planner**
(`wm_planner.py`) then beam-searches action sequences *inside the model* (free) and
verifies the chosen action on the real engine.

**The flywheel (`exit_cycle.sh`).** Each cycle: solve more games (breadth) →
re-harvest the bigger corpus → **warm-start** retrain the world model → let the
planner attempt the frontier. Self-improving by construction; this is ExIt.

---

## 3. The data-science craft (the part interviewers care about)

### 3.1 Honest generalisation measurement
The single most important design choice: **a frozen held-out split of *games* the
model never trains on** (`cn04, ka59, sk48, tu93, wa30` + a stable hash-bucket of
~10% of all games, so it scales to 274 games). The metric is **held-out
"changed-pixel accuracy" (chg-acc)** — next-frame accuracy *restricted to pixels
that actually changed* (overall pixel accuracy is misleadingly high because most
pixels are background). Measuring on *changed* pixels is the difference between a
vanity metric and a real dynamics metric. This is leakage-aware evaluation done
right: the split is by *game*, not by *frame*, because the unit of generalisation
is the game.

### 3.2 The save-gate (model selection on transfer, not train loss)
Training **warm-starts from the previous weights and only overwrites them if
held-out chg-acc improves**, with **early-stopping on a plateau** (`--patience`).
Two payoffs: (a) the shipped model improves *monotonically* across flywheel cycles
instead of being re-rolled each time; (b) the plateau is an explicit, logged
signal that "more epochs won't help." This is early-stopping + checkpoint-selection
done on the *generalisation* objective — textbook, but applied where it matters.

### 3.3 Controlled ablations — including the one that failed
The world model's chg-acc plateaued at **~0.17** and *over-fit* (peaked ~epoch 3,
degraded by epoch 15). Hypothesis: it memorises each game's specific colours, so it
won't transfer. **Test:** a clean A/B (`wm_augment.py`) — identical model, raw vs
**colour-permutation-augmented** transitions, evaluated on the *same raw held-out
games*. Result, logged honestly:

```
baseline = 0.149   colour-perm aug = 0.130   lift = −0.020   (no help; slightly hurt)
```

A **negative result, reported as a negative result.** That single number redirected
the entire roadmap: the bottleneck is *not* colour-overfitting, so it's deeper
(spatial/relational structure or data quantity), and colour augmentation was
correctly *abandoned* rather than tuned to death. Knowing when to kill an idea is a
core DS competency.

### 3.4 The decision tree: data vs capacity vs architecture
The project's governing rubric (`WM_REPR_EXPERIMENT.md`) — *before* spending money
on a GPU:

> - chg-acc **climbs as the corpus grows** → **data-limited** → keep solving breadth.
> - chg-acc climbs **only with model capacity** (`net_mult 4`) → **capacity-limited** → the GPU is justified.
> - flat in both, but **object/relational features** move it → **representation-limited**.
> - nothing moves it → **task/architecture-bound**; stop pouring compute in.

This is the discipline of **diagnosing the limiter before scaling**. The honest
guardrail — "GPU power buys faster iteration on the prior, *not* a higher score
ceiling; don't graduate to the RTX until Stage 1 shows it's data/capacity-limited"
— is exactly the cost-aware judgement senior ML roles screen for.

### 3.5 Search-as-data-generation (the ExIt insight)
The deepest idea: **use a perfect (but expensive, white-box) solver to manufacture
labelled data for a cheap, generalising model.** That's the AlphaZero / Expert-
Iteration pattern — bootstrap a learner from a search oracle, then (eventually) use
the learner to guide the search. It turns "I have no labels" into "I have a label
factory," which is a high-leverage move whenever you have a verifier but no dataset.

---

## 4. Engineering & MLOps (production-mindedness)

- **Hardware-aware scaling, write-once-run-anywhere:** `auto_bsz` picks batch from
  VRAM (1024 @96GB → 256 @T4 → 128 on Mac/MPS), `--net-mult` scales model width,
  `--amp` + TF32 on CUDA, version-robust AMP shims (`make_grad_scaler`,
  `amp_autocast`) that no-op cleanly on MPS/CPU. The *same* code trains on a
  MacBook (MPS) and an RTX PRO 6000 Blackwell.
- **Checkpoint portability:** models are saved as plain `state_dict`s and
  `from_state_dict`/`mult_of` *infer the width from the checkpoint*, so any-width
  weights load anywhere automatically.
- **Reproducibility & verification discipline:** every claimed solve is verified
  by the real engine's `levels_completed` (never fabricated); each module ships a
  `--selftest`; experiments auto-append results to versioned logs (`WM_LOG.md`,
  `CAMPAIGN_LOG.md`, `PLANNER_LOG.md`) — an audit trail of *improvement per
  iteration*, in the spirit of an experiment tracker.
- **Resumable, fault-tolerant orchestration:** the flywheel is lock-guarded
  (no overlapping runs), warm-starts everything, skips already-solved levels,
  persists BFS frontiers, and **auto-commits progress every few rounds** — so a
  dead cloud instance recovers on a fresh box by `git pull`-ing the pushed state.
- **Two-notebook deploy** (`vastai_setup` / `vastai_train_rtx`): clean separation
  of environment bring-up vs. the training flywheel, with the gotchas handled
  (using the image's CUDA-torch kernel rather than a broken venv).

---

## 5. Results & honest limitations

**What works:**
- The white-box solver is strong: e.g. `ls20` L0 in the optimal 13 actions, `tu93`
  L0–L8, a corpus of ~40 games / 92 levels and climbing toward 274.
- The black-box search agent (v18 lineage) genuinely **generalises by construction**
  — it solved **3/5 held-out games** it had never seen (cn04, sk48, tu93), no stored
  answers, via directed (Go-Explore-style) exploration.
- The full ExIt pipeline runs end-to-end and self-improves with logged metrics.

**What doesn't (yet), stated plainly:**
- The **world model's transfer is weak (~0.17 chg-acc) and plateaued** — the
  learned prior doesn't yet beat the search agent, so the WM-planner cracks no hard
  levels *yet*. Diagnosed as representation/architecture-bound (colour aug didn't
  help; object-centric/relational features are the next test).
- White-box search is **action-expensive** (great for offline data generation,
  not for the live RHAE-scored game) — which is *why* the model-based direction
  exists.

The value is the **clarity of the open question**: it's a representation problem,
not a compute problem, and the experiments to resolve it are queued and cheap.

---

## 6. What this project demonstrates (skills map)

| Data-science competency | Where it shows up in v19 |
|---|---|
| **Problem framing** | recasting "solve a game" as transfer / representation learning |
| **Leakage-aware evaluation** | held-out split *by game*; changed-pixel metric, not vanity accuracy |
| **Model selection on the right objective** | save-gate on held-out transfer + plateau early-stop |
| **Controlled experiments & honest negatives** | the colour-aug A/B (`−0.02`), reported and acted on |
| **Diagnosis before scaling** | the data/capacity/architecture decision tree; GPU-graduation guardrail |
| **Model-based RL / world models** | next-frame+reward dynamics model, optimistic reward shaping, MPC planning |
| **Expert Iteration / search-as-labels** | BFS solver → distilled prior (AlphaZero pattern) |
| **Sparse-reward & sample-efficiency** | RHAE-aware exploration, "never repeat a transition" |
| **MLOps** | HW-aware scaling, AMP, checkpoint portability, resumable auto-committing flywheel, reproducible logs, self-tests |
| **Literature grounding** | StochasticGoose/Blind-Squirrel, Go-Explore, Iterated Width, optimistic world models, ARC colour/D4 equivariance |

**The one-line pitch:** *I built a self-improving Expert-Iteration system for a
hard generalisation benchmark, and — more importantly — I measured its transfer
honestly, ran the ablation that disproved my own hypothesis, and used a
data-vs-capacity-vs-architecture rubric to decide that the bottleneck is
representation, not compute — so I didn't waste a GPU on it.*
