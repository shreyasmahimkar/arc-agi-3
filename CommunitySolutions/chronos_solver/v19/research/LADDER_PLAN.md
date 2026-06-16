# Plan: LADDER + TTRL → close the 0.22 → 1.0+ gap on ARC-AGI-3

**Paper:** LADDER — *Self-Improving LLMs Through Recursive Problem Decomposition*
(Simonds & Yoshiyama, **Tufa Labs**, Mar 2025, arXiv:2503.00735).
**Why it matters here:** Tufa Labs — the authors of this method — are the team
sitting at **1.0+** on ARC-AGI-3 while we're at **0.22**. LADDER + TTRL is almost
certainly *how they got there*, and it maps onto our repo because we already have
the one thing it requires: **a reliable verifier** (the engine's
`levels_completed`).

---

## 1. What LADDER + TTRL actually is

- **LADDER (training):** for a hard problem the model can't solve, recursively
  generate a **tree of progressively *simpler* variants** (each with one parent →
  a clean difficulty gradient). Solve the easy ones, use them as stepping stones,
  and train with **GRPO** (RL with *verifiable* rewards — no human labels, no
  critic). Result on integration: Llama-3B **1% → 82%**; 7B on the MIT Bee
  **50% → 73%**.
- **TTRL (inference):** for *each test problem*, generate variants **at test time**,
  run a short GRPO loop to **adapt the policy to that specific problem**, then
  solve. MIT Bee **73% → 90%**, beating OpenAI o1.
- **Two findings that decide our design:**
  1. *"SFT memorises, RL generalises"* — RL with verifiable rewards transfers; SFT
     does not. **This is the exact diagnosis of our world model's ~0.17 plateau**
     (it's trained by supervised next-frame prediction → it memorises).
  2. *"RL without variants fails."* The **curriculum** (the variant ladder) is the
     active ingredient, not the RL algorithm. TTRL also only works on a
     LADDER-pretrained policy.

---

## 2. The mapping — every LADDER requirement, we already have

| LADDER needs | ARC-AGI-3 / our repo provides |
|---|---|
| a **verifier** (cheap to check, hard to solve) | the engine's `levels_completed` — a level-complete is trivially verifiable, solving is hard. Perfect generator-verifier gap. |
| a **variant generator** (easier sub-problems) | the **white-box engine + BFS** give it *for free*: re-root at a **BFS progress-landmark** (closer to the goal), **shrink** the level (fewer keys, nearer goal), **isolate one mechanic**, or use the game's own **L0→Lₙ** as a built-in ladder. |
| **GRPO RL** with that reward | replace the **SFT** world-model/policy training with **GRPO over the engine reward** — the fix for the plateau. |
| **TTRL** at test time | on each scored game, RL-adapt the policy to *that* game **on a forked engine** (Kaggle ships the sources) → **free under RHAE** (no scored actions spent). |

The crucial RHAE point: because Kaggle ships the game sources, **TTRL's adaptation
runs on a forked simulator, not the scored episode** — so the test-time RL costs
*zero scored actions*. That's what makes TTRL viable here when it would be too
action-expensive in a pure black-box setting.

---

## 3. Why this is the right lever for *our* gap

- **0.22 is the BFS floor** — white-box BFS solves the games it can reach. The gap
  to 1.0+ is the **hard levels BFS can't crack** and the **games where the agent
  must learn novel rules**. That is *precisely* what TTRL does: learn the game at
  test time.
- **Our world-model plateau (~0.17 chg-acc, augmentation failed twice) is an SFT
  symptom.** LADDER's headline finding says the cure is **RL, not more data or a
  bigger model** — which also matches our own "architecture/data-bound, not
  compute-bound" diagnosis. So the next investment is the *training paradigm*
  (SFT → GRPO), not the GPU.
- It's **proven by the people ahead of us** on this exact benchmark.

---

## 4. Staged plan (research-first; no agent-code changes until proven)

**Stage 0 — validate the lever cheaply (1 game, prototype).**
Pick one held-out level BFS can't solve (e.g. `ls20` L5 or a `ka59` level).
Generate ~8 variants by **re-rooting at BFS progress-landmarks** (easier starts) +
**shrinking**. Run a tiny **GRPO** loop (rollouts on the forked engine, reward =
level-complete) to adapt a small policy, then attempt the original level. **Success
metric:** does TTRL crack a level the base agent could not? One crack = the lever
is real. *(This is a standalone research notebook — not a change to the shipped
agent.)*

**Stage 1 — LADDER training (offline, on the 274-game corpus).**
Build the **variant generator** (`make_variants`: re-root at landmarks, shrink,
isolate-mechanic, with BFS verifying each variant's difficulty so unsolvable ones
are filtered — addressing the paper's ~8% bad-variant problem). Train a **policy**
(reuse the ChangeNet backbone + an action head) with **GRPO** over the variant
trees, engine reward. This replaces SFT next-frame training as the *generalising*
component. Measure on the held-out games (the same honest split we already use).

**Stage 2 — TTRL at inference (the score lever).**
Wire TTRL into `combined_agent` *behind BFS*: when BFS stalls on a game, fork the
engine, generate variants of its reachable levels, run a short GRPO adaptation,
then solve. Parallelise across the RTX (the paper notes TTRL is embarrassingly
parallel). **Keep BFS-first** — it holds the 0.22 floor; TTRL adds the hard/hidden
games on top.

**Stage 3 — dynamic difficulty calibration (their stated future work).**
Adjust variant difficulty from observed success rate so each variant is in the
"learnable zone" (not trivial, not impossible).

---

## 5. Expected impact, risks, honest caveats

- **Upside:** this is the *documented recipe* of the team at 1.0+. The MIT-Bee
  analog (50→73→90) is a +40-point swing on hard reasoning from the same two moves
  (curriculum RL + TTRL). On ARC-AGI-3 the equivalent is converting unsolved hard
  levels into solved ones at test time.
- **Cost:** GRPO + TTRL is **engineering-heavy** (an RL loop, variant generation,
  reward plumbing) and **compute-heavy at inference** — but the RTX + parallelism
  + the *forked-engine, free-under-RHAE* property make it feasible.
- **Risks:** (a) variant quality — bad variants waste compute (mitigate with BFS
  difficulty-verification); (b) TTRL needs a LADDER-pretrained policy first (Stage
  1 before Stage 2); (c) the Kaggle wall-clock budget must allow the per-game RL
  (time-box it, fall back to BFS/cache on timeout).
- **Honest framing:** this is a *bigger* bet than the world-model tweaks — but it's
  the one with a proven ceiling. The cheap Stage-0 prototype de-risks it before any
  agent-code change.

---

## 6. One-line summary
*We're at the BFS floor (0.22); the gap to Tufa's 1.0+ is "learn the hard/hidden
games at test time." LADDER (curriculum RL) + TTRL (test-time RL) is their proven
recipe for exactly that, it fixes our SFT-memorisation plateau with RL, and —
because Kaggle ships the engine — TTRL's adaptation is free under RHAE. Validate it
on one game first, then stage it in behind BFS.*
