# Chronos v21 — Plan to crack ARC-AGI-3 on Kaggle (generalize-first)

**Status:** research + plan for review. Written 2026-07-06.
**Thesis:** we do not need a new architecture. The frontier proves the *shape* we already
have in `v20` is correct; v21's job is to make it **RHAE-optimal, offline-safe, and
self-improving on a 4-hour cadence** — validated against 3 games (`ls20`, `ft09`, `vc33`)
chosen because each stresses a *different* capability, so a solver that generalizes to all
three generalizes to the taxonomy of the whole public set.

---

## 1. What the research settled (and why it changes our priorities)

The single most important external result is the **"Explore Before You Solve"** paper
(arXiv:2605.25931). It classifies **all 25 public games** and shows the Kaggle-winning
kernel is not clever reasoning — it is disciplined **EXPLORE-before-PLAN**:

- The public competition winner, `arc-agi3-v31-zorojuro-hybrid-v9`, scored **RHAE = 0.30**
  (30%) using **offline BFS pre-solve + cached plan replay + heuristic fallback**. That is
  *structurally identical to our v20 cascade* (Memory → BFS → Forge → LADDER).
- **Kaggle is fully offline** — no internet, no external LLM API. The 58% "executable world
  model" agent (arXiv:2605.05138, GPT-5.5) is **not competition-eligible as-is** because it
  calls a coding LLM. Any world-model stage we ship must run local-only on a T4/P100.
- **RHAE scoring is the whole game.** Per level `score = min(1, human_actions/ai_actions)²`,
  with a **hard 5× cap** (>5× human actions ⇒ attempt terminated) and quadratic falloff (2×
  longer = 25% of the score). **Shortest correct solution wins.** Our own logs already
  confirm this: v19 `auto` (13/45/39/43/44 on ls20) ≫ v13 greedy (63/71/78).

**Second-order confirmations:**

| Source | Finding | v21 implication |
|---|---|---|
| BentoLabs self-learning layer | +2.6× (1.27%→3.32%), no retraining, 34% cheaper | Upgrade Memory from exact-level cache → cross-game **macro/pattern retrieval** keyed on frame features. Cheapest high-leverage win. |
| StochasticGoose (Tufa, 12.58% preview) | CNN change-predictor + conv coord head for ACTION6 | This is our `forge_agent.py` lineage — keep as the no-source fallback, warm-start it. |
| Graph exploration (arXiv:2512.24156) | training-free frontier search over the state graph | Already in `v20/src/graph_explore.py`; make it the default no-source explorer. |
| Executable World Models (58% public) | LLM writes/verifies Python transition fn | **Not offline-eligible.** Keep as an *offline research generator* only (see §6), never in the submission path. |

**Bottom line:** the gap between us and the top is **not approach** — it is (a) RHAE
optimality, (b) coverage/generalization to unseen games, and (c) an honest self-improving
loop. v21 targets exactly those three.

---

## 2. The 3 games as a generalization "source of truth"

The paper's taxonomy (Table 8/9) places our three chosen games in **three distinct tiers** —
this is why they are a good cracking set: solve all three well and you've covered the span
from blind reflex to search to click-orchestration.

| Game | Tier (paper) | Ground-truth win | Human baseline | What it stresses | v20 status |
|---|---|---|---|---|---|
| **ft09** | ACTION6 depth-1 (blind) | single `ACTION6` wins blindly | low | Reflex / one-shot hypothesis | cache: L0=4, L1=7 actions |
| **vc33** | Other depth-1 (blind), `click` tag | 1-step blind win (click) | low | Click-target selection / orchestration | cache: L0–L3 (3/7/23/21) |
| **ls20** | Budget-constrained (keyboard) | `ACTION2` ×129 (blind cheese) **or** BFS-optimal 13/45/39/43/44 | high (350–1843) | Multi-step agent reasoning / maze | cache: L0–L4, optimal counts |

Two design lessons fall straight out of this table:

1. **A "cheese" (repeat-one-action) solve can be RHAE-excellent** when the human baseline is
   large. `ls20` via `ACTION2`×129 is *blind* yet 129 < human baseline ⇒ RHAE near 1.0. So
   the cascade must **race a cheap blind/repeat probe against BFS** and keep whichever is
   shorter — not assume search is always better.
2. **ft09/vc33 must never burn actions.** They are one-step wins; any exploration before the
   winning action is pure RHAE loss. The Memory/probe stage has to commit *immediately* when
   a depth-1 win is known or quickly found on the forked engine.

> ⚠️ **Do not rely on the null-coordinate `ACTION6(x=None,y=None)` exploit** (paper §appendix):
> it triggers a `TypeError` the library mis-handles as WIN. It is unintended engine behavior,
> likely patched on the private set, and is not a legitimate solve. v21 flags and refuses it.

---

## 3. Target architecture — v21 cascade (offline, RHAE-optimal)

Keep the v20 `MyAgent` cascade skeleton; make five changes. Every search stage runs on a
**forked simulator snapshot** (zero scored actions), and every stage returns the **shortest**
verified plan it can find within its time box.

```
choose_action(frame, level):
  STAGE 0  BLITZ PROBE   race, on the fork, the cheap wins first:
                         { depth-1 blind (each ACTIONn once), repeat-action×K,
                           click-on-each-object } — pick shortest that verifies.
                         (covers ft09, vc33, and ls20's cheese in <1s)
  STAGE 1  MEMORY        verified recall from the flywheel corpus for THIS level
                         (replay-verify on a clean fork; STALE/MISS -> escalate)
  STAGE 1b SELF-LEARN    cross-game macro retrieval by frame-feature key -> BFS seed
                         (BentoLabs 2.6x lever; retrieve from similar games, not just identical)
  STAGE 2  BFS 'auto'    white-box optimal ladder (A*/masked BFS, shortest-first),
                         only when source is reachable; time-boxed vs the 5x cap
  STAGE 3  GRAPH/FORGE   no-source: graph_explore frontier search; warm-started ChangeNet
  STAGE 4  LADDER        variant re-root (Go-Explore) + TTRL suffix-BFS for the wall levels
  -> commit shortest verified plan; persist to flywheel; chain next level
```

The five changes vs v20:

1. **Stage 0 "Blitz probe"** (new): the paper's whole point — try the trivial wins *first* on
   the fork. Makes ft09/vc33 one-step and gives ls20 a cheap RHAE-strong fallback.
2. **RHAE-optimal everywhere:** BFS prefers shortest (A*/masked), greedy only as last-resort
   "solve at all." Keep the shortest of {blitz, memory, BFS} per level.
3. **Self-learning memory (Stage 1b):** store *transferable macros/patterns* keyed by frame
   features (object count, palette, motion signature), retrieve for *similar* unseen games.
   This is the coverage lever for the private set.
4. **Offline guardrail:** hard assert no network / no external API is ever touched in the
   submission path. World-model synthesis is offline-only (§6).
5. **5× cap governor:** a per-level action budget monitor that aborts a losing line before it
   trips the cap and terminates the attempt.

---

## 4. The 4-hour cadence — two halves: SOLVE, then EVOLVE

The cadence is not just "re-run BFS." Each run has a **SOLVE** half (attack the games) and an
**EVOLVE** half (improve the machinery that does the solving). Four assets get better every
run, each behind a verify-or-discard gate. This is the BentoLabs self-learning layer made
concrete + the offline form of EXPLORE-before-PLAN.

**SOLVE (attack the 3 games + a held-out probe):**
1. Version-exact source resolution (newest hash; stale hashes poison the verifier).
2. Solve/tighten each level; keep **only strictly-shorter** verified plans (RHAE monotone).
3. Replay-verify every kept plan on a clean fork; reject non-reproducing wins.
4. Compute RHAE per level vs the **official** `baseline_actions`; scorecard + regression gate.

**EVOLVE (improve the agent, not just its answers):**
5. **Skill/macro library** — abstract each new solve into transferable macros keyed by frame
   features. Lets unseen games be solved without new code.
6. **Intuition prior** (`intuition.py`) — re-distill the whole corpus into an action-ordering
   prior (early-move-weighted) that biases blitz/search *next* cycle. This is the amortized
   "System-1"; the interface is fixed so a trained policy net drops in later. Runs every cycle.
7. **Code-writer evolution** (`evolve.py`, the champion/challenger loop) — the LLM proposes
   challenger configs + heuristic code aimed at the **wall levels**; each is evaluated and a
   challenger is promoted **only if it beats the champion on held-out without regressing
   train** (generalization-gated). Nothing ships unverified.

**How it compounds:** `plans → distilled into skills → distilled into the intuition prior →
speeds search → finds more plans`, while the code-writer widens *what is solvable at all*.

**Two code-writers (this is the "thinks and writes code" layer you asked for):**
- **Runtime, on-the-fly** (`runtime_coder.py`) — inside the live agent, a **local** Qwen2.5-Coder
  writes an executable `WorldModel` for the current level from observed transitions, we exec it
  in a **restricted sandbox** (numpy/math only, timeout), verify it reproduces frames, enumerate
  its candidate plans, and commit the shortest that wins on the fork. This is the 58% "Executable
  World Models" method made **offline-eligible** with a local model. Verified in tests: it writes
  and runs models that solve ft09-like (1×ACTION6) and ls20-like (repeat-ACTION2) plans.
- **Between-rounds** (`evolve.py`) — a larger model (or API) on the cadence box evolves the
  solver itself across cycles. Never in the submission.

**Local models (open-source, HF):** runtime = `Qwen/Qwen2.5-Coder-7B-Instruct` 4-bit (~6GB,
fits a 16GB T4), fallback `-1.5B-Instruct`; cadence box may use `Qwen/Qwen3-Coder-Next`
(3B-active MoE, top 2026 local coder). Weights bundle as a Kaggle **dataset** with
`HF_HUB_OFFLINE=1` — no download at scored runtime. If the model is absent, `llm_backend`
falls back to a deterministic mock and the non-LLM stages still run (the LLM is additive).

Cadence rationale: 4h × 6/day gives escalating-budget passes (180→600→1800 s/level) plus one
evolve round each cycle, without any single run exceeding a Colab session, and keeps the shipped
bundle fresh against engine-version rotation.

---

## 5. Phased roadmap

| Phase | Goal | Exit criterion |
|---|---|---|
| **P0 – Wire cadence** | `cadence_runner.py` runs the 3 games end-to-end, writes scorecard + logs, scheduled every 4h | 1 clean run with RHAE for all 3 games recorded |
| **P1 – Blitz + RHAE-optimal** | Add Stage 0 blitz probe; force shortest-plan in BFS | ft09/vc33 solved in ≤ their human baseline; ls20 RHAE ≥ 0.9 via best of cheese/BFS |
| **P2 – Self-learning memory** | Cross-game macro bank + retrieval | ≥1 held-out game solved using a macro learned from the 3 (no per-game code) |
| **P3 – No-source robustness** | graph_explore default; warm-started Forge | ≥3/5 held-out games solved black-box within budget |
| **P4 – Submission hardening** | Self-contained Kaggle notebook, offline guardrail, 5× governor | Notebook runs offline on T4, reproduces cached RHAE, no network calls |
| **P5 – LADDER on the wall** | variant re-root + TTRL for ls20 L5 / deep levels | ls20 L5 cracked through the cascade |

---

## 6. Offline world-model track (research only, never shipped live)

The 58% executable-world-model method is powerful but needs an LLM. We use it **offline**: on
the cadence machine (which *does* have internet), an optional generator can synthesize/verify
a Python transition model per game and distill its winning plans into the flywheel corpus. The
**submission only ever ships the distilled plans + the local graph/Forge explorer** — never an
API call. This captures the method's coverage without violating Kaggle's offline rule.

---

## 6b. Current RHAE, grounded in official baselines (the coverage picture)

Wired the official per-level `baseline_actions` (from each game's version-exact
`metadata.json`) and scored the shipped cache against them. This reframes where the work is:

| Game | Solved / levels | Mean RHAE (solved) | Where the points are |
|---|---|---|---|
| ls20 | 5 / 7 | 0.966 | L1 is the only sub-optimal solve (**45 vs baseline 41** → 0.83); **L5, L6 are the wall** |
| ft09 | 2 / 6 | 1.000 | both solves perfect, but **L2–L5 unsolved** — the real gap despite "reflex" tier |
| vc33 | 4 / 7 | 1.000 | all solves perfect; **L4–L6 unsolved** |

Two levers fall straight out and set P1/P5 priorities:
1. **Optimality is nearly done on what we solve** (only ls20 L1 is loose → tighten to ≤41).
2. **Coverage is the real score driver** — unsolved levels score 0 and the benchmark
   depth-weights deeper levels. The wall levels (ft09 L2–L5, vc33 L4–L6, ls20 L5–L6) are
   where LADDER/graph/self-learning must earn their keep. ft09 being only 2/6 is the loudest
   signal: a "reflex" game whose later levels we can't yet reach.

## 7. Resolved decisions & remaining risks

1. **White-box vs black-box — RESOLVED: build both, cascade auto-detects per game** (R2.8).
   Each level first tries version-exact source resolution → white-box BFS/graph over engine
   snapshots; if no source loads, it falls through to the black-box track (blitz probe →
   graph_explore → warm Forge) on API frames only. We never assume the whole set is one mode;
   the detected mode is logged per game so the private set's mix is measured, not guessed.
2. **Human baselines — RESOLVED: official `baseline_actions` wired** from version-exact
   `metadata.json` (R1.4); proxy only if metadata is absent, and labeled. See §6b.
3. **Engine-version rotation.** The private set may use version hashes we've never solved;
   generalization (Stage 1b/3) — not the cache — is what carries those. This is why the 3-game
   set is a *generalization* probe, not a memorization target.
4. **Compute for the world-model track.** Local LLM vs paid API on the cadence box — cost/latency
   tradeoff to decide before P2.

See `REQUIREMENTS.md` for the itemized, reviewable requirements and `requirements.txt` for deps.
