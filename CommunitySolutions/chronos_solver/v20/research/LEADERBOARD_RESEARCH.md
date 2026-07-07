# v20 research — how the top of ARC-AGI-3 is actually scored (2026-06)

Online research into how Tufa Labs (leaderboard **1.21**) and the rest of the frontier
reach the top, and what it means for v20. Sources at the bottom.

---

## 0. The scoring reality (this reframes everything)

- **RHAE is a 0–100% scale.** Per level: `min(1, h/a)²` where `h` = human baseline
  actions, `a` = agent actions. Aggregated: level scores → environment mean (linearly
  weighted by level depth) → benchmark mean. **Humans = 100%.**
- **The leaderboard is in *percent*.** Tufa's **"1.21" = 1.21%** of human efficiency.
  Frontier models at release scored **0.00–0.37%**. So the *entire field is under ~2%*;
  this is a brutally hard benchmark.
- **Quadratic penalty + hard 5× cap.** 10× human actions = **1%** (not 10%); >5× human
  actions on a level = **attempt terminated**. ⇒ **action efficiency is the whole game.
  Optimal (shortest) solutions matter enormously** — a 2× longer solve scores 25%.
- The official ARC-AGI-3 eval is **BLACK-BOX**: agents get **frames via the SDK/API**,
  **not source code**. Private game sources are "tightly guarded," not shipped.

### ⚠️ The #1 strategic question for us: is *our Kaggle* white-box or black-box?
Our whole premise ([[v18-blackbox-pivot]]) is "Kaggle ships sources → white-box BFS wins
(v12 = 0.22)." But the *official* benchmark above is black-box. Two readings:
- **If our Kaggle competition ships the scored games' sources (white-box):** BFS reading
  the real source is a *near-perfect world model* — we are playing an easier game than the
  official black-box leaders, and our ceiling is far above 1.21% (their handicap is having
  to *infer* the dynamics we can just *read*). Then the gap is **plumbing + optimal-RHAE +
  coverage**, not approach. v12's 0.22 vs Tufa's 1.21 is **not apples-to-apples**.
- **If our Kaggle hides scored sources (black-box, like the official board):** BFS only
  scores where a source happens to ship (the 25 public games); held-out games need the
  black-box methods below, and our 0.22 ≈ frontier-model territory.
- **How we'll know for free:** the pending v19 `v19_run.log` line. `BFS ACTIVE: loaded
  Ls20 from .../ls20.py` for *scored* games ⇒ white-box. `no white-box source` ⇒ black-box.
  **Read that first — it decides v20's whole direction.**

---

## 1. The best documented approaches (all black-box, ranked by score)

| Approach | Score | Mechanism |
|---|---|---|
| **Executable World Models** (coding agent) | **32.58% mean RHAE** on 25 public games; 7 solved 100%, 106/209 levels | A coding agent (Codex CLI + GPT-5.4) **writes Python** encoding its hypothesis of the env (state repr, transition fn, goal check), **verifies** it against recorded frames, **refactors** to simplest form (MDL), **plans** by simulating action sequences in the executable model, then **acts**, halting on prediction↔observation divergence. `world_model_engine.py`/`_state_io.py`/`_main_planner.py`. **No hand-coded game logic.** |
| **Self-learning layer** (BentoLabs) | **1.27% → 3.32% (2.6×)**, +6 levels, 34% cheaper, **no retraining** | A memory loop around an existing agent: **after** a run, extract successful patterns/code/workflows into a curated KB; **before** a run, retrieve by task similarity and inject into context; **during**, a light oversight layer nudges when stuck. Same model, same budget — pure memory/retrieval. |
| **StochasticGoose** (Tufa, preview winner) | **12.58%** preview (3 unseen games), 18 levels | CNN (16-ch 64×64 → 32/64/128/256) predicts which actions change the frame; action-type head + **conv coordinate head** for ACTION6 (keeps 2D bias). Off-policy, hash-deduped 200k buffer, **reset per level**. Targeted exploration, not random. (= our `forge_agent.py` lineage.) |
| **Graph-based exploration** | — | Transition-graph + frontier exploration (Blind Squirrel lineage; what v18/v19 borrow). |
| **Tufa 0.68→1.17 "novel approach"** | **1.17%** (first real ceiling break) | **Undisclosed** — to be open-sourced at competition end. Greg Kamradt: "guessing 1.17% is a novel approach." Watch for the release. |
| **LADDER / TTRL** (Tufa paper) | (their training recipe) | Recursive variant decomposition + GRPO + test-time RL (see [[ladder-stage0-finding]], `LADDER_PLAN.md`). |

---

## 2. What this means for v20

**The RHAE penalty is the headline.** Whatever the stage, **return the SHORTEST solution**,
not just *a* solution — our own finding lines up: v19 `auto` (optimal 13/45/39/43/44) ≫
v13 greedy (63/71/78) for RHAE. So:
- **BFS stage must prefer optimal** (masked BFS / A*), not greedy. Greedy only as a
  last-resort "solve at all."
- Time-box per level against the **5× cap** — never burn the action budget exploring.

**The cascade we built is the right skeleton.** Map the research onto it:
- **Memory →** upgrade from exact-level cache to a **self-learning layer** (BentoLabs):
  store *transferable patterns/macros* keyed by frame-features, retrieve for *similar*
  games, not just identical levels. Retraining-free, 2.6× documented. This is the cheapest
  high-leverage win.
- **BFS (white-box) →** keep first *if our Kaggle is white-box*; force optimal for RHAE.
- **Forge (black-box) →** the CNN change-predictor (12.58%) is dominated by **Executable
  World Models (32.58%)**. The big upgrade: when no source is reachable, **build an
  executable world model from observations** (learn/synthesize the transition fn, verify
  vs frames, plan in sim). This is the strongest documented no-source method and the right
  Forge replacement.
- **LADDER/TTRL →** hardest levels / generalization, as planned.

**Prioritized v20 bets (research-first):**
1. **Confirm white-box vs black-box** from the v19 log — gates everything.
2. **Optimal-RHAE everywhere** (shortest solutions; respect the 5× cap). Cheap, immediate.
3. **Self-learning memory layer** (BentoLabs-style cross-game pattern reuse). Cheap, 2.6×.
4. **Executable-world-model Forge stage** for no-source games (32.58% method). Bigger build.
5. **Watch Tufa's open-source** of the 1.17% novel approach when released.

---

## 3. Honest caveat on the numbers
The 32.58% is on the **public** 25 games (where you can iterate); the **leaderboard
(private)** is ~1.21% — the gap is **generalization to unseen games**. So public-set wins
don't transfer 1:1. The robust v20 is the **hybrid cascade**: read the source when it ships
(BFS, near-perfect), infer an executable world model when it doesn't, reuse learned patterns
across both.

---

## Sources
- ARC-AGI-3 Technical Report — arXiv:2603.24621 (RHAE def, black-box setup, 5× cap)
- Executable World Models for ARC-AGI-3 — arXiv:2605.05138 (32.58% coding-agent)
- BentoLabs, "2.6× higher scores with a self-learning layer" — bentolabs.ai/blog/self-learning-ai-agents-arc-agi
- Dries Smit / Tufa, "1st Place ARC-AGI-3 Agent Preview" + github.com/DriesSmit/ARC3-solution (StochasticGoose, 12.58%)
- Digg: "Tufa Labs raises ARC-AGI-3 from 0.68% to 1.17%" (novel approach, to be open-sourced)
- arcprize.org/competitions/2026/arc-agi-3 ; docs.arcprize.org
