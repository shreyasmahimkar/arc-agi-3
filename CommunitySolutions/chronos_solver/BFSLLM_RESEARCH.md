# BFSLLM — training a model with BFS as its brain

Research note, 2026-06-12. Question: v13's symbolic BFS solves ls20 L0–L4
(13/45/39/43/44 actions) and then dies at L5 with the 500k-state budget
exhausted. Can we take those "learnings" and port them into a model — a
BFSLLM — that solves what raw BFS can't?

## 1. What v13's BFS actually knows (and what it throws away)

The cache files (`v13_bfs_cache_<game>.json`) store only the **final
solutions**: per level, a list of `[action_id, data]`. For ls20 that is
184 actions total across 5 levels. That alone is nowhere near enough to
train anything — behavioral cloning on 184 tokens is a memorization
exercise.

But the solver *computes* far more than it saves. During a solve of one
level, BFS produces:

- every **visited state** with its exact depth-from-start (the visited
  hash set in `_bfs_search_fifo`),
- the **expansion order** (which nodes were tried, which were dead ends),
- the **transient-pixel mask** (what part of the frame is HUD vs. game),
- the **effective action set** per game (`_scan_actions`: which of the
  4096 clicks + 5 buttons actually do anything),
- **hidden scalar fields** that matter for state identity (`_probe_hidden_fields`),
- the chained **true start state** per level (`_make_start_state`).

That discarded search tree is the real training set. And because the 19
solved games' engines live in `arc-prize-2026-arc-agi-3/environment_files/`
(e.g. `ls20/9607627b/ls20.py`), regenerating it fully instrumented is
free — same trick v14's `gen_data.py` already uses.

## 2. Four ways to put BFS in a model's brain (literature)

**(a) Behavioral cloning of solutions.** Train on (state → next expert
action). Weakest option: tiny data, no notion of *why*, brittle off the
demonstrated path. Baseline only.

**(b) Procedure cloning / search-trace cloning.** Train the model to
imitate the *computation*, not just the answer.

- *Procedure cloning* (Yang et al., NeurIPS 2022) showed that
  autoregressively imitating the expert's intermediate computations (BFS
  expansions included — their maze experiments literally clone BFS)
  generalizes to configurations where plain behavioral cloning fails.
- *Searchformer* (Meta, 2024) trains a transformer on **A\* execution
  traces** (tokens for node create/expand/close), then fine-tunes with
  expert iteration. Result: solves unseen Sokoban 93.7% optimally with
  fewer search steps than the A\* teacher, and beats solution-only
  training with 5–10× smaller models and 10× less data. The trace is the
  signal.
- *Dualformer* (2024 follow-up) randomly drops trace segments during
  training → the model learns both "fast" (answer-only) and "slow"
  (trace) modes.
- *Stream of Search* (Gandhi et al., 2024) flattens search — including
  **mistakes and backtracking** — into one string, pretrains on it, then
  self-improves (STaR/expert iteration). SoS models solved **36% of
  problems the heuristic teacher solvers could never solve** — exactly
  the "L5 hope" here.

**(c) Learned heuristic + keep the symbolic search.** DeepCubeA
(Agostinelli et al., Nature MI 2019) trains a cost-to-go network from
states labeled with distances, then runs **weighted A\*** with that
heuristic — solves Rubik's cube, Sokoban, 15/24/35/48-puzzles, Lights
Out, mostly optimally. Every state BFS visits already carries an exact
depth label; states on the solution path carry exact cost-to-go. This
is the most sample-efficient port of "BFS learnings" and directly
attacks the failure mode at L5: BFS dies of breadth — a learned
heuristic turns it into greedy/A\* search that goes deep.

**(d) Expert iteration (AlphaZero/ExIt).** Loop: model proposes →
search verifies/extends → retrain on the verified successes. This is
the engine that turns (b) or (c) from imitation into improvement, and
it's how Searchformer surpassed its own teacher.

## 3. Concrete pipeline for ls20 (L0–L4 → BFSLLM → L5)

**Phase 0 — instrument the solver (`gen_traces.py`).**
Subclass `BFSSolver` so `_bfs_search_fifo` logs, per expansion:
`(parent_hash, action, child_hash, depth, outcome)` where outcome ∈
{new, visited, dead, level_complete}. Re-solve L0–L4 from the local
engine. Also dump each state's frame once (keyed by hash). One level's
tree = tens-of-thousands of labeled transitions instead of ~40 actions.

**Phase 1 — serialization.** 64×64 raw pixels × thousands of states
won't fit an LLM context. Serialize **object-centric deltas**, not
frames: reuse v14's `ObjectChannels` (connected components) to write a
state as an object list (`obj color=4 bbox=12,3,15,6 n=9 ...`), and a
transition as a delta (`moved obj#3 +2,0; vanished obj#7`). Trace
vocabulary (SoS-style):

```
LEVEL 2  STATE <objects…>
TRY a2 → delta…   TRY a3 → VISITED   TRY a6 x=14 y=22 → delta…
DEADEND BACK
… GOAL plan: a2 a2 a3 a6(14,22) …
```

**Phase 2 — train.** Two tracks, same data:

- *Track A (heuristic, cheapest, do first):* small net (CNN or the v14
  encoder) regressing depth-to-goal from frames on solution paths +
  visited-set negatives. Plug into the existing solver as a priority
  function — `strategy='greedy'` already exists, it just needs a brain.
  Pure PyTorch, Kaggle-legal, no LLM at all.
- *Track B (BFSLLM proper):* decoder-only transformer (v14's 30–80M
  budget, or LoRA on a small open model) trained on flattened traces.
  **Critical: train on all 19 cached games, not just ls20.** Five levels
  of one game will be memorized; the transferable thing is the *shape*
  of search — try, observe delta, prune visited, backtrack. Dualformer
  trick: drop trace chunks randomly so it can also answer plan-only.

**Phase 3 — eval gate (honesty first).** Hold out one solved level:
train on ls20 L0,L1,L2,L4 (+ other games), test whether the model
solves L3 — a level whose answer exists but was never shown. If that
fails, L5 is fantasy. This mirrors v14's held-out-game gate.

**Phase 4 — attack L5 with expert iteration.** Model proposes top-k
actions per state; the **real engine verifies** (it's local — perfect
verifier, no hallucination risk); any deeper-than-before progress gets
appended to the training traces; retrain; repeat. This is SoS/ExIt with
BFS as fallback expert. Realistic expectation: the win comes from
learned pruning (search depth ~45+ at L5 is breadth-death for BFS, but
a model that has internalized "keys open locks, don't revisit, this
delta is progress" searches a sliver of that space).

## 4. Relation to v14 (don't build a competitor, build the missing half)

v14's PLM already does "BFS in imagination": `planner.py` runs latent
BFS over a learned world model. But its planner is *uninformed* — brute
beam over top-k actions. BFSLLM supplies exactly what it lacks:

- Track A's cost-to-go net = a **value head** for the latent planner.
- Track B's trace model = a **policy prior** over actions (the ExIt
  apprentice biasing the search).

That is the full AlphaZero triad: v14 world model + BFSLLM policy/value
+ search. The v13 integrity line holds: caches and engines are offline
training data only; what ships to the hidden eval is a general prior.

## 5. Recommended order

1. `gen_traces.py` — instrumented re-solve of the 19 cached games (a
   weekend of CPU, all local).
2. Track A heuristic + greedy strategy in the v13 solver → immediate
   shot at ls20 L5 with zero LLM work.
3. Track B trace transformer with the L3-holdout gate.
4. Expert-iteration loop on L5 / the other stuck levels (vc33 L4,
   ar25 L2, …).
5. If 2–4 work, wire policy/value into v14's latent planner.

## Sources

- Searchformer — Beyond A*: Better Planning with Transformers via
  Search Dynamics Bootstrapping: https://arxiv.org/abs/2402.14083
  (code: https://github.com/facebookresearch/searchformer)
- Dualformer — Controllable Fast and Slow Thinking by Learning with
  Randomized Reasoning Traces: https://arxiv.org/pdf/2410.09918
- Stream of Search (SoS) — Learning to Search in Language:
  https://arxiv.org/abs/2404.03683
- Chain of Thought Imitation with Procedure Cloning:
  https://arxiv.org/abs/2205.10816
- DeepCubeA — Solving the Rubik's cube with deep reinforcement learning
  and search: https://openreview.net/pdf?id=rNdmaQqWwn2
- A* Search Without Expansions (DeepCubeAQ): https://arxiv.org/pdf/2102.04518
- Expert Iteration background (BRExIt survey of ExIt):
  https://arxiv.org/pdf/2206.00113
