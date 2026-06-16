# Puzzle-LM — iterations 2 & 3 (+ 200-games and agentic-swarm research)

## Iteration 2 — imagination rollouts in the MCTS

Built world-model **imagination rollouts**: at an MCTS leaf, instead of rolling
out in the render-bound real engine, roll forward in the Puzzle-LM world model
(feature space, no render): `feat -> wm.next(feat, policy(feat)) -> ...`,
accumulating predicted progress as a value lookahead (`mcts.solve_mcts_az(...,
world_model=, wm_policy=)`).

Measured on cd82 (the most promising game):

| config | progress | sims | sim/s |
|---|---|---|---|
| real micro-rollout (4) + imagination bonus | 7 | 1834 | 102 |
| **pure imagination value (micro=0)** | **5** | 1808 | 100 |

**Finding: imagination didn't help, and removing the real rollout *hurt*.** Two
reasons, both important:
1. **Descent/expansion dominates cost, not the rollout.** Each simulation's real
   cost is the PUCT descent + expansion (real engine steps with macro), which is
   far longer than a 4-step rollout — so replacing the rollout with imagination
   barely changes sim/s (102 → 100). The render cap is in the *tree*, not the leaf.
2. **Real rollouts *discover* progress; imagination only *estimates* it.** The
   v1 world model (37% better than copy, but still approximate) can't predict the
   exact key/lock event, so pure-imagination value lost the progress-7 path
   (dropped to 5). The real engine's playout was actually finding events.

Imagination is the *right architecture* — it's the only thing that breaks the
render cap and enables a GPU-batched swarm — but it pays off only with a strong,
high-fidelity world model. On CPU with a lossy v1 model it's a wash-to-negative.

## Iteration 3 — bigger pool, retrain

Tripled the pool (5,025 → **17,608 transitions**, 6,740 with progress) and
retrained the world model (hidden 160) + apprentice.

| metric | v1 (5k) | v3 (17.6k) |
|---|---|---|
| world-model: % better than copy | 37.4% | 35–39% |
| cross-game policy acc | 0.37 | **0.29** (worse) |
| cd82 progress (re-benchmark) | 7 | 7 (no change) |

**Finding: 3× the data did NOT strengthen the model.** The world model's edge
over copy is flat, and the cross-game policy *degraded* with more games (more
conflicting "right actions" seen through a lossy summary). **The bottleneck is the
representation — the 76-d object-feature summary — not data quantity.** A summary
of per-colour counts/centroids throws away the spatial detail (which wall, which
door, exact adjacency) that dynamics and policy actually depend on.

### The clear conclusion for iteration 4

Two changes, in order:
1. **A real spatial tokenizer (v15's codebook over 8×8 patches)** instead of the
   76-d summary, so the world model and policy see actual geometry. This is the
   single highest-leverage change — the data is already proving the summary is the
   ceiling.
2. **Then** the 200 community games for diversity (below) — diversity only helps
   once the representation can use it.

---

## Does the Kaggle "200+ games" testbed help? YES — but representation first.

`https://www.kaggle.com/code/poonszesen/arc-agi-3-interactive-testbed-200-games`

What I confirmed by research:
- The 200+ games are **handcrafted ARC-AGI-3 environments on the same
  `arc-agi` / `ARCBaseGame` engine contract** as the 25 official games (the same
  `is_done` / `choose_action` toolkit). So they load with our existing
  `engine.py` with a ~5-line change (point `game_py_path` at a second env dir) —
  no new harness.
- **They are the data-diversity lever this project keeps needing.** Our pool is 25
  games; 200+ is ~8× the dynamics diversity (Sokoban, flood-fill, slide, memory,
  mirror, rule-switch…). Iteration 3 showed *quantity* of the same 25 games
  doesn't help — but *diversity* of mechanics is a different axis, and it's
  exactly what a cross-game world model needs to generalise.
- **Caveats (from the competition rules):** the private eval is sandboxed with no
  internet and solutions must be MIT-0/CC0; community games are **training data
  only** — the official-games holdout is the only score that counts (keep them
  held out). Quality varies, so filter for solvability before pooling.
- **Practical blocker here:** this sandbox has no Kaggle/internet access and the
  `arc-agi` package isn't installable, so I can't vendor the 200 games in-session.
  To use them: `git clone` the arc-interactive repo (or download the Kaggle
  dataset) into `environment_files/`, then re-run `gen_pooled.py --games <list>`.

**Verdict:** worth it, and cheap to wire in — but sequence it *after* the
tokenizer, because diversity through a lossy 76-d summary won't land (iter 3 is
the evidence).

---

## Agentic frameworks / swarms — 10+, and which actually fit this problem

Frameworks surveyed (2026): **LangGraph** (stateful graph + checkpointing),
**CrewAI** (role-based teams), **AutoGen** (conversational GroupChat, async),
**OpenAI Agents SDK** (ex-Swarm; explicit handoffs), **Google ADK** (hierarchical
agent tree), **Claude Agent SDK** (tool-use + subagents — what this run uses),
**Microsoft Semantic Kernel** (planners + plugins), **LlamaIndex Agents**,
**Camel-AI** (role-play cooperation), **MetaGPT** (SOP-driven teams), **Swarms /
kyegomez** (explicit swarm orchestration), **AgentVerse** (emergent multi-agent).

**The honest mapping to ls20 / the 25 games:** the bottleneck is **search over a
perfect, fast simulator — not LLM reasoning.** So an *LLM-chat* swarm is the wrong
tool for the core solve (it adds latency/cost without adding search throughput).
The swarm that helps is two-layered:

1. **Compute swarm (the workhorse) — NOT an LLM framework.** N parallel MCTS
   workers with diversified seeds/landmarks (portfolio + the landmark fan-out that
   already works in `mcts_waypoint`). This is **Ray / multiprocessing**, and on a
   GPU it becomes "a batch of N imagined states in one forward pass" — the real
   100-way swarm. This is where the wins are.
2. **LLM strategist layer (thin, optional) — where a framework fits.** 3–15 LLM
   agents that *propose* subgoals/rule-hypotheses ("avatar must match colour 9
   then 8") which the search swarm **verifies** on the real engine (FunSearch /
   AlphaProof pattern). For *this* layer use a lightweight orchestrator —
   **LangGraph** (stateful, checkpointable, model-agnostic) or the **Claude Agent
   SDK** (subagents) are the best fits; CrewAI if you want role DSL speed.

**Recommendation:** don't build an LLM-swarm for the solver. Build a Ray/MP MCTS
**portfolio** (cheap, high-leverage), and bolt on a *small* LangGraph/Claude-SDK
strategist layer only as the apprentice-proposer in the ExIt loop. Reserve the
heavy 100-way parallelism for the GPU world-model imagination search, where a
"swarm" is literally a batched forward pass.

---

## Status

Puzzle-LM is a working cross-game world model (35–39% better than copy) + policy/
value apprentice, with imagination rollouts wired into the MCTS. It does **not yet
lift level-solving** — and iterations 2–3 pinpoint exactly why (representation, not
data or search), which sets a sharp iteration-4 agenda: **patch/codebook tokenizer
→ vendor the 200 games → GPU imagination swarm.**

Files: `puzzlelm.py`, `gen_pooled.py`, `mcts.py` (imagination rollout),
`models/pool.npz` (17.6k), `models/puzzle_wm.npz`, `models/puzzle_trm.npz`.
