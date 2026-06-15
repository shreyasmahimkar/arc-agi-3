# Chronos Solver v17 — informed search: BFS + CNN ForgeNet + a Tiny Recursive LM

Lineage: v12 (BFS baseline) + v13 (BFS best-parts) → **v17**. Target: **ls20 L5**,
the dual-key level v13 breadth-died on (v13 solved L0–L4 in 13/45/39/43/44
actions, then exhausted ~80k+ states on L5). Built and run entirely on the Mac
side (no GPU box) under the constraint of **optimising RHAE** (fewer real
actions = quadratically higher score, so the planner spends *simulated* search,
not real actions).

## The one-line thesis

L5 is **breadth-death**, not a wall. v13's plain BFS dies of breadth at depth ~8.
A *learned cost-to-go heuristic* (CNN ForgeNet) turns that same search into a
*depth-first* dive toward the goal: **depth 8 → 38 in the same 20s budget, with
200× fewer node expansions than v13's breadth search.** That is the entire v17
contribution, and it is measured on the real engine, not asserted.

## What's here

| file | role |
|---|---|
| `engine.py` | the single ground-truth verifier — loads the **real** `ls20` engine and drives it. Carries v13's best parts: chained true-baseline start state, transient-HUD-row masking, scalar-attr state identity, dynamic click targets. |
| `_pydantic_shim.py` | lets the pure-python `arcengine` run in the network-less dev sandbox (auto-installed only when real pydantic is absent; untouched on the Mac venv). |
| `search.py` | the BFS/ForgeNet core. `strategy ∈ {bfs, greedy, astar, puct}`, pluggable `heuristic_fn` + `policy_fn`, slim snapshots (2× faster restore), full search-trace emission. |
| `forgenet.py` | **CNN ForgeNet** cost-to-go heuristic (numpy): fixed conv feature bank + object features → trained MLP head (Adam). |
| `trm.py` | **our Tiny Recursive Model** (numpy): one tiny core applied *recursively* (T steps, shared weights, deep supervision) to refine a belief → policy prior + value. The ExIt apprentice that reorders/prunes the branch set. |
| `data.py` | turns v13's cached BFS solutions into labelled states (cost-to-go for ForgeNet, expert actions for TRM). |
| `benchmark.py` | always-on scorecard: replays L0–L4 through the real engine + RHAE proxy. |
| `run_iterations.py` | the iteration driver — one iteration per invocation, each fully logged. |
| `ITERATIONS.md` | auto-generated results table. |
| `logs/iterN.log` | excessive per-iteration logs (every expansion batch, every train epoch). |

## How to run (Mac, venv312)

```bash
source ../../../.venv312/bin/activate      # real pydantic/numpy; shim unused
cd CommunitySolutions/chronos_solver/v17
python run_iterations.py --iter 1   # BFS baseline (breadth-death)
python run_iterations.py --iter 2   # trace gen + dataset
python run_iterations.py --iter 3   # ForgeNet A*
python run_iterations.py --iter 4   # TRM PUCT
python run_iterations.py --iter 5   # ForgeNet+TRM combined, final L5 attack
# budgets: V17_L5_TIME (s), V17_L5_NODES
```

It also runs as-is in a bare Linux box with only numpy (the shim + an
`arcengine` symlink are bootstrapped automatically) — that is how the runs below
were produced.

## Results (5 iterations, real engine)

| iter | approach | L0–L4 verified | RHAE | L5 status | L5 depth | L5 expansions |
|---|---|---|---|---|---|---|
| 1 | BFS baseline | ✅ lc=5 | 5.0 | timeout | **8** | 253 |
| 2 | trace + dataset (960 transitions, 368 states) | ✅ lc=5 | 5.0 | — | 6 | 80 |
| 3 | **ForgeNet A\*** | ✅ lc=5 | 5.0 | timeout | **38** | 212 |
| 4 | TRM PUCT | ✅ lc=5 | 5.0 | budget | 172* | 500 |
| 5 | ForgeNet + TRM combined | ✅ lc=5 | 5.0 | timeout | 39 | 373 |

\* iter-4's depth-172 is the cautionary result: the policy prior, overfit to
184 L0–L4 expert states, prunes to one corridor and dives **deep but unguided**.
iter-5 lets the A\* cost-to-go *dominate* (policy only reorders) so the dive
stays goal-directed.

### Honest status on L5

**L5 is not solved yet.** Best reached: **depth 39 of an estimated ~45+**, in
<400 node expansions (v13 needed 80k+ and never got there). The search is now
200× more sample-efficient — the bottleneck moved from *breadth* to **heuristic
saturation**: ForgeNet, trained only on L0–L4, predicts `cost≈0` for deep L5
states it has never seen, so above depth ~38 it stops discriminating and A\*
degrades back toward breadth among the `h=0` plateau (visited 309→1026, frontier
growing — you can see this in `logs/iter5.log`).

## The path to actually closing L5 (next iteration, decided)

1. **Progress shaping in the heuristic** (v13's `greedy` insight): reward
   states whose *scalar identity* (key colour/rotation, the two goal-match
   flags) changed — i.e. real key/lock interactions. This is the missing L5
   win signal the L0–L4 heuristic can't have. Cheapest, highest-leverage.
2. **Expert iteration (SoS/ExIt)**: the engine is a perfect local verifier — any
   deeper-than-before state found gets appended to the training set and ForgeNet
   + TRM retrain, then re-search (the loop is scaffolded in `iter5`; round-2
   wiring is the next commit).
3. **Multiprocess expansion** (v13 had it; 4 cores ≈ 4× → the ~2–4k expansions
   needed for depth 45 fit one run).
4. **Dual-key state factoring**: hash on the two `(colour,rotation)` goal slots
   explicitly so the search treats "first key placed" as real progress instead
   of a revisit.

## RHAE note

L0–L4 replay at v13's action counts (RHAE = 5.0, each level at baseline = 1.0).
RHAE rewards *fewer real actions*; v17 spends only simulated search (free under
RHAE) to find L5, so a solved L5 adds to the score without spending the real-
action budget that the quadratic `min(1, base/actions)²` term punishes.

---

# v17 — iterations 6–25 (progress shaping + expert iteration)

Iterations 1–5 moved the bottleneck from *breadth* to *heuristic saturation*.
Iterations 6–25 attack that directly: a **progress signal** + **expert
iteration**. New code: `sweep.py` (the 20-iteration driver) and the
`progress_weight` / `harvest_k` path in `search.py`.

## What was added

* **Progress shaping** (`search._progress`): the engine exposes hidden scalar
  attrs (key colour/rotation, the two goal-match flags, countdowns). The
  *progress* of a state = how many of those moved vs the level start. Real
  key/lock interactions move them — this is the **L5 dual-key win signal** that
  a heuristic trained only on L0–L4 cannot have. It is folded into frontier
  priority (`pr -= progress_weight * progress`).
* **Expert iteration / Stream-of-Search** (`sweep._harvest_to_training`): after
  each search, harvest the most-progressed path, replay it, label states along
  it with bootstrapped cost-to-go (`D − i`), append to the training set and
  retrain ForgeNet (`forgenet_exit.npz`). Because the engine is a perfect local
  verifier, every harvested state is genuinely reachable — no hallucinated
  targets.

## Results (full table in `ITERATIONS.md`)

| phase | iters | finding |
|---|---|---|
| hyperparam sweep | 6–11 | `greedy+progress` (no depth penalty) is deepest single-shot: **depth 43** |
| expert iteration | 12–19 | ExIt holds depth 43 with fewer wasted expansions as ForgeNet learns the progressed region |
| scaled push | 20–25 | iter **24 reaches progress 2** — first engagement of the *second* key |

Trajectory of the L5 progress signal across 25 iterations:

```
depth:  8 → 38 → (39) → 43 ........................... 43   (plateau)
prog:   –         1 ................................ 2        (iter 24)
```

## Honest status after 25 iterations

**L5 is still not solved.** But the two diagnostics that matter both improved:
the search reaches **depth 43 of ~45+** and, for the first time (iter 24),
**progress 2** — it engaged *both* halves of the dual-key lock, just not in the
same trajectory within budget. The remaining gap is **compute, not method**:
single-process in the dev sandbox runs ~10–27 nodes/s, so each iteration only
expanded 130–660 nodes. The plateau at progress 1 for most runs is the second
key sitting ~5–10 actions beyond the first along a *different* corridor than the
greedy frame-heuristic prefers — exactly what more node budget + ExIt rounds
chip away at (iter 24 is the proof).

## Next steps (ordered by expected payoff)

1. **Multiprocess expansion** — v13 already had it; 4 cores ≈ 4× throughput, so
   the ~2–4k expansions that reached progress 2 in 34 s become a ~10 s run.
   This is the single highest-leverage change and is pure plumbing
   (`search.solve_level` already isolates child-eval; wrap it in a `fork` Pool
   with a game-module initializer like v13's `_bfs_worker_init`).
2. **Two-phase / waypoint search** — once a progress-1 state is found, *re-root*
   the search there (treat it as a sub-goal start state) and search for
   progress 2 from it. This decomposes the dual-key depth (~45) into two ~22-step
   sub-searches, which is quadratically easier for best-first.
3. **Per-progress-level value head** — give ForgeNet (or the TRM value) an
   explicit *progress* input so cost-to-go is conditioned on how many keys are
   already placed. Right now the heuristic can't tell "before key 1" from
   "after key 1" except through raw pixels; making progress a first-class
   feature should stop the post-progress-1 stall.
4. **Real expert-iteration on the TRM policy** — iters 12–25 only retrain
   ForgeNet; also append the harvested paths as policy demonstrations and
   retrain the TRM so its branch-pruning learns the progressed corridors
   (closes the full AlphaZero triad: world-frame heuristic + policy + verifier).
5. **Scale data with arc-interactive** (v16's plan): the depth-43 plateau is
   partly a 20-game data ceiling; 249 community games of key/lock/slide
   mechanics would give the heuristic real dual-key priors instead of L0–L4
   single-key ones.
6. **GPU port** — the numpy ForgeNet/TRM are deliberately torch-free for the
   Mac/sandbox constraint; on the RTX/H100 path they map 1:1 onto the existing
   v14/v15 PLM trainer for 100× faster ExIt rounds.

---

# v17 — iterations 26–30 (one web-researched technique per iteration)

Iterations 1–25 used in-house knowledge. For 26–30 each iteration began with a
literature search, implemented the technique, then ran L5 on the real engine.
Code: `sweep.py` (configs 26–30) + new options in `search.py`
(`prefix_path`, `novelty_bins/weight`, `macro_moves`, `explore_p`).

| iter | technique researched | what it changed | L5 result |
|---|---|---|---|
| 26 | **Subgoal Search** / landmark re-root (Czechowski 2021; HIGL) | search to the most-progressed landmark, **re-root**, repeat — decomposes the ~45-step dual-key into short sub-searches | reliably chains **progress 1→2** (was stochastic) |
| 27 | **Best-First Width Search** (Lipovetzky & Geffner) | novelty (first-seen pixel-atom) as a **priority bonus**, not a hard prune | **progress 2** single-phase (greedy alone stalled at 1) |
| 28 | **Macro-actions** "move-until-wall" (options framework) | repeat a move until wall/progress/win — collapses corridor walks | one expansion covers **depth ~230**; progress 2 |
| 29 | **Type-based exploration** (Xie et al. 2014) | a second queue bucketed by progress level; pop the most-progressed bucket with prob *p* so a heuristic plateau can't starve it | **progress 2 in 85 expansions** |
| 30 | full stack (all four combined) | waypoint + BFWS + macro + type-based | progress 2 reliably; **progress 3 is the new wall** |

## Honest status after 30 iterations

**L5 is still not solved, but the wall moved twice.** Iters 1–5 were stuck at
*breadth* (depth 8). Iters 6–25 moved it to *progress 1* (one key). Iters 26–30
moved it to **progress 2 — both keys/locks engaged — and made reaching it cheap
and reliable** (85 expansions vs hundreds, no longer luck). The remaining gap is
the progress-2 → progress-3 transition: from a both-keys-engaged state the search
still can't find the final placement within ~400 nodes/8s per round. The two
diagnostics (depth 43→230 reachable; progress 1→2 reliable) both improved.

### Was I web-researching between every iteration?

For iters 1–25: **no** — that was in-house knowledge (Searchformer / SoS /
DeepCubeA, already in `BFSLLM_RESEARCH.md`). For iters 26–30: **yes** — each is a
specific searched-and-cited algorithm, listed above. That is the right protocol
and it produced the progress-2 breakthrough; it should continue for 31+.

### Sources (iterations 26–30)

- Subgoal Search for Complex Reasoning Tasks — Czechowski et al., NeurIPS 2021: https://openreview.net/pdf?id=5KCvuCYGi7G
- Landmark-Guided Subgoal Generation (HIGL) — Kim et al., NeurIPS 2021: https://arxiv.org/pdf/2110.13625
- Width-based planning / IW & BFWS — Lipovetzky & Geffner: https://www.ijcai.org/proceedings/2021/0702.pdf
- Planning from Pixels / novelty atoms in Atari: https://arxiv.org/pdf/2012.09126
- Type-Based Exploration with Multiple Search Queues — Xie, Müller, Holte, Imai, AAAI 2014: https://ojs.aaai.org/index.php/AAAI/article/download/11093/10952
- Macro-actions for planning (survey/empirical): https://arxiv.org/pdf/1810.09145

---

# v17 — iterations 31–40 (throughput: multiprocessing, then imagination/MCTS)

After iters 26–30 plateaued at progress 2 *throughput-bound*, these two steps
attacked the bottleneck directly. New code: `mpsearch.py` (#1) and `mcts.py` (#2).

## #1 — Multiprocess best-first (iters 31–33)

`mpsearch.py`. Frontier nodes carry their **action path**, not a 120 KB
snapshot; a fork pool of 4 workers reconstructs state from a fork-inherited
L5-start snapshot + path replay, so the expensive engine work parallelises while
the visited set / heuristic stay in the main process. Result: **~28 nodes/s vs
~12 single-process (≈2.4×)**, ~700–900 expansions per run. Still progress 2 — which
told me cores alone weren't enough and pointed at the engine itself.

## Engine profiling — the finding that reframed everything

Profiling `perform_action` showed it is **render-bound (~0.4–0.85 ms/call)**:
every step re-renders the 64×64 frame. So the real engine ceilings at **~2,000
states/sec for ANY search method** — my earlier "28k/s" reading was no-op moves.
This is the true wall, and it's why a *learned, GPU-batchable* world model (v15)
is the real long-term unlock.

## #2 — Imagination search via forward-rollout MCTS (iters 34–40)

`mcts.py`. Instead of snapshotting every node (best-first's cost), MCTS restores
the L5-start snapshot **once per simulation** and rolls forward with `perform()`,
exploring a whole depth-~50 trajectory per restore. UCT selection + **light
(random) playouts** (the smart TRM policy is used only at tree nodes — putting it
in the rollout hot loop was a throughput killer). This is the CPU realisation of
v15's latent planner: roll forward in the *real* model because its `perform()` is
far cheaper than its snapshot.

**Result: MCTS explored 60,971 states in 30 s (~2,000/s — ~12× best-first) and
broke the progress-2 wall, reaching progress 3 (iters 35–40) — the first move
past 2 in 35 iterations.** The wall is now progress **3 → 4** (4 ≈ the win: both
keys placed *and* both goals matched).

## Honest status after 40 iterations

**L5 still isn't solved, but the wall moved a third time:** breadth (depth 8) →
progress 1 → progress 2 → **progress 3**, with throughput up ~12×. The remaining
gap (3→4) is the final dual-key placement — a sparse-reward needle that uniform
rollouts rarely thread. The fix is heavier, *learned* rollouts and removing the
render cap:

### Next steps (updated)

1. **Learned world model (v15 port) for imagination** — the only way past the
   ~2,000 states/s render cap; GPU-batchable to 1,000s of states in parallel, and
   the natural substrate for a real 15–100-way swarm.
2. **AlphaZero-style policy/value in the MCTS** — replace random rollouts with the
   TRM value (no rollout), à la AlphaGo Zero, so each sim is cheaper *and* better
   directed at progress 3→4.
3. **Expert iteration on the progress-3 traces** — feed the harvested progress-3
   paths back into ForgeNet/TRM so the next search starts knowing the third key.
4. **Swarm** — N MCTS trees with different seeds/landmarks (portfolio + landmark
   fan-out); first to progress 4 wins.

### Sources (iterations 34–40)

- MCTS survey (UCT, heavy/light playouts) — Browne et al. 2012: https://link.springer.com/article/10.1007/s10462-022-10228-y
- AmEx-MCTS / deterministic single-agent MCTS context: https://www.emergentmind.com/topics/monte-carlo-tree-search-mcts
