# v17 — iteration log (auto-generated, 25 iterations)

Target: **ls20 L5** — the dual-key level. v13 solved L0–L4 (13/45/39/43/44 actions) and breadth-died on L5 (~80k states).

L0–L4 are re-verified through the REAL engine every iteration (`lc=5` means all five replay correctly; no self-reported wins).

`depth` = deepest action sequence searched. `prog` = v17 progress signal = # of engine scalar-attrs (key colour/rotation, goal-match flags) moved vs level start; the dual-key WIN needs the full set.

| iter | approach | L0–L4 | RHAE | L5 status | depth | prog | expansions |
|---|---|---|---|---|---|---|---|
| 1 | BFS baseline | lc=5 | 5.0 | TIMEOUT | 8 | - | 253 |
| 2 | trace+dataset | lc=5 | 5.0 | BUDGET | 6 | - | 80 |
| 3 | ForgeNet A* | lc=5 | 5.0 | TIMEOUT | 38 | - | 212 |
| 4 | TRM PUCT | lc=5 | 5.0 | BUDGET | 172 | - | 500 |
| 5 | ForgeNet+TRM+ExIt | lc=5 | 5.0 | TIMEOUT | 39 | - | 373 |
| 6 | sweep pw=4 | lc=5 | 5.0 | TIMEOUT | 39 | 1 | 182 |
| 7 | sweep pw=8 | lc=5 | 5.0 | TIMEOUT | 39 | 1 | 137 |
| 8 | sweep pw=16 | lc=5 | 5.0 | TIMEOUT | 35 | 1 | 131 |
| 9 | greedy+progress | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 135 |
| 10 | astar w=3 pw=8 | lc=5 | 5.0 | TIMEOUT | 39 | 1 | 134 |
| 11 | astar+policy pw=12 | lc=5 | 5.0 | TIMEOUT | 39 | 1 | 130 |
| 12 | ExIt round 1 | lc=5 | 5.0 | TIMEOUT | 39 | 1 | 163 |
| 13 | ExIt round 2 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 198 |
| 14 | ExIt round 3 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 277 |
| 15 | ExIt round 4 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 214 |
| 16 | ExIt round 5 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 206 |
| 17 | ExIt round 6 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 207 |
| 18 | ExIt round 7 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 293 |
| 19 | ExIt round 8 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 311 |
| 20 | scaled push 1 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 407 |
| 21 | scaled push 2 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 609 |
| 22 | scaled push 3 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 527 |
| 23 | scaled push 4 | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 438 |
| 24 | scaled push 5 | lc=5 | 5.0 | TIMEOUT | 43 | 2 | 587 |
| 25 | final attack | lc=5 | 5.0 | TIMEOUT | 43 | 1 | 659 |
| 26 | Subgoal Search (waypoint re-root) | lc=5 | 5.0 | TIMEOUT | 44 | 2 | 427 |
| 27 | BFWS novelty (Best-First Width) | lc=5 | 5.0 | TIMEOUT | 43 | 2 | 450 |
| 28 | Macro-actions (move-until-wall) + BFWS | lc=5 | 5.0 | TIMEOUT | 230 | 2 | 431 |
| 29 | Type-based explore + waypoint + novelty + macro | lc=5 | 5.0 | TIMEOUT | 149 | 2 | 185 |
| 30 | Full-stack final attack | lc=5 | 5.0 | TIMEOUT | 117 | 2 | 257 |
| 31 | MP greedy+macro (4 workers) | lc=5 | 5.0 | BUDGET | 136 | 2 | 701 |
| 32 | MP greedy+macro big-batch | lc=5 | 5.0 | BUDGET | 204 | 2 | 904 |
| 33 | MP waypoint (landmark re-root on cores) | lc=5 | 5.0 | TIMEOUT | 129 | 2 | 294 |
| 34 | MCTS forward-rollout (UCT+heavy playout) | lc=5 | 5.0 | TIMEOUT | 130 | 2 | 336 |
| 35 | MCTS short rollout (more sims) | lc=5 | 5.0 | TIMEOUT | 100 | 3 | 493 |
| 36 | MCTS high exploration (c=2.5) | lc=5 | 5.0 | TIMEOUT | 136 | 3 | 440 |
| 37 | MCTS progress-bonus 20 | lc=5 | 5.0 | TIMEOUT | 108 | 3 | 436 |
| 38 | MCTS waypoint re-root | lc=5 | 5.0 | TIMEOUT | 103 | 3 | 624 |
| 39 | MCTS waypoint deep | lc=5 | 5.0 | TIMEOUT | 124 | 3 | 639 |
| 40 | MCTS final big-budget attack | lc=5 | 5.0 | TIMEOUT | 108 | 3 | 591 |

## Iterations 26–30 — one web-researched technique each

| iter | technique | paper | effect on L5 |
|---|---|---|---|
| 26 | Subgoal Search / landmark re-root | Czechowski et al. 2021; HIGL (Kim et al. 2021) | reliably chains progress 1→2 by re-rooting at a landmark |
| 27 | Best-First Width Search (novelty bonus) | Lipovetzky & Geffner 2012/2017 | reaches progress 2 in a single phase (greedy stalled at 1) |
| 28 | Macro-actions (move-until-wall) | Botea et al.; options framework | one expansion covers depth ~230; progress 2 |
| 29 | Type-based exploration (multi-queue) | Xie, Müller, Holte, Imai 2014 | progress 2 in 85 expansions (vs 2 rounds before) |
| 30 | full stack (all of the above) | — | progress 2 reliably; progress 3 is the new wall |

## Iterations 31–40 — throughput (multiprocessing, then imagination/MCTS)

| iter | technique | paper | effect on L5 |
|---|---|---|---|
| 31–32 | multiprocess best-first (4 cores) | v13 expander | ~2.4× nodes/s (28 vs 12); progress 2 |
| 33 | MP landmark re-root on cores | — | progress 2, depth 129 |
| 34 | **forward-rollout MCTS** (UCT + heavy playout) | Browne et al. 2012 | 60,971 states in 30s (~2,000/s, ~12× best-first); progress 2 |
| 35 | MCTS short rollout / more sims | light-playout MCTS | **progress 3 — first break past 2 in 35 iters** |
| 36 | MCTS high exploration (c=2.5) | UCT | progress 3 |
| 37 | MCTS progress-bonus 20 | reward shaping | progress 3 |
| 38–39 | MCTS waypoint re-root | Subgoal Search + MCTS | progress 3 |
| 40 | MCTS big-budget (20k sims) | — | progress 3 (wall now 3→4 = the win) |

**Key engine finding:** profiling showed `perform_action` is render-bound (~0.4–0.85 ms), so the real engine ceilings at **~2,000 states/sec for ANY method**. MCTS wins by spending that budget without per-node snapshots (it restores once per simulation and rolls forward), exploring ~12× more states than best-first — which is what cracked progress 2→3. Going beyond the render cap needs a *learned* world model (v15) that is GPU-batchable.

## Reading the trajectory

* **1 → 3**: plain BFS depth 8 → ForgeNet-A\* depth 38. The learned cost-to-go converts breadth-death into a goal-directed dive (200× fewer expansions than v13's breadth search).
* **4**: TRM-PUCT depth 172 but unguided — policy overfit to 184 expert states drives one corridor. Cautionary: the value/policy prior must *reorder*, not *dominate*.
* **6–11**: progress-shaping sweep. `greedy+progress` (iter 9) is the deepest single-shot at **depth 43** — pure heuristic+progress, no depth penalty.
* **12–19**: expert iteration (SoS/ExIt). Harvest the most-progressed path → bootstrap cost-to-go labels → retrain ForgeNet → re-search. Holds depth 43 with fewer wasted expansions as the heuristic learns the progressed region.
* **20–25**: scaled budgets. Iter **24 reaches progress 2** — the first time the search engages the *second* key of the dual-key lock. The ceiling is budget-reachable, not architectural.

See `logs/iterN.log` for the full per-iteration trace.