# Puzzle-LM — iteration 1 (cross-game world model + policy/value, v15-style)

Goal: replace the ls20-only ForgeNet/TRM (which don't transfer) with ONE model
trained on transitions pooled from **all 25 games**, then use it as the
apprentice inside the value-MCTS — the v15 idea, cross-game.

## What was built (all pure numpy, runs on the Mac/sandbox)

1. **`gen_pooled.py`** — pooled transitions from all 25 games:
   **5,025 transitions** (1,965 with a progress event, 43 wins), in the shared
   object-feature language (`engine.object_features`, 76-d). Sources: v13
   expert-solution replay + random/macro exploration from RESET.
2. **`puzzlelm.py` WorldModel** — `(state_features, action) -> next_state_features
   (+ P(progress))`, predicting the residual delta on a copy of the current state
   (the v15 "copy is default, learn the delta" lesson).
3. **Cross-game TRM apprentice** — the recursive policy/value core retrained on the
   pool: policy prior = "the action that makes progress here", value = "is this a
   progress-promising state", across all games.

## Results

### The world model works — it learned cross-game dynamics

| metric | value |
|---|---|
| next-state MSE | **0.00061** |
| copy baseline (predict no change) | 0.00097 |
| **improvement over copy** | **37.4%** |
| progress-event prediction acc | 99.5% |

This is the v15 success metric: the model is meaningfully better than the copy
baseline, i.e. it predicts the *deltas* (avatar moves, object changes) — and it
does so from data pooled across 25 different games. That's a genuine cross-game
dynamics model in v1.

### The apprentice is weak, and it did NOT lift the search yet

- Cross-game policy accuracy **0.37** (5 classes, chance 0.20 — real but weak).
- Plugged into value-MCTS and re-ran the benchmark on the 6 most promising games.
  Progress reached, **Puzzle-LM prior vs no-model baseline** (6 s/game):

  | game | baseline | Puzzle-LM |
  |---|---|---|
  | cd82 | 7 | 7 |
  | sc25 | 2 | 2 |
  | sp80 | 2 | 2 |
  | r11l | 2 | 1 |
  | tr87 | 2 | 2 |
  | su15 | 2 | 2 |

  **No lift.** The prior is roughly even with uniform exploration.

## Honest learnings (the gate for iteration 2)

1. **The world model is the real v1 win** — 37% better than copy across 25 games
   proves the cross-game-dynamics idea is sound and learnable even from a tiny,
   shallow pool. This is the foundation to build on.
2. **The apprentice is too weak to help search yet.** Two reasons, both fixable:
   (a) the 76-d object-feature summary is lossy — a per-cell/pixel tokenizer (v15's
   codebook) carries far more; (b) the pool is shallow (random walks + short expert
   replays) — only 5k transitions, ~2k with progress. Policy 0.37 can't out-guide
   random rollouts.
3. **The world model is trained but NOT yet used for what matters** — *imagination
   rollouts*. Right now the search still rolls out in the real (render-bound)
   engine. The whole point of a world model is to roll out in IT (no render, and
   eventually GPU-batchable). v1 only wired in policy/value, not the simulator.

## Iteration 2 plan (decided)

1. **Use the world model for imagination rollouts** inside MCTS: replace the
   real-engine random playout with a WorldModel forward rollout (predict feature
   trajectories + progress), verifying only the committed plan on the real engine.
   This is the throughput unlock the render-cap analysis pointed to.
2. **Stronger state language**: add a small patch/codebook tokenizer (v15-style)
   instead of the 76-d summary, so the world model and policy see real spatial
   detail.
3. **Bigger, better pool**: harvest progress-≥2 trajectories from the MCTS runs
   (cd82=7, etc.) back into the pool (expert iteration across games), and grow
   episodes-per-game.
4. **Re-benchmark** the 25 games and require: Puzzle-LM progress > baseline on the
   6 promising games, and ≥1 extra level cracked (cd82 is the prime target).

Files: `gen_pooled.py`, `puzzlelm.py`, `models/pool.npz`, `models/puzzle_wm.npz`,
`models/puzzle_trm.npz`, `v17_multigame_puzzle.json`.
