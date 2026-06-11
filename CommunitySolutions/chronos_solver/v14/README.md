# Chronos Solver v14 — PLM (Puzzle Language Model)

A tiny recursive world model for ARC-AGI-3, built on the v13 harness.
Pure deep learning, pure PyTorch. No LLM, no JAX, no exotic deps —
everything must ship as offline-installable code on Kaggle.

## Why this architecture (one paragraph of honesty)

v13's symbolic BFS is unbeatable when it can load the game engine and
simulate. On Kaggle's hidden eval it cannot — there the agent must learn
the rules by interacting, which is exactly a world-model problem. We also
hold a unique training asset: 25 local game engines that generate unlimited
labeled transitions (S, A → S', R) for free. v14 pre-trains a small world
model on those engines, then at test time runs the classic Dreamer loop:
explore to reduce prediction error ("stochastic goose"), update a recursive
belief state, plan by latent BFS inside the model's imagination, act.

## Architecture (maps 1:1 to `plm/` modules)

```
[ ENV 64x64 grid ]                                [ execute action ]
        │                                                 ▲
        ▼                                                 │
┌─ plm/encoder.py ──────────────┐        ┌─ plm/planner.py ───────────┐
│ ObjectChannels: connected-    │        │ Latent BFS over imagined   │
│  component ids via BFS (the   │        │ futures: top-K actions ×   │
│  v13 trick, now an input      │        │ depth D, batched on GPU,   │
│  channel)                     │        │ dedup by predicted tokens, │
│ GridEncoder: small CNN        │        │ pick seq with best reward  │
│ VectorQuantizer: EMA codebook │        └────────────▲───────────────┘
│  → Z_t: 8x8=64 discrete tokens│                     │
└──────────────┬────────────────┘        ┌─ plm/curiosity.py ─────────┐
               │ Z_t                     │ Stochastic Goose: while    │
               ▼                         │ world-model error is high, │
┌─ plm/trm.py ──────────────────┐        │ pick actions that MAXIMIZE │
│ BeliefCore (GRU): O(1) memory │        │ surprise; hand over to the │
│ H_t = GRU(H_{t-1},[Z_t,A_t-1])│        │ planner once error drops   │
└──────────────┬────────────────┘        └────────────▲───────────────┘
               │ H_t                                  │
               ▼                                      │
┌─ plm/world_model.py ──────────────────────────────────┐
│ BlockCausalSimulator (small transformer):              │
│   (H_t, action) → all 64 next-frame tokens at once     │
│   + reward head (neutral / level-complete / reset)     │
│   + change head (did anything happen — no-op detector) │
└────────────────────────────────────────────────────────┘
```

Parameter budget ≈ 30–80M total. FP32 weights ~0.3GB; the RTX PRO 6000's
96GB go to batched imagination (thousands of parallel rollouts), not
weights. On Kaggle T4 the same nets run with smaller planner batches via
the existing HW profile.

## Module contracts

- `plm/config.py` — one `PLMConfig` dataclass; every dimension in one place.
- `plm/encoder.py` — `ObjectChannels` (numpy BFS, no torch), `GridEncoder`
  (CNN), `VectorQuantizer` (EMA codebook, straight-through), `GridDecoder`
  (for reconstruction loss only).
- `plm/trm.py` — `BeliefCore`: GRU over pooled Z + action embedding.
  `reset()`, `step(z, a) -> H`. Belief never exceeds one vector.
- `plm/world_model.py` — `BlockCausalSimulator.forward(H, a) ->
  (token_logits[64, K], reward_logits[3], change_logit)`. Teacher-forced
  cross-entropy on next-frame tokens.
- `plm/planner.py` — `latent_bfs(H, sim, actions, depth, beam) -> action_seq`.
  All rollouts batched; dedup states by argmax-token bytes (the v13 visited-
  hash idea transplanted into latent space).
- `plm/curiosity.py` — `Goose`: running EMA of prediction error; exposes
  `should_explore()` and `pick(H, candidate_actions)` = argmax predicted
  surprise (disagreement between predicted and... measured post-hoc; the
  ICM-style bonus is predicted-change entropy).
- `plm/agent_plm.py` — `PLMAgent.choose_action(frames, lf)`: the runtime
  loop (encode → belief update → goose-or-plan). Slots into my_agent.py as
  a tier: symbolic BFS (if engine available) → PLM → CNN/heuristic.

## Action vocabulary

`a = (id ∈ {1..7}, x ∈ {0..63}, y ∈ {0..63})`, embedded as
`E_id(id) + E_x(x) + E_y(y)` (x=y=0 for simple actions). Candidate clicks
at runtime come from object centroids (v13's `_dyn_clicks`) — the planner
never enumerates 4096 raw pixels.

## Training curriculum

- **Phase 0 — data factory** (`gen_data.py`): load each local engine the way
  BFSSolver does, roll random + epsilon-goose policies, store shards of
  (grid_t, action, grid_t+1, reward, change) as uint8 npz. Millions of
  transitions, zero cost. ALSO mix in cached v13 solutions as expert
  trajectories (they exercise the win transitions, which random play
  rarely reaches).
- **Phase 1 — see** (`train_wm.py --phase tok`): VQ-VAE reconstruction until
  tokens are pixel-faithful (codebook 1024, patch 8x8). Exact reconstruction
  matters more than compression — ARC is exact-match logic.
- **Phase 2 — predict** (`--phase wm`): freeze tokenizer; train
  BeliefCore + Simulator on K-step teacher-forced rollouts (BPTT through
  the GRU, K=8). Metrics: next-frame token accuracy, reward accuracy,
  no-op detection. Hold out 5 games entirely for generalization measurement.
- **Phase 3 — act**: no policy net at first — the planner IS latent BFS over
  the frozen world model (MuZero-without-the-policy-head). Add a small
  actor later only if planner latency matters on Kaggle.
- **Phase 4 — deploy**: weights → Kaggle dataset (like ForgeNet's
  pretrained_weights.pt). Pretrained-on-public-games weights are standard
  ML practice and competition-legal; what reaches the hidden eval is a
  *general dynamics prior*, not answers. All adaptation happens in-episode.

## Integrity line (decided in v13)

No pre-solved solution caches and no engine source loading on the hidden
eval. The PLM learns at test time from its own interactions. Local engines
and v13 caches are used ONLY as offline training data generators.

## Evaluation gates (each phase must pass before the next)

1. Tokenizer: ≥99.5% pixel-exact reconstruction on held-out frames.
2. World model: ≥90% next-frame token accuracy on held-out GAMES (not just
   held-out frames of trained games) at depth 1; ≥70% at depth 5.
3. Agent: on a held-out game, beats the v13 numpy-bandit fallback's
   levels/actions within the same action budget.
4. Replay sanity: full env replay via play_game.py --fast, like v12/v13.

## Status

- [x] architecture plan (this file)
- [x] module skeletons in `plm/` (untested — sandbox was down; first task
      on a live machine is `python -m plm.smoke`)
- [x] data factory + trainer skeletons
- [ ] harness files (my_agent.py etc.): copy from v13 with
      `sed s/v13/v14/` when a shell is available — same structure, PLM
      added as a fallback tier behind a `V14_PLM=1` env flag
- [ ] Phase 0 run on Mac/RTX
```
