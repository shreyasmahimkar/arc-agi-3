# v15 — token-conditioned world model (the memorization fix)

## What changed and why (30 seconds)

v14's simulator predicted next-frame tokens from a pooled belief vector
alone — the current frame had no direct path to the output. Measured
result: 0.997 train accuracy, **0.366 on fresh episodes of the same
game** (below the ~0.9 copy baseline) — it memorized training tapes
instead of learning dynamics. v15's simulator cross-attends over the
current frame's 64 token embeddings, so "copy" is the default and
dynamics are learned as deltas. Tokenizer is UNCHANGED — v14's trained
tokenizer weights are reused.

Changed files vs v14: `plm/world_model.py` (the fix), `plm/planner.py` /
`plm/curiosity.py` / `plm/agent_plm.py` (carry current tokens through),
`train_wm.py` (one line in rollout_loss + warmup), `diag_mismatch.py`
(copy baseline added). Env vars are now `V15_REQUIRE_WEIGHTS` /
`V15_PLM_WEIGHTS`; Kaggle dataset name will be `v15-plm`.

---

## TONIGHT — Mac pilot (~25 min total, venv312)

The question this answers: does the architecture fix kill the
memorization failure? Gate: diag [C] fresh-episode acc >= 0.90 AND above
the printed copy baseline (v14: 0.366).

```bash
cd CommunitySolutions/chronos_solver/v15

# 0. contracts check (<30s)
python -m plm.smoke

# 1. reuse v14's trained tokenizer (unchanged architecture)
cp ../v14/plm_weights.pt .

# 2. small dataset, 4 trained games (~3 min)
python gen_data.py --games ar25,bp35,cn04,dc22 \
    --episodes-per-game 100 --max-steps 100 --out /tmp/v15_shards

# 3. wm pilot on MPS (~15 min; resumes tokenizer, pretokenizes, trains)
python train_wm.py --phase wm --shards /tmp/v15_shards \
    --epochs 3 --steps-per-epoch 300 --bsz 64

# 4. THE verdict (compare with v14's 0.366)
python diag_mismatch.py --game ar25 --shards /tmp/v15_shards

# 5. if [C] >= 0.90: watch it actually play
V15_REQUIRE_WEIGHTS=1 python play_game.py --game ar25 --fast
```

Reading step 5's log: `plm:goose(err=...)` should now FALL with each
step (v14 flatlined at 0.70). Below 0.12 the planner engages:
`plm:bfs(...)` / `plm:plan(...)`. With only 3 pilot epochs the reward
head may be too weak to find wins — purposeful behavior + falling err +
planner engagement is tonight's success, not level completion.

If [C] stays < 0.7: the architecture fix wasn't sufficient — stop,
nothing to gain from the RTX run; we go deeper instead.

---

## NEXT — RTX box, full training (~2.5h, only if the Mac gate passes)

Same vast.ai recipe as v14 (vast-train-v14.ipynb cells work — change
`v14` -> `v15` in every path). Or by hand:

```bash
cd /workspace/arc3 && git pull
cd CommunitySolutions/chronos_solver/v15

# tokenizer: reuse v14's (skip ~1h of tok training)
cp ../v14/plm_weights.pt .        # if v14 weights present on the box
# (no v14 file? run --phase tok --epochs 8 --bsz 512 first)

# data: v14 shards are bit-compatible — reuse if still on the box
ls /workspace/v14_shards/*.npz 2>/dev/null \
    || python gen_data.py --out /workspace/v14_shards \
        --episodes-per-game 400 --max-steps 150

nohup /workspace/venv/bin/python train_wm.py --phase wm \
    --shards /workspace/v14_shards --epochs 20 --steps-per-epoch 1000 \
    --bsz 1024 --compile --holdout ls20,vc33,tu93,ft09,sp80 \
    > train.log 2>&1 &
```

Notes for the box (all v14 lessons, already coded in):
- compile warmup falls back to eager if Blackwell ptxas chokes — watch
  for `wm: compile failed at warmup` (harmless, ~40% slower).
- every improving epoch writes a complete 3-key plm_weights.pt — safe to
  download mid-run after the first `*best, checkpointed*`.
- token caches: /workspace/v14_shards_tokens from v14 ARE reusable (same
  tokenizer fingerprint) IF you copied v14's plm_weights.pt. A retrained
  tokenizer invalidates them automatically.
- gates: train_tok_acc near v14's 0.997 is expected; the interesting
  number is HELDOUT_tok_acc — v14 got 0.252. The copy path alone should
  lift it near ~0.9 (static patches transfer); above that means real
  zero-shot dynamics transfer. The 0.90 gate may STILL fail on true
  game-specific mechanics — that's what augmentation (color perms + D4)
  and test-time training attack next; they are NOT in this run.
- after download: `python diag_mismatch.py --game ar25 --shards ...` on
  the Mac, then play_game on ar25 (trained) and ls20 (held out).
- DESTROY the instance when plm_weights.pt + train.log are local.

## THE THREE-PASS PIPELINE (2026-06-12 restructure)

v15 is now explicitly three passes sharing one artifact — the game
scratchpad (`scratchpad.py`, JSON per game in `v15_scratch/`):

**PASS 1 — symbolic scout (offline only; needs engine source).**
`pass1_bfs.py`: v13's pure BFS runs ~600s/game, probes every action's
effect from the start state, solves what it can, and WRITES the
scratchpad: exact solutions, action-effect measurements, stuck points,
LLM-readable notes. Resumable; hydrates v13/v12 caches first so budget
goes into NEW levels.

```bash
python pass1_bfs.py --games ar25,bp35,cn04,dc22 --budget 600
```

**PASS 2 — distillation (offline).** gen_data's expert replays now read
pass-1 scratchpads FIRST (then v13/v12 caches), so every level pass 1
cracks becomes chained WIN-rich training data for the value/reward
heads. Same commands as before — the hookup is automatic
(`V15_SCRATCH` env overrides the scratchpad dir).

**PASS 3 — the PLM plays (everywhere, incl. hidden eval).** Deep-think
planner (600s budget when stuck) + a LIVE scratchpad learned strictly
in-episode: per-action effect tallies that prune proven-useless actions
from the planner's branching. Offline scratchpads never ship to the
eval as lookup tables (integrity line) — their knowledge arrives only
as weights via pass 2.

## Queued after this (decided, not yet built)

1. Augmentation in gen_data: per-episode color permutation + D4
   transforms (with action-direction remap) — attacks the held-out-games
   gap directly.
2. Codebook dead-code revival (700/1024 near-duplicates, 35 in use —
   stable, so deferred).
3. Test-time training: finetune wm on the episode's own transitions when
   goose error stays high (the v15 architecture finally makes the
   in-distribution model worth carrying to new games).
4. Kaggle notebook + `v15-plm` dataset (copy v14's notebook, rename) —
   only after gates 1–3 in the README pass.


V15_REQUIRE_WEIGHTS=1 V15_THINK_BUDGET=60 python play_game.py --game ar25 --fast