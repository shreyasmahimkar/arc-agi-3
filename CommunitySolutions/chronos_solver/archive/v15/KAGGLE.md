# v15 — Kaggle deployment runbook

Entry point: `my_agent.py`. Everything game-specific happens inside it
at runtime (scout → TTT → plan); what you ship is code + one weights
file. Three stages to get there.

---

## Stage A — RTX PRO 6000: mint the prior (`plm_weights.pt`)

The weights in the dataset are the agent's general grid-physics prior.
Train the full version on the box before uploading (the Mac pilot
weights work but are 4-games-thin).

```bash
cd /workspace/arc3 && git pull
cd CommunitySolutions/chronos_solver/v15

# 1. tokenizer: reuse v14's trained one if present, else train
cp ../v14/plm_weights.pt . 2>/dev/null \
    || python train_wm.py --phase tok --shards /workspace/v15_shards \
        --epochs 8 --bsz 512

# 2. pass-1 scratchpads for ALL games (BFS + probes, feeds gen_data)
python pass1_bfs.py --games all --budget 600

# 3. data: all 25 public games, WIN-rich chained expert replays
python gen_data.py --out /workspace/v15_shards \
    --episodes-per-game 400 --max-steps 150 --expert-frac 0.4

# 4. world model + value head (~2h)
nohup /workspace/venv/bin/python train_wm.py --phase wm \
    --shards /workspace/v15_shards --epochs 20 --steps-per-epoch 1000 \
    --bsz 1024 --compile --holdout ls20,vc33,tu93,ft09,sp80 \
    > train.log 2>&1 &
```

Gates before shipping: every `wm epoch` line finite (the NaN fuse logs
skips), final `wm: best HELDOUT_tok_acc` recorded, and on the box run
the diag: `python diag_mismatch.py --game ar25 --shards /workspace/v15_shards`
→ [C] >= 0.90 and [E] correlation > 0.7.

**Blind-suite rehearsal on the box (the real go/no-go):** simulate the
hidden eval by denying the agent its local advantages:

```bash
for g in ls20 vc33 tu93 ft09 sp80; do   # the holdout games
  V15_BFS_BUDGET=0 V15_REQUIRE_WEIGHTS=1 \
  V15_SCOUT_ACTIONS=80 V15_TTT_SECONDS=180 V15_THINK_BUDGET=120 \
  python play_game.py --game $g --fast
done
```

`V15_BFS_BUDGET=0` disables the engine shortcut → the agent runs the
exact scout→TTT→plan path Kaggle will see, on games the prior never
trained on. Count levels completed vs the bandit baseline. This number
IS your expected Kaggle behavior.

---

## Stage B — the `v15-plm` dataset

Contents — EXACTLY these three things:

```
v15-plm/
├── plm/              <- whole package incl. ttt.py
├── my_agent.py
└── plm_weights.pt    <- from Stage A
```

**Must NOT contain** (integrity line + they'd be dead weight anyway):
`*_bfs_cache_*.json`, `v15_scratch/`, `v13_agent.py`, `pass1_bfs.py`,
`gen_data.py`, `train_wm.py`, engine sources, `__pycache__`.

```bash
cd CommunitySolutions/chronos_solver/v15
mkdir -p /tmp/v15-plm && cp -r plm my_agent.py plm_weights.pt /tmp/v15-plm/
rm -rf /tmp/v15-plm/plm/__pycache__
kaggle datasets create -p /tmp/v15-plm --dir-mode zip   # first time
kaggle datasets version -p /tmp/v15-plm -m "v15 prior"  # updates
```

## Stage C — the notebook (`claude-code-v15-plm.ipynb`)

Upload it; in the editor: attach competition data + the `v15-plm`
dataset, accelerator **GPU**, internet **OFF**. Then **Save & Run All**
and read the Logs tab before submitting:

1. `weights OK: ['tokenizer', 'belief', 'world_model']` and
   `syntax OK` lines, smoke test passes (interactive run only)
2. nothing else runs interactively — the rerun cell is gated on
   `KAGGLE_IS_COMPETITION_RERUN`; the dummy submission cell writes
   `submission.parquet`
3. submit; during the scoring rerun grab `v15_run.log` from the output
   artifacts for the post-mortem (Logs tab is hidden during scoring)

---

## Budgets — what to set and why

| Knob | Kaggle (T4) | RTX rehearsal | Reasoning |
|---|---|---|---|
| `V15_BFS_BUDGET` | irrelevant (no engine; probe exits in ms) | 0 to simulate Kaggle; 600 for max-capability runs | hidden eval has no source |
| `V15_SCOUT_ACTIONS` | **80** | 80 | real actions = RHAE debt; 80 is the floor that still feeds TTT a usable buffer |
| `V15_TTT_SECONDS` | **180** | 180 (box trains ~5x more steps in the same budget) | costs zero actions, only wall clock; per-game overhead must fit the 8h total |
| `V15_THINK_BUDGET` | **120** | 120 | deep-think per stuck moment; 600 risks the global clock if many games stall |
| `V15_RESCOUT_ACTIONS` | **40** | 40 | second helping of data when stuck |
| `V15_STUCK_WINDOW` | **60** | 60 | actions without a level before re-scout+retrain |

Wall-clock sanity: per game ≈ scout (fast) + TTT 180s + a few deep
thinks (~10 min worst case) + 1–2 retrain cycles → ~20–30 min/game
ceiling. If the eval has many games and the 8h budget pinches, drop
`V15_TTT_SECONDS` to 120 and `V15_THINK_BUDGET` to 60 in the notebook's
rerun cell — they're just env vars on the `main.py` line.

## What the Logs should narrate, per hidden game

```
V15 agent ready: plm=on torch=yes phase=scout ...
V15 pass1 gate OPEN — attempting real BFS
V15 pass1-BFS: no engine/v13 module — scout path      <- EXPECTED
scout(79 left):np:a1 ...                              <- pass 1
V15 TTT #1: training on 1 episodes (80 transitions)...
V15 TTT done: {'steps': ..., 'wins_in_buffer': ...}   <- pass 2
plm:bfs-soft(p=0.31,commit=4) / plm:think(3p)-...     <- pass 3
V15: no progress in 60 actions — re-scouting 40       <- stuck loop
```

`plm=OFF` or `np:`-only reasonings after TTT = staging failure; fix the
dataset attach before submitting. Tier guide: `bfs-exec` will never
appear on the hidden eval (that's the local-only tier) — its absence is
correct, not a bug.

## Known limits going in (so the score reads honestly)

- The prior's zero-shot HELDOUT accuracy is the weak link; TTT exists
  to close exactly that gap in-episode. The Stage-A blind rehearsal
  tells you how well it does before you spend a submission.
- Scout actions are RHAE debt on every game (~80 actions vs baselines
  of 15–100 means early levels score low even when solved). Tuning the
  scout shorter is the first lever if rehearsal shows the TTT learns
  fine from less data.
