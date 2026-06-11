# v14 PLM — Runbook: what to run where, and how to ship it to Kaggle

Three machines, three jobs:

| Machine | Job | Why |
|---|---|---|
| **MacBook (M1 Pro)** | develop, smoke-test, generate data, short training runs | fast iteration, 8 perf cores, MPS is fine for small runs |
| **RTX PRO 6000 box** | full training (Phases 1–2), big evals | the CUDA/bf16/compile path; hours instead of overnight |
| **Kaggle (T4)** | submission only — inference, never training | offline, time-limited, hidden eval games |

Everything below assumes repo root + `source .venv312/bin/activate`.
All v14 commands run from `CommunitySolutions/chronos_solver/v14/`.

---

## Stage 1 — MacBook: validate + data (DONE / repeat as needed)

```bash
cd CommunitySolutions/chronos_solver/v14

# 1. module smoke test (passed 2026-06-10)
python -m plm.smoke

# 2. training data from the 25 local engines -> /tmp/v14_shards
#    (~60 episodes/sec on the M1; this takes ~2 min)
python gen_data.py --episodes-per-game 50 --max-steps 100

# 3. tokenizer pilot (~10-15 min on MPS)
python train_wm.py --phase tok --shards /tmp/v14_shards \
    --epochs 5 --steps-per-epoch 300 --bsz 64
```

**Gate:** `pixel_acc >= 0.95 and climbing` → go to Stage 2.
Stalled at 0.5–0.8 → VQ codebook collapse; stop and report (fixes:
dead-code revival, lower commit weight — flagged, not yet needed).

**Known bottleneck:** `frame_to_tensor`'s object-channel BFS is pure
Python (~ms per frame). Acceptable on pilots; before Stage 2 long runs,
swap in `scipy.ndimage.label` or precompute tokens per shard.

---

## Stage 2 — RTX PRO 6000: full training

The PLM auto-detects CUDA: bf16 autocast + cudnn.benchmark turn on by
themselves (`device_setup()` in train_wm.py). No flags needed.

```bash
# 1. bigger dataset (CPU-bound; the box's core count is the win here)
python gen_data.py --out /data/v14_shards \
    --episodes-per-game 400 --max-steps 150

# 2. Phase 1+2 in one shot; held-out games NEVER appear in training
python train_wm.py --phase all --shards /data/v14_shards \
    --epochs 20 --steps-per-epoch 1000 --bsz 256 \
    --holdout ls20,vc33,tu93,ft09,sp80
```

**Gates (from README "Evaluation gates"):**
1. tokenizer `pixel_acc >= 0.995`
2. world model `HELDOUT_tok_acc >= 0.90` — this is THE number: next-frame
   prediction on games the model has never seen. If train acc is high but
   held-out is low, the model memorized mechanics instead of learning
   grid physics → more data diversity, not more epochs.

Output: `plm_weights.pt` (tokenizer + belief + world_model state dicts).

### Stage 2 on vast.ai — step by step

Provisioned instance (2026-06-10): **RTX PRO 6000 WS** @ 185.99.66.48,
95.6GB VRAM, EPYC 7452 (16 cores allotted), **270GB disk** (roomy),
112GB RAM, image `vastai/pytorch:cuda-13.1.2-auto`, Jupyter enabled.
$0.095/hr — it bills while idle, destroy when done.

Repo cleanup already done (old `.venv` + 2.8GB v11 wheels zip deleted;
.gitignore prevents their return). KEEP `.venv312` locally — that is the
active Mac environment.

#### Step 1 — Mac: pack the repo (one archive, no cherry-picking)

```bash
cd <repo root>
tar czf /tmp/arc3.tar.gz --exclude=.git --exclude=.venv312 \
    --exclude='*.log' --exclude='images' .
```

#### Step 2 — upload to the instance

Drag `arc3.tar.gz` into the Jupyter file browser (it lands in
`/workspace`). Alternative: get the SSH port from the instance's
"Open SSH Interface" button, then:

```bash
scp -P <port> /tmp/arc3.tar.gz root@185.99.66.48:/workspace/
```

#### Step 3 — instance setup + sanity (Jupyter: New -> Terminal)

```bash
mkdir -p /workspace/arc3 && cd /workspace/arc3 && tar xzf /workspace/arc3.tar.gz
pip install \
    arc-prize-2026-arc-agi-3/arc_agi_3_wheels/arcengine-0.9.3-py3-none-any.whl \
    arc-prize-2026-arc-agi-3/arc_agi_3_wheels/arc_agi-0.9.8-py3-none-any.whl \
    python-dotenv
cd CommunitySolutions/chronos_solver/v14
python -c "import torch; print(torch.cuda.get_device_name(0))"  # RTX PRO 6000
python -m plm.smoke                                             # must pass
```

#### Step 4 — data + training (nohup: a dropped tab must not kill the run)

```bash
python gen_data.py --out /workspace/v14_shards \
    --episodes-per-game 400 --max-steps 150            # ~5 min

nohup python train_wm.py --phase all --shards /workspace/v14_shards \
    --epochs 20 --steps-per-epoch 1000 --bsz 256 \
    --holdout ls20,vc33,tu93,ft09,sp80 > train.log 2>&1 &
tail -f train.log        # Ctrl-C detaches from tail; training continues
```

Watch: `pixel_acc` (gate 0.995, tokenizer) then `HELDOUT_tok_acc`
(gate 0.90, world model). Expect low GPU utilization on this first run —
the pure-Python object-channel featurization is the known CPU bottleneck.

#### Step 5 — download `plm_weights.pt` (BEFORE destroying the instance)

- Easiest: Jupyter file browser -> navigate to
  `/workspace/arc3/CommunitySolutions/chronos_solver/v14/` ->
  right-click `plm_weights.pt` -> Download. Save it into your local
  `v14/` folder. Grab `train.log` too for the record.
- Or from the Mac:

```bash
scp -P <port> root@185.99.66.48:/workspace/arc3/CommunitySolutions/chronos_solver/v14/plm_weights.pt \
    ~/gitrepos/OpenSource/kaggle/arc3/CommunitySolutions/chronos_solver/v14/
```

- Note: `*.pt` is gitignored — weights travel by hand/Kaggle dataset,
  never via git. Verify locally (expect `['tokenizer', 'belief', 'world_model']`):

```bash
python -c "import torch; print(list(torch.load('plm_weights.pt', map_location='cpu', weights_only=True)))"
```

#### Step 6 — DESTROY the instance

vast.ai console -> trash icon. A stopped instance can still bill for
storage; a destroyed one bills nothing.

**Local end-to-end check before shipping** (uses a copy of the v13
play_game runner; the agent will log `plm:` reasonings instead of `bfs:`):

```bash
# weights are picked up automatically from the v14 dir
V14_REQUIRE_WEIGHTS=1 python play_game.py --game ls20 --fast
```

Compare levels/actions on held-out games against the v13 bandit baseline
(gate 3 in the README). The PLM must beat the bandit before it's worth
shipping.

---

## Stage 3 — Kaggle: deploy

### 3a. Create the dataset (one-time, update on every retrain)

Upload a private Kaggle dataset named exactly **`v14-plm`** containing:

```
v14-plm/
├── plm/              <- the whole package directory
├── my_agent.py
└── plm_weights.pt    <- from Stage 2 (optional but the whole point)
```

CLI route (or use the website UI):

```bash
cd CommunitySolutions/chronos_solver/v14
mkdir -p /tmp/v14-plm && cp -r plm my_agent.py plm_weights.pt /tmp/v14-plm/
kaggle datasets create -p /tmp/v14-plm --dir-mode zip   # first time
kaggle datasets version -p /tmp/v14-plm -m "retrained"  # updates
```

### 3b. The notebook

Upload `claude-code-v14-plm.ipynb`. In the notebook editor:
- attach the competition data + the `v14-plm` dataset
- accelerator: GPU (T4 is fine — the PLM is ~50M params)
- internet: OFF (required by the competition)

Cell flow (already written): pip wheels → stage code/weights from the
dataset + `ast.parse` syntax guard + `plm.smoke` (interactive only) →
competition-rerun cell (gateway curl, harness copy, run with
`PYTHONUNBUFFERED=1 ... | tee /kaggle/working/v14_run.log`) → dummy
submission fallback (keep!).

### 3c. Test, then submit

1. **Save & Run All** first. Watch the Logs tab (real-time): you want
   `V14 agent ready: plm=on torch=yes`, `PLM: loaded weights from ...`,
   and per-step `plm:goose(...)` → `plm:bfs(...)` reasonings.
   `plm=OFF` means the dataset isn't attached or weights failed to load.
2. Download `v14_run.log` from the version's output artifacts for
   post-mortems — during the *scoring* rerun the Logs tab is not visible.
3. Submit to the competition once the public run looks sane.

### What the hidden eval actually exercises

No caches, no engine sources (the v13 integrity line). The agent arrives
with a general grid-dynamics prior and must, per game: explore while the
world model is surprised (goose), compress what it learns into the
belief state, then plan in imagination. Expect the first ~10 actions per
game to be exploration — that's by design, not a bug.

---

## Troubleshooting quick table

| Symptom | Cause | Fix |
|---|---|---|
| `plm=OFF torch=yes` in logs | weights/package not found | check dataset attached, paths in staging cell |
| `PLM step failed (...) — bandit takes over` | runtime bug in model code | grab v14_run.log traceback, fix, new dataset version |
| pixel_acc stalls <0.8 | codebook collapse | report — needs dead-code revival in VectorQuantizer |
| HELDOUT_tok_acc ≪ train acc | memorization | more games/diversity in gen_data, fewer epochs |
| Kaggle run very slow | planner batches too big for T4 | lower `plan_beam`/`plan_topk_clicks` in PLMConfig |
| OOM during training | batch too big | trainer halves batch automatically; or lower --bsz |

## Current status (2026-06-10)

- Smoke test: PASSED on M1
- gen_data: working, 25/25 games
- Tokenizer: training confirmed (loss 2.77→2.02 in 100 pilot steps);
  full pilot per Stage 1 step 3 is the next action
- World model / eval / Kaggle: not yet run
