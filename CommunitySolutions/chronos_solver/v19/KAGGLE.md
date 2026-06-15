# v19 → Kaggle: package, run, and read the score

Companion to `v19-to-kaggle.ipynb`. v19 is the **black-box** agent for ARC-AGI-3
competition mode (API-only, source unreachable, RHAE scoring). White-box BFS dies
on the private set; the legal path is the `ForgeAgent` CNN + transition graph,
warm-started by an offline-trained prior whose knowledge lives **in the weights,
not in stored answers**.

## 1. Build the dataset

Create a **private** Kaggle dataset (e.g. `v19-forge`) with EXACTLY these 3 files
at the top level — nothing else:

| file | role |
|---|---|
| `combined_agent.py` | entry point, `class MyAgent` (BFS-probe → black-box fallback) |
| `forge_agent.py` | the black-box `ForgeAgent` (ChangeNet CNN + frame-hash graph) |
| `pretrained_weights.pt` | the offline ChangeNet prior — **required** for real performance |

**Do NOT include** `solutions/`, any `*_bfs_cache_*.json`, engine sources, or
`v13`/`v15` scratch. The notebook's cell 2 hard-asserts these are absent.

```bash
# from CommunitySolutions/chronos_solver/v19/
mkdir -p /tmp/v19-forge
cp combined_agent.py forge_agent.py pretrained_weights.pt /tmp/v19-forge/
# upload /tmp/v19-forge as a private Kaggle dataset
```

## 2. Configure the notebook

1. Open `v19-to-kaggle.ipynb` on Kaggle.
2. **Add Input** → attach (a) the `v19-forge` dataset and (b) the competition data.
3. **Accelerator: GPU (T4)**. **Internet: OFF**.
4. **Save Version → Save & Run All**. The dummy `submission.parquet` is written
   interactively; the real agent runs only during the scoring rerun
   (`KAGGLE_IS_COMPETITION_RERUN`).

The run is forced honest with `V19_STORE_SOLUTIONS=0` — solve live, never hydrate
a cached answer (the hard no-stored-answers rule).

## 3. Is it scoring well? — what to check

**On the leaderboard:** the score is summed RHAE across the private games. Non-zero
= the black-box agent cleared real levels blind (the only thing that counts here).
Baseline: preview winners scored ~6–13%; v18's white-box "solves" do **not**
transfer, so compare v19 only to other black-box runs.

**In the Logs tab / `v19_run.log`:**
- `prior=loaded` — the weights staged. `prior=cold` → fix the dataset, the score
  is meaningless without the prior.
- `BFS: game source not found` then `forge` actions — expected (API-only).
- Count solved levels **and the actions each took**. RHAE squares the
  human/AI action ratio, so 40 actions ≫ 400 actions for the same level. The lever
  is action-*efficiency*, not raw clears.

## 4. Improve the prior, re-measure (the loop)

The architecture is fixed; the score moves when the **prior** gets better:
1. Grow the offline corpus and retrain — `pretrain.py`, or the ExIt flywheel
   (`solve_all.py` → `harvest_wm.py` → `train_wm_v19.py`).
2. Track held-out **chg-acc** in `WM_LOG.md` (next-frame accuracy on changed
   pixels) — that is the dynamics metric that predicts transfer.
3. Re-upload **only** the new `pretrained_weights.pt`, resubmit, compare. Same
   agent + a better warm start = fewer wasted actions = higher RHAE.
