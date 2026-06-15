# v19 — scaling the offline training to a big GPU (RTX PRO 6000)

The Kaggle submission never needs a big GPU (black-box `ForgeAgent` is tiny, RHAE
is action-scored). The **only** place GPU power helps is **offline training of the
prior** — `pretrained_weights.pt` (ChangeNet) and `wm_weights.pt` (WorldModel).
The training scripts are now ready to exploit a big card the moment your corpus is
large enough. Nothing about the shipped agent changes: weights stay a plain
`state_dict`, and every consumer infers the trained width from the checkpoint, so
a wider prior auto-loads on Kaggle.

## New flags (both `pretrain.py` and `train_wm_v19.py`)

| flag | meaning |
|---|---|
| `--bsz -1` | batch auto-scales with VRAM: **1024** @96 GB, **512** @48 GB, **256** @T4, **128** cpu/mps. Pass a number to override. |
| `--net-mult N` | trunk width ×N (ChangeNet base 32→`32N`, WorldModel base 64→`64N`). `4` = the "beast" prior the RTX PRO 6000 profile targets. |
| `--amp` | fp16 mixed precision (CUDA only; no-op elsewhere). Version- and device-safe. |
| `--patience K` | held-out **plateau early-stop**: if transfer accuracy isn't beaten for K epochs, stop and say so — your signal that more compute won't help. |

Both scripts **gate on held-out transfer**: they only overwrite the shipped weights
when accuracy on games never trained on improves (`pretrain` = change-prediction
acc on the 5 held-out games; `train_wm` = next-frame **chg-acc**). A `--net-mult`
change starts a fresh lineage (warm-start is skipped on a width mismatch).

## The graduation procedure

**Stage 1 — grow the corpus first (CPU + time, GPU idle).** Run the ExIt flywheel
on a many-core box until the corpus is large; the bottleneck is `solve_all.py`'s
parallel BFS, which is CPU-bound. *Advance only when held-out acc is still climbing
as data grows* — that means you're data-limited, so more training will pay off.

```bash
# many-core box; GPU not the bottleneck here
./exit_cycle.sh                      # solve -> harvest -> train, looped
```

**Stage 2 — scale batch + AMP on the big GPU.** Now epochs over the big corpus are
the cost; this is where a T4 caps out and the 96 GB card pulls ahead.

```bash
python pretrain.py     --bsz -1 --amp --epochs 30 --patience 6
python train_wm_v19.py --bsz -1 --amp --epochs 120 --patience 20
```

**Stage 3 — scale capacity (GPU genuinely needed).** Train a higher-capacity prior.

```bash
python pretrain.py     --net-mult 4 --bsz -1 --amp --epochs 40 --patience 8
python train_wm_v19.py --net-mult 4 --bsz -1 --amp --epochs 150 --patience 25
```

**Then:** re-upload only the new `pretrained_weights.pt` to your Kaggle dataset and
resubmit. The shipped `ForgeAgent` reconstructs the trained width automatically.

## When to STOP scaling

If `--patience` keeps firing the **PLATEAU** message while held-out acc stays flat,
the GPU is not your limiter — the prior is algorithm-limited, not data/capacity-
limited. Spend the effort on better features / a progress-aware target (PROGRESS
lever #2) instead of a bigger card.

## Smoke tests (no GPU / no engine needed)

```bash
python pretrain.py --selftest      # width-scaling, AMP path, gate, save->reload round-trip
python wm_planner.py --selftest    # planner loads the WM at its trained width
```
