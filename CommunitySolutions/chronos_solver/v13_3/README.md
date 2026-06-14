# chronos_solver v13_3

v13_2 base + **partial-observability fixes** for space-limited levels.

## What's new

Three mechanisms on top of v13_2 (see `RESEARCH.md` for full analysis):

1. **Reveal-novelty atoms** — IW tracks 4×4 grid-cell regions where pixels transition from background to non-background. Exploratory moves that uncover hidden board areas are now kept, not pruned.
2. **Reactive-block atoms** — Non-player pixels that co-move with the player (color-changing blocks, rotation blocks) are tracked as coarse `(grid_y, grid_x, color)` atoms instead of per-pixel atoms. One atom per 8×8 cell per rotation step; keeps IW width tight on ls20-style levels.
3. **Sensing prepass rung** — Short BFS that maximises revealed pixels, then immediately retries IW(1) from each revelation checkpoint. Targets levels where the clean-start reachable space exhausts without a solution.

Ladder: `bfs-sprint → sense(5%) → iw1(8%) → iw2(8%) → ehc(7%) → waypoint(6%) → astar(5%) → bfs-rest → greedy → rescues`

---

## Files

| File | Purpose |
|------|---------|
| `my_agent.py` | Drop-in solver (Kaggle + vast.ai + local) |
| `benchmark.py` | v13_3 vs v13_2 head-to-head harness |
| `_bench_runner.py` | Per-(version, game) subprocess driver |
| `solve_offline.py` | Pre-solve levels offline and cache results |
| `solve_all.py` | Convenience wrapper for solve_offline |
| `vastai-benchmark-runner.ipynb` | vast.ai benchmark notebook |
| `claude-code-v13-3-partial-obs.ipynb` | Kaggle competition submission |
| `RESEARCH.md` | Full analysis of space-limited / partial-observability problems |

---

## Testing locally on Mac (M1 Pro)

```bash
cd CommunitySolutions/chronos_solver/v13_3

# Quick smoke test — single level, short budget
python solve_offline.py --game ls20 --level 0 --budget 60 --workers 8

# Full pre-solve (run overnight)
python solve_offline.py --game ls20 --budget 7200 --bfs-timeout 1200 --workers 8
python solve_offline.py --game ar25 --budget 3600 --bfs-timeout 600  --workers 8
```

Mac uses `spawn` multiprocessing (macOS requirement) and 8 workers.
The sense rung runs single-threaded (it's a short prepass ≤20s).

---

## Benchmark v13_3 vs v13_2 on Mac

```bash
cd CommunitySolutions/chronos_solver/v13_3

# ls20 + ar25 only (recommended first run)
python benchmark.py \
    --games ls20:7,ar25:3 \
    --versions .,../v13_2 \
    --budget 600 --workers 8 --max-states 5000000

# Full suite
python benchmark.py \
    --games ls20:7,ar25:3,cd82:3,vc33:5,su15:1 \
    --versions .,../v13_2 \
    --budget 600 --workers 8 --max-states 5000000
```

Results written to `benchmark_results.json` + `BENCHMARK.md`.

---

## Running on vast.ai (RTX Pro 6000)

1. Push latest commits:
   ```bash
   cd /path/to/arc3
   git add CommunitySolutions/chronos_solver/v13_3/
   git commit -m "v13_3 partial-obs solver"
   git push
   ```

2. Open `vastai-benchmark-runner.ipynb` in the vast.ai Jupyter instance.

3. Run cells top-to-bottom:
   - Cell 1: clone/pull repo
   - Cell 2: install arcengine into kernel interpreter (`sys.executable -m pip`)
   - Cell 3: verify CPU count + GPU
   - Cell 4: ls20 benchmark (1200s/level, v13_3 first)
   - Cell 5: ar25 benchmark (600s/level)
   - Cell 6: print results table
   - Cell 7: print BENCHMARK.md

Workers auto-detected as `cpu_count() - 1` (typically 31 on a 32-core EPYC box).

**Important**: the install cell uses `sys.executable -m pip` — not bare `pip` or `!pip`. This ensures arcengine lands in the same Python the kernel uses.

---

## Kaggle submission

Upload `claude-code-v13-3-partial-obs.ipynb` to Kaggle as a new notebook version.

- Accelerator: **CPU** (BFS is pure CPU — GPU VRAM unused)
- Internet: off (wheels are bundled in the competition dataset)
- Runtime: the competition rerun calls `main.py` automatically; the dummy submission cell handles non-rerun test runs

Per-level budget defaults to **180s** (`V13_BFS_TIMEOUT` env var). To increase:
```python
import os
os.environ['V13_BFS_TIMEOUT'] = '600'
```
Add this cell before the competition rerun cell.

---

## Budget and worker recommendations

| Platform | Workers | Budget/level | Notes |
|----------|---------|-------------|-------|
| M1 Pro (local) | 8 | 600s | `spawn` ctx, good for smoke tests |
| vast.ai RTX Pro 6000 | `nproc-1` (~31) | 1200s ls20, 600s ar25 | `fork` ctx, ~1.7x throughput vs M1 |
| Kaggle CPU | 3-4 | 180s | Competition default |

The sense rung is capped at 20s and runs even within the 180s Kaggle budget.

---

## Expected improvements over v13_2

- **ar25 L1**: space-limited at ~30-50k states in v13_2. Reveal-novelty atoms prevent IW from pruning paths that uncover hidden board regions. Sensing prepass finds revelation checkpoints if the hidden area is reachable.
- **ls20 L5**: reactive blocks (color-change + rotation) generated 16+ noisy per-pixel atoms per step. Coarse reactive-block atoms reduce this to 1 atom per 8×8 cell — IW's width stays tight.
- Other levels: no regression expected (sprint BFS is unchanged; new rungs only add budget-capped prepasses).
