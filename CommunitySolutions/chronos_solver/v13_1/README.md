# Chronos Solver v13_1 — v13 base + space-shrinking search rungs

Implementation of `RESEARCH.md` (combinatorial-optimization concepts applied
to the level search). Lineage: FORGE v19 → v12 → v13 → **v13_1**. Same
architecture as v13 — snapshot-frontier search, multiprocess node expansion,
transient masking, hidden-field probing, chained true baselines — plus four
new rungs that *shrink the searched space* instead of chewing it faster.

## New search rungs (all in `my_agent.py`)

1. **Waypoint/TSP decomposition** (`--strategy waypoint`):
   - *Movement games*: detect the player sprite (the color whose centroid
     moves under directional actions), enumerate orderings of object
     centroids (partial tours included, cheapest Manhattan tour first, ≤24
     tours over ≤5 waypoints), A* a small sequential leg to each waypoint,
     then a short finish-BFS from the tour end. Searches `k! × k·b^d`
     instead of `b^(k·d)`.
   - *Click games*: macro BFS restricted to **object-centroid clicks**
     (`force_dyn`) — branching = #objects instead of #scanned pixels.
2. **A\*** (`--strategy astar`): best-first on `depth + 1.5·manhattan(player,
   goal)/step`; goal = rarest non-player object. Degrades to BFS order
   (h=0) when player/goal aren't detectable.
3. **IW(1)/IW(2) novelty pruning** (`--strategy iw`): a child is kept only
   if it makes some atom true for the first time in the whole search.
   Atoms: per-cell colors (bool array, O(1) check) + public scalar attrs;
   IW(2) escalates to pairs of object-level atoms. Bounds kept states
   linearly in #atoms — the cure for combinatorially wide levels.
4. **Dominance pruning** (inside astar/iw rungs): prune states whose
   (player-cell, color-histogram, scalar attrs) were already seen at ≤
   depth — strictly more aggressive than exact-hash dedup.

## Ladder (`--strategy auto`, the default)

`bfs(28%) → waypoint(16%) → astar(14%) → iw1(12%) → iw2(10%) → greedy(20%)`
then v13's rescues (unmasked-hash retry, hidden-field retry) on leftover
budget. Exact rungs first (optimality preserved on easy levels), aggressive
rungs after.

**Every non-exact rung solution is verified by replay** from the search
baseline before being banked (`_verify_from_snap`) — waypoint composes
sub-solutions and iw/astar are incomplete, so unverified answers are never
trusted. Workers now also return the child's public scalar state, so
novelty/dominance/heuristics are computed parent-side with no snapshot
restores.

## Files

- `my_agent.py` — v13 agent + solver with the new rungs (drop-in compatible)
- `solve_offline.py` / `solve_all.py` — same CLIs, default `--strategy auto`,
  caches to `v13_1_bfs_cache_<game>.json`
- `benchmark.py` + `_bench_runner.py` — v13 vs v13_1 head-to-head: fresh
  solvers, no caches, equal per-level budget (v13 = bfs+greedy split,
  v13_1 = auto ladder). Writes `benchmark_results.json` + `BENCHMARK.md`.
- `RESEARCH.md` — the analysis this version implements

## Benchmark results (sandbox: 4-core/3GB Linux VM, 28s/level, workers=3)

Full table in `BENCHMARK.md`. Headline — **v13_1 8/11 levels, v13 7/11**,
equal total budget per level:

- **ar25 L0: v13 FAILED (11.7k states explored), v13_1 SOLVED — IW(1)
  novelty rung, 15 actions, 1.9s in-rung** (and passed replay
  verification). The RESEARCH.md bet paying off: ar25's space is wide, and
  novelty pruning collapses it.
- Easy levels (cd82, vc33, ls20 L0): identical — the bfs sprint preserves
  v13's behavior exactly.
- ls20 L1/L2: both solve with identical states explored (the bfs-rest rung
  resumes the sprint frontier), but v13_1 pays a ~10s "ladder tax" trying
  the aggressive rungs first. The tax is constant per level; it shrinks to
  noise at real budgets (600s).
- bp35 L0, su15 L0, ar25 L1: unsolved by both at 28s — these need the M1
  Pro run with real budgets.

Rerun on the M1 Pro with `--workers 8 --budget 600` for the real
comparison:

```bash
cd CommunitySolutions/chronos_solver/v13_1
python benchmark.py --games ls20:7,ar25:3,cd82:3,bp35:2,vc33:5,su15:1 \
    --budget 600 --workers 8 --max-states 5000000
```

## Notes / caveats

- IW and dominance rungs are **incomplete** (can prune the only path) —
  that's why they sit behind exact BFS in the ladder and why their wins are
  replay-verified. The hidden-countdown lesson from v13 is covered: scalar
  attrs are part of the novelty atom set, so pixel-identical countdown
  chains stay novel.
- Waypoint legs run sequentially (no pool) — legs are tiny; the pool's
  per-batch shipping would dominate.
- `solve_all.py` now runs a single `auto` pass per level instead of
  bfs-then-greedy (the ladder subsumes both).
