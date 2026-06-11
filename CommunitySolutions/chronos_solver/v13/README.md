# Chronos Solver v13 — v12 base, attacking the unsolved levels

v13 is a direct continuation of v12 (same structure, caches carried over).
Lineage: FORGE v19 → v12 (9 fixes) → v13. No LLM anywhere.

## New in v13

1. **Warmup-prefix fix**: when a locked level (sc25-type) is unlocked by a
   warmup action, that action is now PREPENDED to the found solution — v19
   solved from the post-warmup state and the replay desynced by one action.
2. **`--max-states` flag** (default 5M): cn04 explores at ~5700 sims/s and
   slammed the hardcoded 500k cap mid-search.
3. **Frontier disk guards**: persist skips when <1.5GB free, frontiers are
   capped at 25k nodes, and solved levels' checkpoints are auto-deleted
   (a runaway cn04 frontier once filled the sandbox disk and wedged it).
4. Per-game diagnosis (all cold games sim at 0-4ms — their spaces are wide,
   not slow): sc25 unlocks via warmup ✓, su15 branches 42 clicks, lf52 is
   unpicklable → auto-sequential, tn36 same.

## Results so far (ls20)

| Level | v19 status | v13 actions | How |
|---|---|---|---|
| L0 | solved (13) | **13** | masked BFS, 1.5s |
| L1 | BFS timeout → CNN flail | **45** | true-baseline BFS |
| L2 | unsolved | **39** | solution transfer + BFS |
| L3 | unsolved | **43** | true-baseline BFS |
| L4 | unsolved | **44** | scalar-hash BFS (133k states) |
| L5 | unsolved | in progress | dual-key level, ~80k states searched |
| L6 | unsolved | pending | Fog level |

Replay through the real env: **5/7 levels, scorecard 53.57** (was 3.57).

## Multi-game sweep (sandbox quick-wins; deep levels → run solve_all.py)

Banked so far (`v13_bfs_cache_<game>.json`) — **31 levels, 13 games**:
tu93 ALL 9 ✓, ls20 5 (L5 frontier ~85k states, L6 cold), vc33 4 (L4 in
progress), ar25 2, cd82 2, m0r0 2, plus L0 of dc22/ft09/lp85/r11l/s5i5/
sk48/sp80. Deep holdouts for solve_all.py on real hardware: ls20 L5-6,
ar25 L2+, cd82 L2+, vc33 L4+, bp35, cn04, g50t, ka59, lf52, re86, sb26,
sc25, su15, tn36 (unpicklable → sequential), tr87, wa30.

Two scanner upgrades from this sweep (both in `my_agent.py`):
- background-pixel click probing (selection games take clicks on empty cells)
- **dynamic click targets**: each search node carries its frame; workers add
  clicks at the current frame's object centroids. vc33 L3 was UNREACHABLE
  with root-scanned clicks (23k states exhausted) and solved in 9.7s with
  dynamic targets.
- games with unpicklable state (tn36's lambdas) auto-fall back to
  sequential expansion.

## Bugs found in v19 (each killed levels)

1. **`depth < 30` BFS cap** silently truncated any solution longer than 30
   actions (ls20 L1 needs 45+). Raised to 200 — visited-dedup already bounds
   the search.
2. **Timer-bar state aliasing**: the HUD timer changes pixels every step, so
   every (position × timer-tick) hashed unique → state explosion. v13
   auto-detects transient rows (hot under EVERY single-action rollout) and
   masks them from the dedup hash, with an automatic unmasked retry if the
   masked space exhausts (some levels are genuinely time-dependent).
3. **Hidden-countdown pruning**: ls20's lock opens via an internal countdown
   (`akoadfsur`) during which frames are pixel-identical — BFS pruned the
   chain as "visited" and the win was unreachable. v13 folds the game
   object's public scalar attrs (key shape/color/rotation, countdowns,
   player coords) into every state hash.
4. **Wrong BFS baseline for levels > 0** (the big one): `set_level(N)` + RESET
   produces a *different* start state than naturally advancing from L(N-1)
   (player position, carried key rotation, ~1400px frame diff on ls20 L1).
   Solutions solved from the synthetic baseline fail when replayed in the
   env. v13 builds level N's start state by **chaining the cached solutions
   for L0..L(N-1)** — fixing correctness AND shortening solutions
   (L1 58→45, L2 97→39, L3 140→43).
5. **Action pruning at spawn**: `_scan_actions` dropped actions that did
   nothing at the start state; state-dependent actions (interact buttons)
   are now kept, ordered last.

## Architecture additions

- **Snapshot-frontier BFS**: compressed pickle snapshots stored in the queue
  (no O(depth) history replay per node); pickle is ~2x faster than deepcopy.
  `sys.modules['game_mod']` registration makes game objects picklable.
- **Multiprocess expansion** (`workers=N`): one task per node (snapshot ships
  once), workers apply all actions and return (hash, win, child-snapshot,
  histogram).
- **Resumable search**: frontier (queue/heap + visited + counters) persists
  to disk on timeout and resumes on the next invocation. Solved levels
  persist to `v13_bfs_cache_<game>.json` and are hydrated at runtime by the
  agent (instant replay, no re-solving).
- **Greedy strategy** (`--strategy greedy`): best-first on "progress events"
  (color-histogram changes = map interactions like key pickups); falls back
  automatically to unmasked hashing when the masked space proves a dead end.
- **Hardware auto-profile** (`HW`): detects M1 Pro (mps, spawn ctx, 8
  workers), Kaggle T4 (cuda std), or RTX 6000 (>30GB VRAM → 4x wider
  ForgeNet, bf16 autocast, TF32, torch.compile, 2M replay buffer). One
  codebase: build local, deploy anywhere. Torch is optional — without it a
  numpy experience-bandit replaces the CNN fallback.

## RTX 6000 readiness (audited twice; fixes applied)

- pretrained weights load BEFORE `torch.compile` (compile prefixes
  state-dict keys with `_orig_mod.`; the old order silently dropped them)
- adaptive training batch `min(bsz, len(buf))` with floor 64 — the 2048
  batch otherwise gated training OFF on small levels (CLTI too)
- **GPU OOM backoff**: bsz=2048 × mult=4 stores ~40GB of activations at
  64x64; on OOM the batch halves permanently and the run continues
  (a 48GB Ada card WILL trigger this; expect a settle at 512-1024)
- single H2D transfer per training batch (was bsz separate copies)
- bf16 autocast inference + `cudnn.benchmark` for fixed 64x64 inputs
- solver scripts set `CUDA_VISIBLE_DEVICES=''` before importing the agent:
  the pre-solver is pure CPU, this avoids unsafe fork-after-CUDA-init in
  the worker pools and gives max workers on big multi-core GPU boxes
- caveats: the mult=1 checkpoint barely transfers into the 4x net (only
  shape-matched layers load — train a mult=4 checkpoint if the CNN path
  becomes the scorer), and NONE of the cuda branches have run on real
  hardware yet — smoke-test with
  `python v13/play_game.py --game ls20 --fast` and confirm the log says
  `RTX_6000 device=cuda` before any long run.

## How to run

```bash
source .venv312/bin/activate

# offline pre-solve (resumable; run as long as you like)
python CommunitySolutions/chronos_solver/v13/solve_offline.py --game ls20 \
    --budget 600 --bfs-timeout 300 --workers 8

# tougher levels often benefit from greedy:
python CommunitySolutions/chronos_solver/v13/solve_offline.py --game ls20 \
    --level 5 --strategy greedy --budget 600

# replay + scorecard (uses the cache; logs to v13_run.log)
python CommunitySolutions/chronos_solver/v13/play_game.py --game ls20 --fast
# V13_BFS_TIMEOUT=10 caps in-play solving when relying on the cache
```

Frontier checkpoints live in `/tmp/v13_frontier_<game>_L<n>.*.pkl`;
solutions live in `v13_bfs_cache_<game>.json` next to the agent.

## Full workflow (the three commands that matter)

```bash
source .venv312/bin/activate   # from repo root

# 1. SOLVE — sweep all 25 games, 3 rounds, resumable, Ctrl-C-safe.
#    Solutions -> v13_bfs_cache_<game>.json
#    Live status -> v13_progress.json | Full log -> v13_run.log
python CommunitySolutions/chronos_solver/v13/solve_all.py
#    (faster shallow sweep: --rounds 1 --level-budget 300)

# 2. VERIFY — after (or during) the sweep: replay every cache through the
#    real environment and collect official scorecards.
#    Output -> v13_scorecards.json + per-game summary table
python CommunitySolutions/chronos_solver/v13/verify_all.py

# 3. INSPECT — any game where env replay completes fewer levels than the
#    cache claims is a replay desync (report it); any level still missing
#    is solver work for the next iteration.
cat CommunitySolutions/chronos_solver/v13/v13_scorecards.json
```

Keep the Mac awake during the sweep (`caffeinate -is` in a spare terminal).
A full 3-round sweep is roughly 18-24h; Round 1 alone (~7-9h) banks most
of the readily solvable levels.
