# Chronos Solver v12 — FORGE v19 base, pure BFS/CNN (no LLM)

v12 starts from **FORGE v19** (`my_agent.py` with its 4 targeted fixes) and
iterates on it with pure symbolic search + CNN fallback. No LLM anywhere.

## Results so far (ls20)

| Level | v19 status | v12 actions | How |
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

Banked so far (`v12_bfs_cache_<game>.json`): ls20 5 lvls (13/45/39/43/44),
ar25 2 (15/11), cd82 2 (5/6), m0r0 2 (15/23), dc22 1 (20), ft09 1 (4),
lp85 1 (5), r11l 1 (3), s5i5 1 (13), sp80 1 (4) — **17 levels, 10 games**.
Still cold (need longer budgets): bp35, cn04, g50t, ka59, lf52, re86, sb26,
sc25, sk48, su15, tn36, tr87, tu93, vc33, wa30.

## Bugs found in v19 (each killed levels)

1. **`depth < 30` BFS cap** silently truncated any solution longer than 30
   actions (ls20 L1 needs 45+). Raised to 200 — visited-dedup already bounds
   the search.
2. **Timer-bar state aliasing**: the HUD timer changes pixels every step, so
   every (position × timer-tick) hashed unique → state explosion. v12
   auto-detects transient rows (hot under EVERY single-action rollout) and
   masks them from the dedup hash, with an automatic unmasked retry if the
   masked space exhausts (some levels are genuinely time-dependent).
3. **Hidden-countdown pruning**: ls20's lock opens via an internal countdown
   (`akoadfsur`) during which frames are pixel-identical — BFS pruned the
   chain as "visited" and the win was unreachable. v12 folds the game
   object's public scalar attrs (key shape/color/rotation, countdowns,
   player coords) into every state hash.
4. **Wrong BFS baseline for levels > 0** (the big one): `set_level(N)` + RESET
   produces a *different* start state than naturally advancing from L(N-1)
   (player position, carried key rotation, ~1400px frame diff on ls20 L1).
   Solutions solved from the synthetic baseline fail when replayed in the
   env. v12 builds level N's start state by **chaining the cached solutions
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
  persist to `v12_bfs_cache_<game>.json` and are hydrated at runtime by the
  agent (instant replay, no re-solving).
- **Greedy strategy** (`--strategy greedy`): best-first on "progress events"
  (color-histogram changes = map interactions like key pickups); falls back
  automatically to unmasked hashing when the masked space proves a dead end.
- **Hardware auto-profile** (`HW`): detects M1 Pro (mps, spawn ctx, 8
  workers), Kaggle T4 (cuda std), or RTX 6000 (>30GB VRAM → 4x wider
  ForgeNet, bf16 autocast, TF32, torch.compile, 2M replay buffer). One
  codebase: build local, deploy anywhere. Torch is optional — without it a
  numpy experience-bandit replaces the CNN fallback.

## How to run

```bash
source .venv312/bin/activate

# offline pre-solve (resumable; run as long as you like)
python CommunitySolutions/chronos_solver/v12/solve_offline.py --game ls20 \
    --budget 600 --bfs-timeout 300 --workers 8

# tougher levels often benefit from greedy:
python CommunitySolutions/chronos_solver/v12/solve_offline.py --game ls20 \
    --level 5 --strategy greedy --budget 600

# replay + scorecard (uses the cache; logs to v12_run.log)
python CommunitySolutions/chronos_solver/v12/play_game.py --game ls20 --fast
# V12_BFS_TIMEOUT=10 caps in-play solving when relying on the cache
```

Frontier checkpoints live in `/tmp/v12_frontier_<game>_L<n>.*.pkl`;
solutions live in `v12_bfs_cache_<game>.json` next to the agent.
