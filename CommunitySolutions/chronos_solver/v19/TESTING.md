# v19 guardrail — genuine ls20 solve gate

**Rule:** every code change must keep the v19 agent able to **genuinely** (no cached
answers) solve ls20 through its multi-level transitions, using the full v13/v17
search ladder. This catches regressions in the live solving path before they ship.

## The gate

```bash
cd CommunitySolutions/chronos_solver/v19
../../../.venv312/bin/python test_ls20.py            # default: clear L0-L3 (reach L4)
../../../.venv312/bin/python test_ls20.py --target 5 # stretch: also clear L4 (offline-only, see below)
```

It drives the **real agent** (`combined_agent.MyAgent`) through one continuous
ls20 episode on the live `arc_agi` engine with **`V19_STORE_SOLUTIONS=0` and
`V19_CACHE_FALLBACK=0`** — so a PASS means the agent searched and solved each level
itself. Exit 0 = PASS, 1 = FAIL. Runtime ~4-6 min (the `auto` ladder is thorough).

## Pre-push hook (enforces the gate)

Installed as a symlink to the tracked copy:

```bash
ln -sf ../../CommunitySolutions/chronos_solver/v19/hooks/pre-push .git/hooks/pre-push
```

`git push` runs the gate and blocks on failure. Emergency bypass:
`SKIP_LS20_GUARDRAIL=1 git push` (discouraged).

## Why target = 4 (reach L4), not 5 (clear L4)

`solve_all.py` found ls20 L4 (44 actions) only via a **persistent frontier**
(`/tmp/v19_frontier_*.pkl`) that accumulates search across **many** flywheel
cycles. A single live episode gets one bounded shot per level, so clearing the
44-deep L4 in one pass isn't feasible — that's an offline/multi-cycle capability,
not a live-solve regression. Clearing **L0-L3 in one live episode (reaching L4)**
is the meaningful gate: it exercises every multi-level transition that regressed.

## Two regressions this gate caught (and the fixes)

1. **Version mismatch** — `play_game.py`/`test_ls20.py` overrode the env's
   `environment_info.local_dir` with a bare glob that picked a *different* ls20
   version (`cb3b57cc`) than the env ran (`9607627b`). The BFS solved the wrong
   version, so plans failed at the first divergent level (L1). Fix: never override
   the env's `local_dir` — it names the exact version loaded. (1/5 → 4/5.)
2. **Plain BFS vs the v13/v17 ladder** — the live agent's `_try_bfs_solve` called
   `solve_level` with the default `strategy='bfs'`, while the corpus solver uses
   `strategy='auto'` (bfs → waypoint → A* → IW → EHC → greedy). Plain BFS times out
   on deep levels. Fix: the agent now uses `strategy='auto'` — "all that v17 did."

Safety nets also added: `verify_solution` (clean-engine replay check) +
`solve_level_deterministic` (workers=1 re-solve) + `V19_BFS_WORKERS` env override.
