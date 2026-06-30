# Chronos v20 — the staged cascade (Phase 1: notebook-only)

v20 is a **research notebook** that references **v19 code read-only** (`v19/src`) and adds
all new logic **in v20 only**. It does not modify the shipped v19 agent.

## The cascade
For every level we run an **escalating cascade**, stopping at the first stage that returns a
**replay-verified** solution. Each stage gets its own time budget (default **~10 min**):

| # | Stage | What it is | Source |
|---|---|---|---|
| 1 | **Memory** | recall the stored plan for the exact level, **replay-verify** on a clean engine (HIT = zero search; STALE/MISS → escalate) | v19 corpus + `BFSSolver.verify_solution` |
| 2 | **BFS** | genuine white-box v13/v17 `auto` ladder | v19 `BFSSolver.solve_level` |
| 3 | **Forge** | pretrained black-box policy (ChangeNet) driven on a forked engine, focused on the wall level | v19 `ForgeAgent` |
| 4 | **LADDER** | variant re-rooting (Go-Explore) + TTRL suffix-BFS, verified | v20 (net-new, from the LADDER Stage-0 nb) |

Any stage that wins **commits its verified plan**, so the next level chains from it. All
search runs on a **forked simulator** → zero scored actions (free under RHAE).

## Honesty contract (carried from v19 memory)
- Memory is **verified recall**, never blind replay: a recalled plan that fails replay-verify
  is discarded and the level falls through to genuine search.
- BFS / Forge / LADDER are all genuine solving.
- `V19_STORE_SOLUTIONS=0`, `V19_CACHE_FALLBACK=0` during runs (no hidden cache).

## Phase 1 goal → migrate trigger
Crack **ls20 L5** and **ar25 L2** through the cascade. **Only then** do we migrate (Phase 2)
the winning cascade into a real v20 agent package and a Kaggle submission.

## Memory bank (v19 + archive)
The cascade aggregates the v19 corpus **and the v12/v13/v17 archive BFS caches**
(`archive/v*/​*bfs_cache_<gid>.json`), then **replay-verifies each plan on the live game
version**. Verified result:
- **ls20** → **L0–L4** chain (from v12/v13) verifies → cascade reaches the real wall **L5** in
  seconds.
- **ar25** → only **L0–L1** exists anywhere → genuine wall at **L2**.

**Version matters:** `environment_files/<gid>/` can hold multiple version-hashes that are
*different puzzles*. The notebook asks `arc_agi` which version the scored engine loads (the
*latest*, e.g. `ls20-9607627b`) and solves/verifies against exactly that — older hashes are
stale and their plans won't transfer.

Notebook: [`notebooks/v20_cascade.ipynb`](notebooks/v20_cascade.ipynb)
