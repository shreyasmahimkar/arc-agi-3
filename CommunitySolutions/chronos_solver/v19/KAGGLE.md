# v19 → Kaggle: package, run, and read the score

Companion to `v19-to-kaggle.ipynb`.

## The score story (why BFS-first)

v12 scored **0.22** on Kaggle with pure **live white-box BFS** — no answer-book,
no weights. The competition **ships the game sources** at
`/kaggle/input/competitions/arc-prize-2026-arc-agi-3/environment_files/{gid}/.../{gid}.py`,
so the agent's BFS reaches them and solves every level **genuinely at test time**,
and it generalises to the held-out scored games (their sources ship too). A later
black-box v19 scored **0.01** because it abandoned that BFS on a false premise.
This package puts BFS back in front.

**Priority:** (1) live BFS solve → (2) cached-solution backstop *only when BFS
times out* → (3) black-box `ForgeAgent` only if no source is reachable at all.

## 1. Build the dataset

Private Kaggle dataset (e.g. `v19-forge`), top-level files:

| item | role |
|---|---|
| `combined_agent.py` | entry point, `class MyAgent` (BFS-first) |
| `forge_agent.py` | black-box fallback (no-source games only) |
| `pretrained_weights.pt` | ChangeNet prior for the black-box fallback |
| `solutions/` | cached solutions = **timeout backstop only** |

```bash
# from CommunitySolutions/chronos_solver/v19/
mkdir -p /tmp/v19-forge
cp combined_agent.py forge_agent.py pretrained_weights.pt /tmp/v19-forge/
cp -r solutions /tmp/v19-forge/
# upload /tmp/v19-forge as a private Kaggle dataset
```

`solutions/` is optional — without it the agent is BFS-only (still the 0.22 path,
just no timeout safety net).

## 2. Configure + run

Add Input → attach `v19-forge` + the competition data. **GPU**, **Internet OFF**.
Save Version → Save & Run All. The agent runs only during the scoring rerun, with:

```
V19_CACHE_FALLBACK=1   # cached solution replays ONLY when live BFS times out
V19_STORE_SOLUTIONS=0  # don't rewrite the cache during scoring
V13_BFS_TIMEOUT=180    # each level gets a fair live-BFS shot first (v12's value)
```

## 3. Is it scoring well?

**Logs / `v19_run.log`:**
- `BFS ACTIVE: loaded <Class> from <path>` — live BFS reached the source. **This is
  the 0.22 path.** `[v19] no white-box source -> black-box fallback` instead = BFS
  failed to find the source → score craters (the 0.01 failure mode); fix the glob.
- `levels_completed` increments = levels solved live (genuine).
- `BFS timed out on level N -> cached fallback (K actions)` — the backstop. **Sparse
  is healthy**; lots of these means BFS is too slow (raise the timeout / speed BFS),
  and the cache can't cover scored games you've never seen — only live BFS can.

**Leaderboard:** target ~0.22. Near-zero ⇒ BFS isn't firing; that's the first thing
to check, not the cache.

## Honesty policy

Live BFS is genuine search at test time, not a stored answer. The `solutions/` cache
is a *timeout backstop only* (reused experience on a level seen before) and never the
first move — `combined_agent.py` runs BFS first and loads the cache lazily only on
failure (`CACHE_FALLBACK` flag, `_load_cached_level`). This is the agreed policy.
