# Chronos v19

BFS-first ARC-AGI-3 agent (live white-box search on the shipped game sources, with
a black-box ForgeAgent fallback). See `research/` for the full story.

## Layout

```
v19/
├── src/         # all code + its co-located weights/data (kept together so the agent's
│                #   relative paths + Kaggle staging keep working). Entry: combined_agent.py
│                #   (class MyAgent). forge_agent.py = black-box fallback. engine.py /
│                #   blackbox_env.py / _pydantic_shim.py are vendored from old v17/v18 so
│                #   v19 is self-contained. The offline flywheel lives here too:
│                #   exit_cycle.sh -> solve_all.py -> harvest_wm.py -> train_wm_v19.py.
├── tests/       # the gates (run the real agent, cache OFF):
│                #   benchmark.py + benchmark.json  -> ar25/ls20 solve-depth must not regress (pre-commit)
│                #   test_ls20.py                   -> ls20 reaches L4 (pre-push)
│                #   play_game.py / play_bfs.py     -> watch/play a game
│                #   hooks/{pre-commit,pre-push}    -> installed into .git/hooks
├── research/    # design + research notes (PROGRESS, V19_OVERVIEW, KAGGLE, SCALING,
│                #   TESTING, LADDER_PLAN, WM_REPR_EXPERIMENT, CHRONOS_EVOLUTION, …)
└── notebooks/   # Kaggle submission (v19-to-kaggle.ipynb) + vast.ai training notebooks
```

## Quick start

```bash
source ../../../.venv312/bin/activate          # repo venv (python 3.12 + torch + arcengine)

# watch the agent solve a game live
python tests/play_game.py --game ls20

# the gates (genuine, no cache)
python tests/benchmark.py            # ar25 + ls20 depth must not regress
python tests/test_ls20.py            # ls20 reaches L4

# offline flywheel (needs many CPU cores; trains the priors)
src/exit_cycle.sh
```

Guardrails are enforced on `git commit` (benchmark) and `git push` (ls20→L4). Details
in `research/TESTING.md`. Kaggle submission steps in `research/KAGGLE.md`.
