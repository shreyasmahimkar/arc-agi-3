# Chronos v21 — generalize-first ARC-AGI-3 agent + 4h cadence

v21 = v20's proven cascade, made **RHAE-optimal, offline-safe, and self-improving on a
4-hour cadence**, validated against 3 games chosen as a generalization source of truth.

- **`PLAN.md`** — the architecture, the research that justifies it, phased roadmap.
- **`REQUIREMENTS.md`** — itemized, reviewable requirements (R1–R7) with acceptance checks.
- **`requirements.txt`** — deps (submission = offline; cadence box = optional LLM).
- **`cadence_runner.py`** — the 4-hour flywheel: solve→verify→shortest→scorecard→macros.
- **`install_cron.sh`** — install the every-4-hours schedule locally.

## The 3 games (one per capability tier)
| Game | Tier | Ground-truth win | Stresses |
|---|---|---|---|
| `ls20` | reasoning / keyboard-maze | BFS-optimal 13/45/39/43/44 (or ACTION2×129 blind) | multi-step agent reasoning |
| `ft09` | reflex / blind | single ACTION6 | one-shot hypothesis |
| `vc33` | orchestration / click | 1 click | click-target selection |

Solve all three *well* (shortest, verified) and you span the whole public-set taxonomy.

## Run it
```bash
# one pass (offline, default 3 games)
PYTHON=../../../.venv312/bin/python python3 cadence_runner.py --bfs-timeout 180

# install the 4-hour cadence (escalating budget)
PYTHON=/abs/path/.venv312/bin/python ./install_cron.sh
```
Outputs: `logs/scorecard.jsonl` (per level + per game RHAE), `logs/rhae_history.csv`,
`logs/last_summary.md`, updated `solutions/<gid>.json` + `v21_macro_bank.json`.

## Guardrails
- **Offline**: the runner installs a socket guard; the submission path makes zero network calls.
- **Regression gate**: a run never ships a corpus with lower per-game RHAE than the last.
- **Shortest-wins**: only strictly-shorter verified plans replace cached ones (RHAE monotone).
- **No exploit**: the null-coordinate ACTION6 TypeError "win" is refused.

See `PLAN.md` §7 for the open questions to review (white/black-box on the private set,
official human baselines, engine-version rotation, world-model compute).
