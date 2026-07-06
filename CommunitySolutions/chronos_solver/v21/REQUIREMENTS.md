# Chronos v21 — Requirements (for review)

Reviewable, itemized requirements for the ARC-AGI-3 Kaggle agent + its 4-hour cadence.
Each item has an ID, a priority (**P0** must-have → **P3** nice-to-have), and an
acceptance check. Sources for the design rationale are listed at the bottom.

Legend: ✅ already exists in v19/v20 · ⚡ = directly moves RHAE.

---

## R1. Scoring & correctness (the RHAE contract) ⚡

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R1.1 | P0 | Every committed level plan must be **replay-verified** on a clean forked engine before use; a plan that fails verify is discarded, not shipped. ✅ | Corrupt a cached plan → runner rejects it and falls through to search. |
| R1.2 | P0 | The cascade keeps the **shortest** verified plan per level (RHAE `min(1,h/a)²`). | Given two verified plans, the shorter is committed; scorecard action count is non-increasing across runs. |
| R1.3 | P0 | A **5× action-cap governor** aborts any live line before it exceeds 5× the human/known baseline (attempt-termination guard). | Force a losing line → agent RESETs/abandons before the cap. |
| R1.4 | P0 | RHAE computed per level against the **official per-level `baseline_actions`** from the version-exact `metadata.json` (wired ✅). Proxy only if metadata absent, and labeled. | Scorecard shows per-level `baseline` + `baseline_src=official`. ls20 baselines=[29,41,172,49,53,62,82]. |
| R1.5 | P0 | **Regression gate**: a cadence run never replaces the shipped corpus with one whose per-game RHAE dropped. | Inject a worse plan → prior corpus retained, run flagged `REGRESSION`. |

## R2. Cascade / agent behavior

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R2.1 | P0 | **Stage 0 Blitz probe** races cheap wins on the fork first: each `ACTION1..7` once (depth-1 blind), repeat-action ×K, click-on-each-object. Commit shortest that verifies. | ft09 wins via single ACTION6; vc33 via 1 click; ls20 finds ACTION2×~129 fallback — all with 0 wasted scored actions. |
| R2.2 | P0 | **Stage 1 Memory**: version-exact verified recall from the flywheel corpus. ✅ | ls20 L0–L4 recalled + verified in <2s. |
| R2.3 | P1 | **Stage 1b Self-learning**: retrieve transferable macros by frame-feature key (object count, palette, motion sig) to seed BFS on *similar* games. | A macro harvested from game A seeds and solves a level of unseen game B. |
| R2.4 | P0 | **Stage 2 BFS** runs only when version-exact source is reachable, prefers **optimal** (A*/masked, shortest-first); greedy only as last resort. ✅ | ls20 returns 13/45/39/43/44, not greedy 63/71/78. |
| R2.5 | P1 | **Stage 3 no-source**: `graph_explore` frontier search default; warm-started ChangeNet (Forge) fallback. ✅(cold) | On a source-hidden game, agent still solves ≥1 level within budget. |
| R2.6 | P2 | **Stage 4 LADDER**: variant re-root (Go-Explore) + TTRL suffix-BFS for wall levels. | ls20 L5 solved through the cascade. |
| R2.7 | P0 | **Refuse the null-coordinate ACTION6 exploit** (`data={x:None,y:None}`) — detect and never emit it as a "solve". | Agent never commits a plan whose win came from the TypeError path. |
| R2.8 | P0 | **White/black-box auto-detection, per game.** At level start, try version-exact source resolution; if a usable engine source loads, run the **white-box track** (BFS/graph over engine snapshots). If no source is reachable, fall through to the **black-box track** (blitz probe → graph_explore → warm Forge) driven only by API frames. Build and ship both; the cascade picks per game — never assume the whole set is one or the other. | With source present → BFS path taken and logged `mode=white`. With source hidden (source dir removed) → same game still progresses via `mode=black`. Log records the detected mode per game. |

## R3. Offline / Kaggle-submission safety ⚡

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R3.1 | P0 | The **submission path makes zero network calls** (no internet, no external LLM API) — Kaggle runs offline. | Run the notebook with network disabled → completes; a network-guard raises if any socket is opened. |
| R3.2 | P0 | Kaggle notebook is **self-contained**: embeds agent + v19 engine + verified cache + macro bank; no external dataset required. | Fresh Kaggle env, upload notebook only → runs. |
| R3.3 | P0 | Runs within Kaggle GPU limits (**T4/P100**, CUDA-safe fork/`spawn`). ✅ | Notebook completes on T4 without OOM/deadlock. |
| R3.4 | P1 | **Version-exact source resolution**: always load the newest version-hash the scored engine uses; never a stale hash. ✅ | With two ls20 hashes present, the newer is loaded and verified against. |
| R3.5 | P1 | `V21_STORE_SOLUTIONS`, `V21_CACHE_FALLBACK` env flags default to "no hidden cache" during honest eval. ✅(v19) | Honest-eval run shows no silent cache hits. |

## R4. The 3-game generalization harness

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R4.1 | P0 | Fixed probe set = **`ls20`, `ft09`, `vc33`** (one per capability tier: reasoning / reflex / click-orchestration). | Runner targets exactly these 3 by default (`--games` overridable). |
| R4.2 | P1 | Each run reports **per-tier** outcome so a fix to one game can't silently regress another (e.g. the v19 ls20↔vc33 grid tradeoff). | Scorecard has a row per game with tier label. |
| R4.3 | P2 | A **held-out probe** (e.g. `cn04, sk48, tu93`) runs weekly to measure true generalization (no stored answers). | Weekly row: held-out games/levels solved by search only. |

## R5. Flywheel corpus & memory

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R5.1 | P0 | Corpus is **append/replace-shorter only**, JSON per game (`solutions/<gid>.json`), fully resumable. ✅ | Interrupt a run → resume continues from last solved level. |
| R5.2 | P1 | **Macro bank** (`v21_macro_bank.json`): frame-feature key → macro sequence, harvested after each run. | New solutions add ≥1 macro; retrieval hit demonstrated (R2.3). |
| R5.3 | P1 | Every run **replay-verifies the whole corpus** against the current engine version and prunes stale entries. | After an engine-version bump, stale plans are pruned, not shipped. |

## R6. 4-hour cadence runner ⚡

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R6.1 | P0 | `cadence_runner.py` runs the full pipeline for the 3 games unattended and exits cleanly. | One invocation produces a scorecard + logs and returns 0. |
| R6.2 | P0 | Scheduled to run **every 4 hours** (cron `0 */4 * * *`), with escalating BFS budget across passes (180→600→1800 s/level). | Scheduled task registered; 2 consecutive runs show budget escalation on unsolved levels. |
| R6.3 | P0 | **Append-only** run log + machine-readable scorecard (`logs/scorecard.jsonl`), one row per run per game. | Two runs → two sets of rows, never overwritten. |
| R6.4 | P1 | Each run ends with a **≤5-line human summary** (games improved, RHAE deltas, regressions) suitable for a notification. | Summary printed + saved to `logs/last_summary.md`. |
| R6.5 | P1 | Idempotent & lock-guarded: a run won't start if the previous one is still active. | Launch twice → second exits with "already running". |
| R6.6 | P2 | Optional offline **world-model generator** step (LLM, cadence box only) distills plans into the corpus; **disabled by default** and never in the submission. | `--world-model` flag runs it; default run touches no LLM. |

## R8. Code-writers (runtime + evolve) ⚡

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R8.1 | P1 | **Runtime on-the-fly writer** (`runtime_coder.py`): a LOCAL LLM writes a Python `WorldModel` per level from observed transitions; we exec→verify→plan→test on the fork and commit the shortest win. Offline-eligible (local model, no API). | With mock backend, writes+runs a model that solves ft09-like (1×ACTION6) and ls20-like (repeat ACTION2). ✅ tested |
| R8.2 | P0 | The world-model exec runs in a **restricted sandbox** (whitelisted builtins + imports numpy/math only, exec timeout). No file/network/system access from generated code. | Generated code importing `os`/`socket` raises; class-only numpy code runs. ✅ tested |
| R8.3 | P0 | Generated plans pass the **same exploit refusal** (R2.7) and **replay-verify** (R1.1) as every other stage. | A generated null-coord ACTION6 plan is refused. ✅ tested |
| R8.4 | P1 | **Pluggable backends** (`llm_backend.py`): `hf` (local Qwen2.5-Coder), `openai` (cadence box only), `mock` (offline test). Auto-selects hf→mock; openai never in the offline path. | `get_backend()` returns mock with no GPU; hf when transformers+model present. ✅ tested |
| R8.5 | P1 | Model runs **offline** on Kaggle: weights bundled as a dataset, `HF_HUB_OFFLINE=1`, 4-bit fits the T4; agent degrades gracefully to non-LLM stages if the model is absent. | Notebook with model dataset + network off loads the model and runs. |

## R9. Self-evolution & intuition (the improving loop) ⚡

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R9.1 | P0 | **Intuition prior** (`intuition.py`) is re-distilled from the whole corpus **every cadence run** — an action-ordering prior (early-move-weighted) that biases blitz/search. Interface `order_actions(game,frame)` is fixed so a trained net can drop in. | After a run, `intuition_prior.json` exists; `order_actions('ls20')` returns a ranked list. ✅ tested |
| R9.2 | P1 | **Champion/challenger evolution** (`evolve.py`): the code-writer proposes config/heuristic challengers targeting the wall levels; each is evaluated and a challenger is promoted **only if it beats the champion on HELD-OUT without regressing train** (generalization-gated). | Flat evaluator → no promotion; a strictly-better held-out challenger → promoted, `champion.json` version bumps. ✅ tested (flat case) |
| R9.3 | P0 | Nothing the LLM writes ships **unverified**: challenger heuristics run in the R8.2 sandbox; promotion RHAE comes only from replay-verified solves. | Evolution history row records train/held RHAE + promoted flag per run. ✅ tested |
| R9.4 | P1 | **Config-aware live evaluator** (P2 upgrade): evolve's `eval_fn` applies `blitz_K`/`action_order` to the real engine so challengers can actually win walls (today it's a config-insensitive corpus floor that never promotes on noise). | Challenger raising `blitz_K` solves a budget-gated level the champion missed. |
| R9.5 | P2 | **Offline WM generator** (cadence box, `--world-model` + `--allow-network`): a larger LLM synthesizes/distills plans into the corpus; **never** in the submission. | Flag runs it; default run touches no LLM/network. |

## R7. Non-functional

| ID | Priority | Requirement | Acceptance check |
|---|---|---|---|
| R7.1 | P0 | References v19/v20 code **read-only**; all new code lives under `v21/`. | `git diff` touches only `v21/`. |
| R7.2 | P1 | A single cadence run over 3 games completes within one pass budget (< ~90 min at 1800s/level worst case, early-exit on cache hits). | Timed run under budget. |
| R7.3 | P1 | Deterministic/reproducible given a fixed corpus + engine version. | Same inputs → same committed plans. |
| R7.4 | P2 | Metrics also emitted as CSV for quick plotting of RHAE-over-time. | `logs/rhae_history.csv` grows one row/run/game. |

---

## Environment / dependencies
See `requirements.txt`. Core: Python 3.12, numpy, scipy, torch (CUDA on Kaggle T4),
the vendored `arcengine` + `ARC-AGI-3-Agents` harness. No network deps in the submission path.

## Sources (research basis)
- ARC-AGI-3 Technical Report — arXiv:2603.24621 (RHAE, black-box, 5× cap)
- Explore Before You Solve (AERA, v31 kernel RHAE=0.30, 25-game taxonomy) — arXiv:2605.25931
- Executable World Models for ARC-AGI-3 (58% public, LLM coding agent) — arXiv:2605.05138
- Graph-Based Exploration for ARC-AGI-3 — arXiv:2512.24156
- BentoLabs self-learning layer (+2.6×, no retraining) — bentolabs.ai/blog
- StochasticGoose / Tufa preview (12.58%) — github.com/DriesSmit/ARC3-solution
- ARC Prize 2026 competition + docs — arcprize.org / docs.arcprize.org
- Internal: `v20/research/LEADERBOARD_RESEARCH.md`, `v19` SCORECARD/PROGRESS logs
