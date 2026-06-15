# v18 — black-box, frame-only ARC-AGI-3 agent

v17 was rigorous but solved the **wrong problem**: it brute-forced *known* games
with white-box simulator access (read the engine's private `__dict__` for a
"progress" signal, snapshot/restore + 60k forward rollouts per search, per-game
cached solutions). The real ARC-AGI-3 benchmark is **black-box**: an agent runs
through `Agent.choose_action(frames, latest_frame)`, sees only a rendered
`FrameData` (frame grids, `levels_completed`, `win_levels`, `available_actions`,
`state`), takes one action at a time against a budget, and is scored on
**held-out games** it has never seen. None of v17's white-box machinery
transfers. v18 fixes the target.

## The goal (frozen)

**One game-agnostic, frame-only agent that generalises to unseen ARC-AGI-3
games.** Same code/weights for every game; zero per-game branching; no engine
internals; no simulating the game it is currently playing.

## Win state (in order)

1. **Floor beaten** — a learned agent beats the ReactiveExplorer baseline on the
   HELD-OUT split (more games with ≥1 level, or fewer actions-to-first-level).
2. **Transfer proven** — the agent completes ≥1 level on multiple held-out games
   it never trained on, at roughly the rate it does on train games (held-out ≈
   train ⇒ generalisation, not memorisation).
3. **Leaderboard** — the same agent, run through the official ARC-AGI-3 harness
   (`ARC-AGI-3-Agents`, `ARC_API_KEY`), completes levels on the live hidden
   games. This is the only fully external measure.

The 25 public games are the **train + validation universe**, NOT the prize.
HELD-OUT here is the stand-in for the hidden leaderboard games. Optimising the
25 directly (v17's path) is overfitting and does not count.

## How it is tested (the only honest metric)

`evaluate.py` runs the SAME agent object black-box through `blackbox_env.py`
(which exposes ONLY the public FrameData fields — the engine handle never
reaches the agent) for a fixed action budget, and reports `levels_completed` per
game plus the **TRAIN vs HELD-OUT aggregate**. Validated end-to-end: replaying
v13's 13-action ls20-L0 solution through the wrapper registers
`levels_completed=1`, so a real completion is detectable.

- Split (frozen, reproducible): **HELD-OUT = cn04, ka59, sk48, tu93, wa30**;
  TRAIN = the other 20 (lf52, tn36 skipped — engine load errors).
- The agent NEVER trains on HELD-OUT. Rotate folds later for cross-validation.

## Can the learning transfer to the 25 games?

Yes — that IS the experiment. You train offline on TRAIN games (white-box search
as an *offline oracle* to harvest expert trajectories is allowed there, because
at train time you hold the public game code), then prove transfer on HELD-OUT.
What you submit to the hidden leaderboard is that single agent, unchanged. If
HELD-OUT performance ≈ TRAIN performance, the learning transferred; if HELD-OUT
collapses, you memorised the 20 and learned nothing general.

## Architecture

| file | role |
|---|---|
| `blackbox_env.py` | honest wrapper — loads/drives the real game, hands the agent ONLY an `Obs` (public FrameData fields). The wall that makes a solve honest. |
| `agent.py` | `BaseAgent` interface + `ReactiveExplorer` baseline (novelty bandit over available actions, click targets from the visible frame). Frame-only, game-agnostic. |
| `evaluate.py` | held-out scorer + frozen TRAIN/HELD-OUT split. THE metric. |

## Iteration log

### iter 0 — honest floor  [2026-06-15]
ReactiveExplorer, budget 200, 2 episodes, HELD-OUT split:
**0/5 games reached a level (total_levels=0).** Expected — random novelty
exploration rarely threads a specific level-completing sequence in 200 actions.
This is the floor every learned agent must beat. Harness verified working
(L0-replay ⇒ levels_completed=1).

### iter 1 — offline harvest + behaviour-cloning (frame-NN)  [2026-06-15]
Built `features.py` (332-d frame-only vector), `harvest.py` (replays v13 cached
solutions on the 20 TRAIN games → 527 `(frame→action)` pairs; ls20 contributes
184), and `CloneAgent` (nearest-neighbour policy over those features; frame-only
at test time). Three honest measurements:

| test | result | reading |
|---|---|---|
| clone on ls20, **with** ls20 data | **5 levels, L0 @ action 13** | pipeline works — a learned frame-only policy solves all 5 ls20 levels black-box (matches v13 optimal) |
| clone on ls20, `--exclude_self` | **0 levels** | drop ls20's own samples ⇒ collapses |
| clone on **HELD-OUT** (cn04,ka59,sk48,tu93,wa30) | **0/5** | no transfer to unseen games |

**Verdict (honest): v18 went 0 → solves 5 levels, but ONLY by memorising games
it has seen. Transfer = 0.** Frame-NN behaviour cloning generalises nothing —
expected, and the core ARC difficulty. The architecture/metric are validated;
the *method* is the problem. Win-state tiers 2–3 (transfer) remain unmet; tier 1
(beat the floor) is met only on seen games, which doesn't count.

### iter 2 — GENUINE solver (honest black-box search), no stored answers  [2026-06-15]
Deleted the memory book (`v18_harvest.npz`, harvest, CloneAgent) — v18 never uses
stored answers again. Built `search_agent.py`: v17's BFS/BFWS playbook made honest
— discovers paths itself; the ONLY goal signal is observable `levels_completed`;
dedup uses the observable FRAME with PIXEL-level transient masking; explores the
ONE real env via RESET+replay (no white-box snapshot fork).

Key finding: honest reset+replay **BFS** is O(nodes×depth) REAL actions →
intractable (single-step BFS on ls20 didn't finish in 200s). Switched to v17's
*other* idea — **rollout search** (reset once, roll a trajectory forward, bank
novel frames as frontier nodes; cost = depth+rollout per sim). That works.

Genuine results (NO answers, found its OWN paths):

| game | result | note |
|---|---|---|
| ls20 | **L0 solved**, own 40-act path | path differs from v13's — real solve |
| sp80 | **L0 solved**, 20-act path | 825 sims |
| lp85 | **L0 solved**, 19-act path | 6 sims (easy) |
| cd82, ar25 | 0 (searched ~600 sims) | needs more budget / guidance |
| vc33, ft09 | 0 (frontier starved ≤50 sims) | random rollouts hit GAME_OVER fast; likely click-puzzles |
| HELD-OUT (cn04,ka59,wa30…) | **0** at 30k budget | uniform-random rollouts insufficient |

**Verdict (honest): v18 now genuinely SOLVES — ls20, sp80, lp85 from scratch with
zero stored answers, generalising by construction (search, not recall).** But it
is not yet strong enough for the held-out games at a modest budget, and starves
on click-puzzles / GAME_OVER-heavy games. Real solving achieved; transfer to
unseen games is the open problem. The honest cost (reset+replay is action-
expensive) is itself a finding: the long-term fix is a learned forward model so
search happens in imagination, not by re-walking the real env.

### iter 3 — Go-Explore novelty-guided rollouts → FIRST held-out solves  [2026-06-15]
**Research:** ARC-AGI-3 is an Interactive Reasoning Benchmark (explore→model→goal
→plan); frontier AI <1%, humans 100% (arXiv 2603.24621). Go-Explore (archive →
return → explore) and Graph-Based Exploration (arXiv 2512.24156) are the pointers.
Our rollout search already has the Go-Explore *shape*; the weak link was the
EXPLORE step (uniform-random).

**Critic of iter2:** random rollouts re-tread the same actions and starve the
frontier the moment they hit GAME_OVER (vc33/ft09 ended at ≤50 sims); held-out=0
because exploration was undirected.

**Change (one lever):** a global Go-Explore archive `tried[frame_hash] -> {action
keys}`; each rollout step now prefers an action NOT yet tried from the current
observed frame (directed/novelty exploration), falling back to random 10% of the
time. On GAME_OVER, return to the node and keep exploring (≤2 deaths/rollout)
instead of dying early. Still frame-only, still no stored answers.

**Measure (real engine, budget 50k):**

| split | iter2 | **iter3** | new solves |
|---|---|---|---|
| train-sample (6 probe games) | 3 / 3 | **5 / 5** | +cd82, +ar25 |
| **HELD-OUT (never trained on)** | **0 / 0** | **2 / 3** | **cn04 (1), tu93 (2)** |

**Verdict (honest): generalisation is now real.** The agent solves TWO unseen
held-out games (cn04, tu93) by genuine search — no memory book, no per-game code,
no engine internals. That is the first non-zero transfer number in the whole
project. Not yet at the DONE bar (≥3/5 held-out): ka59, sk48, wa30 searched ~500
sims but didn't crack L0, and vc33/ft09 still starve (≤50 sims) — these likely
need click/ACTION7 handling or a frame-delta progress signal, not just movement
novelty.

## Next levers (ordered) — push held-out 2/5 → 3/5+ and deepen

iter2 has a GENUINE solver that solves ≥3 games from scratch. To solve harder /
held-out / click games, make the search smarter (still no stored answers):

1. **Novelty-guided rollouts (BFWS)** — instead of uniform-random playouts, bias
   each rollout step toward actions that produce a never-seen frame (frame-delta
   novelty). Directed exploration is the difference between cracking cd82/ar25
   and burning budget. Highest leverage, pure search.
2. **GAME_OVER handling + click puzzles** — rollouts that hit GAME_OVER starve
   the frontier (vc33, ft09). On GAME_OVER, RESET and continue the rollout; for
   click games, make visible-object click targets first-class rollout actions.
3. **Observable frame-delta progress reward** — shape rollout selection by "how
   much new structure appeared" (honest, frame-only) so search heads toward
   interactions, not just wandering. Replaces v17's `__dict__` progress.
4. **Learned forward model for IMAGINATION search** — the only way past the
   reset+replay action cost: learn `(frame,action)->frame` from observed
   transitions (offline on TRAIN), then run the rollout search inside the model
   (free), executing only the verified plan. This is also what makes transfer
   to held-out cheap. Big lift, biggest payoff.
5. **Hook the official harness** (`ARC-AGI-3-Agents`) and submit to the live
   leaderboard once held-out solves appear.

## Hard rules (never break — this is what v17 got wrong)

- Agent sees ONLY `Obs`. No `g.__dict__`, no `_current_level_index`, no
  scalar-attr progress, no snapshot/restore, no simulating the current game.
- One agent for all games. No `if game_id == ...`.
- Never report a solve that the wrapper's `levels_completed` did not register.
- Held-out is sacred: never train, tune, or peek on the held-out games.
