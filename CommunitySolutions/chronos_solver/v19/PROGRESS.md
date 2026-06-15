# v19 — FORGE black-box agent (the competition-correct architecture)

## Why this exists / the pivot

The Kaggle ARC-AGI-3 **private set is scored in competition mode: API-only, game
source unreachable, RHAE scoring** (`level_score = (human_actions/ai_actions)²`,
every action counts). That kills the FORGE-v19 white-box stack (BFS / CLTI /
transfer all need the game source). The 0.22 best-to-date almost certainly came
from the CNN *fallback* (the only black-box-legal part) or source-reachable
validation — not the BFS. So v19 here is the **black-box** agent that targets the
real scored surface (it is the user's "FORGE v20" rewrite, run locally vs v18's
`blackbox_env`).

## Architecture — fuses the two ARC-AGI-3 preview WINNERS

| winner | score | idea | in v19 |
|---|---|---|---|
| StochasticGoose (1st) | 12.58% | CNN predicts which action changes the frame → efficient exploration | `ChangeNet` (per-game, online) |
| Blind Squirrel (2nd) | 6.71% | state graph from frames; prune loop/no-change actions; frontier explore | transition graph + `_plan_to_frontier` |
| graph-exploration (3rd) | — | training-free systematic graph search (arXiv 2512.24156) | frontier planning |

World models (Dreamer) and intrinsic motivation (BYOL-Hindsight) were tried in
the preview and **lost** — finicky under sparse reward / short budgets. Top agents
were NOT LLMs. So the proven path is exactly this fusion. (arcprize.org preview
results; Dries Smit 1st-place writeup.)

Files: `forge_agent.py` (ChangeNet + graph + frontier policy, frame-only, emits
`(action_id,data)`), `eval_forge.py` (drives `blackbox_env`, reports levels +
actions-to-level = the RHAE driver).

## iter 0 — honest baseline  [2026-06-15]

Runs end-to-end, black-box, **RHAE-honest** (one continuous episode per game —
the action count IS the real cost; there is no hidden reset+replay search like
v18's, whose "19-action" solves hid ~50k probe actions).

| game | budget | result |
|---|---|---|
| lp85 (train) | 1500 | **L0 @ action 43** ✅ |
| ls20 (train) | 1500 | 0 (v18 search solved in ~40, but at 50k hidden probes) |
| sp80 (train) | 1500 | 0 |
| **HELD-OUT × 5** (cn04,ka59,sk48,tu93,wa30) | 800 | **0/5** |

Compare: v18 search got 3/5 held-out but at ~50k hidden reset+replay actions/game
(RHAE ≈ 0 on a live game). FORGE gets 0/5 at 800 HONEST actions — the cold CNN
can't explore efficiently enough to thread a goal in budget. Neither is shippable
yet; FORGE is the RHAE-correct shape (the proven preview-winner architecture) but
needs the warm prior.

**Verdict:** the agent is correct and RHAE-honest, but **exploration is
action-inefficient with a COLD CNN** — on movement mazes it doesn't thread the
goal within budget. The cold-start is the bottleneck, and it's the exact thing
the #1 lever fixes. Also slow locally (~22 steps/s, per-step CNN on MPS) — a
dev-speed issue, not a scoring one (Kaggle runs on GPU).

## iter 1 — v17 borrows: macro-actions + online transient masking  [2026-06-15]
Added (both frame-only, legal): **macro-actions** (a movement repeats until the
masked frame stops changing — collapses corridors into one graph edge) and
**online transient-pixel masking** (pixels changing on >85% of transitions are
counters → masked from the hash so the graph doesn't explode on novelty).

**Measure (real engine):**

| set | baseline (iter0) | iter1 (macros+mask) |
|---|---|---|
| held-out × 5 @800 | 0/5 | **0/5** (no change) |
| ls20 / sp80 @800 | 0 | 0 (macros OVERSHOOT mid-corridor turns — v18's documented limit) |
| lp85 @800 | 1 @43 | 1 @45 |
| cn04 @2000 | — | 0; explored 544 states; mask froze at 0 px (no perpetual counter) |

**Verdict (honest): no held-out lift.** Macros help long-corridor games but hurt
turn-heavy ones (ls20); masking is correct but cn04/most held-out games have no
perpetual counter to mask. These are sound *hygiene* fixes, not the score driver.
The diagnostic is clear: the agent explores hundreds of states but a COLD CNN
can't steer exploration toward goals in budget — exactly the cold-start the
research says the pretrained prior fixes. **Conclusion: stop tuning exploration
mechanics; do lever #1 (offline-pretrained ChangeNet) — that was the winners'
actual edge (StochasticGoose).** Keep macros as an *optional* candidate (offer
single-step AND macro moves) rather than replacing single-step, to avoid the
ls20 overshoot.

## Levers (ordered by RHAE impact)

1. **Offline-pretrain ChangeNet on the public games** — generate `(frame,action)
   → changed?` transitions offline (v19-BFS or random rollouts as the oracle on
   public sources, which we hold at train time), pretrain the CNN, ship the
   weights. A warm change/novelty prior transfers to unseen games → far fewer
   wasted exploration actions → higher RHAE. THE priority (fixes iter0's cold
   start).
2. **Progress/win-aware targets** — ChangeNet currently predicts *frame change*,
   not *progress*. Fold the `levels_completed` jump into the target (big positive
   on the action that completes a level; favour persistent/structural change over
   flicker) so exploration heads toward goals, not cosmetics.
3. **Cross-level transfer** — replay/adapt the previous level's winning sub-path
   first (frame-only) so multi-level games don't re-explore shared mechanics.
4. **Death-avoidance + adaptive click density** — down-weight GAME_OVER
   transitions (each death = wasted reset); tune click candidates (v18's
   centroid-vs-grid finding).

## Hard rules

- Black-box only: frame + levels_completed + available_actions + state. No engine
  `__dict__`, no game-source instantiation, no snapshot/restore. (White-box BFS
  on public games is allowed ONLY offline, to pretrain the CNN — never at test.)
- RHAE first: never knowingly repeat a transition; every action must earn its cost.
- One game-agnostic agent; the pretrained prior must transfer to HELD-OUT.
