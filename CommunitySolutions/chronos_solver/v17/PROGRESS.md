# v17 Puzzle-LM — autonomous progress log

**Goal:** one model/system that solves ALL levels of ALL 25 official ARC-AGI-3
games with the FEWEST actions (max RHAE). Iterating hourly, research-driven,
until solved or stopped. Never fake a solve — verify every level on the real
engine.

## Current standing (update each iteration)

- **Levels solved (verified): 38 / (v13 baseline 38).** v17 has NOT yet beaten
  v13 on level count. ls20 L5 (dual-key) still unsolved; best = "progress 3".
- **Per-game frontier progress (CLEAN signal, iter 6):** cd82=5, su15=2, r11l=2,
  sp80=1, ls20 L5=3. **Reclassified as FALSE leads (pure navigation noise):
  sc25 7→0, tr87 2→0, g50t 1→0.** The noisy signal had cd82=7, sc25/sp80/r11l/
  tr87/su15=2 — iter6's nav-masking shows ~half of that "partial credit" was
  just the player coordinate changing as it wandered.
- **Running best model:** `models/puzzle_trm.npz` (cross-game policy/value) +
  `models/puzzle_wm.npz` (world model, 37% better than copy). Cross-game policy
  acc ~0.30 (weak).
- **Key infra:** real engine runs in sandbox (`unset PYTHONPATH`); MCTS +
  value-MCTS (AlphaZero PUCT) + macro-actions + progress shaping + novelty all
  work; `multi_game.py` is the 25-game benchmark; lf52/tn36 unpicklable.

## Research notes (carry forward)

- **2025 ARC Prize winners (NVARC / MindsAI / ARChitects)** all rely on
  **test-time training (TTT)** + **synthetic data generation** + recursive
  self-refinement. TTT (adapt the model on the test task's own examples) is the
  single most consistent winning technique. (arxiv 2601.10904; arcprize.org 2025
  results.)
- ARC-AGI-1/2 are static grid puzzles; their solving leans on program synthesis
  / DSL search + TTT. ARC-AGI-3 (ours) is interactive — the transferable ideas
  are TTT, synthetic-data augmentation (color perms + D4), and search.

## Iteration log

### Iter 6 — Clean progress signal: navigation-scalar masking  [2026-06-15]
**Research:** ARC-AGI-3 technical report (arxiv 2603.24621) + ARC Prize 2026 page
confirm the core difficulty is that the env gives ONLY a sparse level-completion
reward across ≤10 levels/game — so any dense progress proxy we synthesize MUST be
denoised or learning-based methods chase noise. Also noted new technique to try
later: *Graph-Based Exploration for ARC-AGI-3* (arxiv 2512.24156). This iteration
executes the planned lever (clean the signal) since it is the gate for TTT/ExIt.

**What iter5 actually was (corrected):** iter5 said cd82 "112/112 steps register
progress = timer noise" and proposed *transient-scalar masking* (mask attrs that
change under EVERY action). I implemented that first — and it found **0** keys on
every game. Reason: no single scalar changes under *every* action (moving up
changes y, moving right changes x; each survives the AND). The real pollution is
different: **player-coordinate scalars count as "progress" when the agent merely
walks.** prog() = #scalars differing from level-start; one step of movement flips
a coordinate ⇒ "progress 1" for doing nothing meaningful.

**Implemented (`engine.detect_transient_scalars`, wired into both `mcts.solve_mcts`
and `mcts.solve_mcts_az` via `clean_progress=True`, default on):** run a
*movement-only* random walk from the level start (2 seeds, 48 steps) and record
how many DISTINCT values each scalar attr takes. Keys with >3 distinct values
under pure movement are NAVIGATION (free-running coordinates) and are EXCLUDED
from the progress count; genuine state-machine attrs (keys/locks/goals) stay
constant under movement and survive. Two seeds are intersected so a one-off real
event during a walk isn't mis-masked. Also fixed the engine bootstrap to namespace
`/tmp/v17_aelib_<uid>` and clear stale broken symlinks (a prior session's dir was
permission-blocking all runs).

**Benchmark (sims=3000, tb=9s, game-agnostic solve_mcts, clean vs the recorded
noisy baseline in v17_multigame.json; results in v17_multigame_clean.json):**

| game | noisy best_prog | CLEAN best_prog | nav keys masked | verdict |
|---|---|---|---|---|
| cd82 | 7 | **5** | 2 | genuine — strongest lead, 2 were nav |
| su15 | 2 | **2** | 0 | fully genuine |
| r11l | 2 | **2** | 0 | fully genuine |
| sp80 | 2 | **1** | 1 | half genuine |
| sc25 | 2 | **0** | 2 | **FALSE lead — all navigation noise** |
| tr87 | 2 | **0** | 2 | **FALSE lead — all navigation noise** |
| g50t | 1 | **0** | 1 | **FALSE lead** |

**Result vs running best: no new levels solved (still 38 = v13).** But the value/
label signal is now trustworthy: it reclassifies 3 of 6 "partial-credit" leads as
pure navigation artifacts that TTT/ExIt would have wasted compute chasing, and
confirms cd82(5)/su15(2)/r11l(2)/sp80(1) as the genuine frontier targets. This is
the prerequisite the iter5 blocker called for — done and verified on the real
engine. Honest: it's an enabling/diagnostic win, not a score win.

**Next planned lever (iter 7):** now that progress = genuine events, run TTT + 
ExIt (harvest search-verified clean-progress trajectories; adapt the cross-game
TRM per game) ON THE GENUINE-LEAD GAMES ONLY (cd82, su15, r11l, sp80) — drop the
false leads. Pair with synthetic augmentation (color perms + D4 + action remap).
Require levels_solved > 38 or a verified new level. If TTT again gives no lift,
try the Graph-Based Exploration prior (arxiv 2512.24156) as the explorer.

### Iter 5 — Test-Time Training (TTT)  [research: 2025 ARC Prize winners]
- Built `ttt.py`: scout a game's own transitions → fine-tune a copy of the base
  cross-game TRM → search with the adapted model.
- Result: **no lift.** cd82 L2 base=7 / TTT=7; su15 L0 base=2 / TTT=2.
- **Critical finding (the real blocker):** the "progress" signal = count of
  changed engine scalar attrs, but on some games (cd82) **112/112 scouted steps
  register progress** — a timer/counter ticks every step. So "progress N" is
  PARTLY NOISE, not N real key/lock events. This confound has silently driven
  ~30 iterations and pollutes the value/reward signal AND the TTT label.
- TTT can't help while the signal it adapts on is noisy.

## Next planned lever (do this next iteration — iter 7)

DONE in iter6: clean progress signal (nav-scalar masking) — `clean_progress=True`
is now default in both MCTS solvers. The signal is trustworthy. So next:

1. **TTT + ExIt on the GENUINE-LEAD games only** (cd82=5, su15=2, r11l=2, sp80=1).
   Drop the false leads (sc25/tr87/g50t) — they were navigation noise. Harvest
   search-verified CLEAN-progress trajectories, adapt the cross-game TRM per game,
   re-search with the adapted model. The value/label is now meaningful (the iter5
   blocker is cleared).
2. **Synthetic-data augmentation** (2025-winner recipe): color permutations + D4
   transforms with action remap, to grow clean training data per game.
3. cd82 is the deepest genuine lead (5 real events) — give it a focused, longer
   budget; it likely needs a long sequential combination.
4. If TTT still gives no lift, try **Graph-Based Exploration** (arxiv 2512.24156)
   as the explorer prior.
5. Re-benchmark all 25 games; require levels_solved > 38 or a verified new level.

## Honest scorecard of hypotheses tested

| iter | technique | result |
|---|---|---|
| PLM1 | cross-game world model | works (37% > copy) |
| PLM2 | imagination rollouts | no lift (descent dominates; weak model) |
| PLM3 | 3× more data | no lift (not data quantity) |
| PLM4 | tokenizer / game-id | no lift (not representation) — it's the signal/label |
| 5 | test-time training | no lift — exposed polluted progress signal |
| 6 | clean progress (nav-scalar masking) | works as denoiser: cd82 7→5, killed 3 false leads (sc25/tr87/g50t →0); no new solve yet but signal now trustworthy |
| 7 (next) | TTT/ExIt on genuine leads only + synthetic aug | to test |
