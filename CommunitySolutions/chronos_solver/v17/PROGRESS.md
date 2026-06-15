# v17 Puzzle-LM — autonomous progress log

**Goal:** one model/system that solves ALL levels of ALL 25 official ARC-AGI-3
games with the FEWEST actions (max RHAE). Iterating hourly, research-driven,
until solved or stopped. Never fake a solve — verify every level on the real
engine.

## Current standing (update each iteration)

- **Levels solved (verified): 38 / (v13 baseline 38).** v17 has NOT yet beaten
  v13 on level count. ls20 L5 (dual-key) still unsolved; best = "progress 3".
- **Per-game frontier progress (best so far):** cd82≈7*, ls20 L5=3, sc25/sp80/
  r11l/tr87/su15=2 (* cd82's progress is partly timer noise — see iter 5).
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

## Next planned lever (do this next iteration)

1. **Clean the progress signal — transient-scalar masking.** Mirror the
   transient-row frame mask: detect engine scalar attrs that change under EVERY
   single action from a level start (timers/counters) and EXCLUDE them from the
   progress count. Then "progress" = genuine key/lock/goal events only. Re-measure
   per-game progress with the clean signal (cd82's "7" will likely drop a lot;
   the sparse games' "2" should stay — those are real).
2. Re-run TTT + ExIt with the CLEAN signal (the value/label is now meaningful).
3. **Synthetic-data augmentation** (2025-winner recipe): color permutations + D4
   transforms with action remap, to grow clean training data per game.
4. Re-benchmark all 25 games; require levels_solved > 38 or a verified new level.

## Honest scorecard of hypotheses tested

| iter | technique | result |
|---|---|---|
| PLM1 | cross-game world model | works (37% > copy) |
| PLM2 | imagination rollouts | no lift (descent dominates; weak model) |
| PLM3 | 3× more data | no lift (not data quantity) |
| PLM4 | tokenizer / game-id | no lift (not representation) — it's the signal/label |
| 5 | test-time training | no lift — exposed timer-polluted progress signal |
| 6 (next) | clean progress signal + TTT/ExIt + augmentation | to test |
