# v17 vs v13 — cross-game benchmark (1 sweep) + learnings

**Question:** does v17's *game-agnostic* search (forward-rollout MCTS + progress
shaping + macro-actions; NO ls20-specific ForgeNet/TRM) crack any level v13 never
solved, across all 25 official games?

**Setup:** for each game, chain to v13's frontier level (= # levels v13 cached)
using v13's solutions, then run v17 search for ~7 s/game and check whether the
next level completes. The ls20-trained models were deliberately excluded — they
don't transfer (trained on one game's mechanics).

## Result — v17 did NOT beat v13 on level count

**v13 total = 38 levels. v17 total = 38 levels. Extra cracked = 0.**

| game | v13 | v17 | v17 progress on next level |
|---|---|---|---|
| tu93 | 9 | 9 | (all levels already solved) |
| vc33 | 4 | 4 | 0 |
| ar25 | 2 | 2 | 0 |
| **cd82** | 2 | 2 | **7** (deep partial — many key/lock events) |
| dc22 | 2 | 2 | 0 |
| ft09 | 2 | 2 | 0 |
| lp85 | 2 | 2 | 0 |
| m0r0 | 2 | 2 | 0 |
| s5i5 | 2 | 2 | 0 |
| sc25 | 2 | 2 | 2 |
| sp80 | 2 | 2 | 2 |
| bp35 | 1 | 1 | 1 |
| cn04 | 1 | 1 | 1 |
| ka59 | 1 | 1 | 1 |
| r11l | 1 | 1 | 2 |
| re86 | 1 | 1 | 0 |
| sk48 | 1 | 1 | 1 |
| tr87 | 1 | 1 | 2 |
| g50t | 0 | 0 | 1 |
| sb26 | 0 | 0 | 0 |
| su15 | 0 | 0 | 2 |
| wa30 | 0 | 0 | 0 |
| lf52 | 0 | 0 | unpicklable (lambda) — search errored |
| tn36 | 0 | 0 | unpicklable (lambda) — search errored |

## Learnings

1. **Game-agnostic search alone does not beat v13 at small budgets.** v13's level
   counts came from hours of per-game BFS; 7 s/game is ~1000× less compute. At
   equal-ish budget v17 matches but doesn't exceed it. The level-count headline is
   honest: **no extra levels yet.**

2. **But v17's progress signal surfaces partial cracks that v13's BFS can't see.**
   6 games reach progress ≥ 2 on their *unsolved* frontier level (cd82=7, plus
   sc25, sp80, r11l, tr87, su15). v13's plain BFS would just breadth-die there with
   no signal at all. So v17 is making *measurable* headway into levels v13 never
   touched — it's "partial credit," not a solve.

3. **cd82 is the standout and a genuine lead.** Its frontier level reaches
   progress **7** — and with a 34 s / 256k-state run it *stayed* at 7. So cd82 is
   not budget-starved at the entry; its level-2 needs many more sequential events
   (likely a long combination puzzle). It's the most promising single target for a
   focused attack, but it's hard, not cheap.

4. **Engine render cost is game-dependent — and it dominates throughput.** cd82
   runs at ~7,500 perform/s (112 sims/s); ls20 at ~2,000 perform/s (16 sims/s).
   Same search code, 3.7× difference, purely from frame-render cost. This re-confirms
   the v17 finding: the bottleneck is the engine's CPU render, not the algorithm —
   and it's a *per-game* tax.

5. **Two games are unpicklable (lf52, tn36).** Their engines hold lambdas, so the
   snapshot-based MCTS can't restore them. v13 noted this and fell back to
   sequential (no-snapshot) expansion. v17's MCTS needs the same forward-replay-only
   mode to cover them.

6. **The ls20 models are a dead weight off-game, by design.** ForgeNet/TRM trained
   on ls20 encode ls20's key/lock mechanics; they'd mislead search on, say, a
   flood-fill game. This is the precise gap a **cross-game Puzzle LM** fills.

## So: is the Puzzle LM worth building? Yes — and this benchmark says *why*.

The gate question was "does v17 already solve more levels?" — **no.** That's the
signal that game-agnostic search has hit its ceiling and the missing ingredient is
**learned, cross-game priors** (a world model + policy/value trained on transitions
from *all 25 games*, à la v15). Two concrete pieces of evidence from this sweep:

- the progress-≥2 games (cd82, sc25, sp80, r11l, tr87, su15) are exactly where a
  learned "what counts as progress here" prior would convert partial credit into
  solves;
- per-game render cost makes a *neural* forward model (GPU-batchable, no render)
  the only way to lift the throughput ceiling uniformly.

**Next step (the Puzzle LM, v15-style):** train one tokenizer + world model +
policy/value on pooled transitions from all 25 games, so the prior is genuinely
cross-game; then use it inside the value-MCTS (which already works) as the
apprentice. The benchmark above is the before-picture to beat.
