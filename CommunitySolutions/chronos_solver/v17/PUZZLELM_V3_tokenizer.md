# Puzzle-LM — iteration 4 (tokenizer) — a controlled experiment with a negative, informative result

Iters 2–3 concluded "the 76-d object summary is the bottleneck → build a
tokenizer." Iteration 4 built it (`tokenizer.py`, a v15-style k-means VQ codebook
over 8×8 patches → an 8×8 token grid → spatial region features) and then **tested
that hypothesis properly with a controlled ablation.** The hypothesis was wrong.
Here's the honest chain of evidence.

## The tokenizer works as a representation

`tokenizer.py`: 64 patches → 16-d colour histograms → K=24 codebook (k-means) →
token grid that preserves *where* each patch-type sits (the geometry the summary
drops). Pool regenerated with combined token+object features:
**10,589 transitions, 460-d** (`models/pool_tok.npz`).

## Controlled ablation — SAME rows, three representations

Trained the cross-game policy/value on identical transitions, varying only the
input representation:

| representation | policy accuracy (5-way, chance 0.20) |
|---|---|
| object-only (76-d) | **0.301** |
| token-only (384-d spatial) | 0.296 |
| combined (460-d) | 0.291 |
| object + game-id one-hot | 0.293 |

**The richer spatial tokenizer did NOT help (0.296 vs 0.301). Game-id
conditioning did NOT help (0.293).** Both hypotheses — "representation is the
bottleneck" and "cross-game interference is the bottleneck" — are falsified.

## What the data actually says the bottleneck is

The per-game breakdown is the tell:

| game | policy acc |
|---|---|
| ls20 | 0.56 |
| cd82 | 0.27 |

The cross-game *average* (~0.30) hides huge variance. Where the policy has a
clean signal (ls20) it learns fine at 0.56 — with the *same* 76-d features. Where
it doesn't (cd82) no representation rescues it. Combined with the fact that
neither tokens nor game-id move the average, the real bottleneck is the
**training signal / label quality**:

> The pool's actions are mostly **random exploration**, and the policy label is
> "the action observed when a progress event happened." That's a noisy,
> lagged, weak target — progress events follow *sequences*, not single actions,
> and a random action that coincided with progress isn't a good demonstration.
> No representation or conditioning fixes a noisy label.

## Redirected conclusion (this is the real iteration-5 lever)

The apprentice must learn from **search-verified successful trajectories**, not
random-walk coincidences. That is exactly the ExIt loop, applied across games:

1. Run the MCTS on each game, harvest the trajectories that *actually* reached
   progress/wins (verified on the real engine).
2. Train the policy/value on **those** state→action pairs (clean demonstrations),
   not the random pool.
3. The world model can keep using the random pool (dynamics are
   action-agnostic), but the *policy/value* — the part that guides search —
   needs expert data.

So: **stop adding model capacity (tokenizer) and start improving the data
distribution (ExIt-harvested trajectories).** The tokenizer wasn't wasted — it
was the experiment that ruled out the representation hypothesis and pointed at
the data, which is a more valuable thing to know than a guess that "more pixels
will help."

## Honest scorecard across the Puzzle-LM iterations

| iter | hypothesis tested | result |
|---|---|---|
| 1 | cross-game world model learnable? | yes — 37% better than copy |
| 2 | imagination rollouts speed search? | no — tree descent dominates; weak model |
| 3 | more data strengthens the model? | no — flat; "representation" suspected |
| 4 | richer tokenizer / game-id fixes policy? | **no — falsified; it's the training signal** |
| 5 (next) | ExIt-harvested expert data fixes policy? | to test |

Files: `tokenizer.py`, `gen_pooled.py --tokenize`, `models/tokenizer.npz`,
`models/pool_tok.npz`, `models/puzzle_trm_tok.npz`.
