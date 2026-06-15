# v19 — world-model representation experiment (lift the chg-acc plateau)

## The problem
Held-out next-frame **chg-acc is stuck ~0.17** and does NOT climb as the corpus
grows (22→30 games) — and it *overfits* (peaks ~ep3, degrades by ep15). That is
the algorithm-limited signal: the WM's 21-channel **pixel** features memorise each
game's specific colours/orientation instead of learning colour/orientation-
invariant *mechanics*, so they don't transfer to unseen games.

## Why this hypothesis (research + repo)
- ARC colours and orientations are **arbitrary labels** — the same mechanic appears
  in any colour/rotation. A pixel-level model overfits to them.
- "ARC-AGI Without Pretraining" (arXiv 2512.06104): make the net equivariant to
  **colour permutation + D4 (rotations/flips)** → transformed in ⇒ transformed out.
- Object-centric world models (Dyn-O; SSWM, arXiv 2410.08822): factorise state into
  permutation-invariant **object slots** + **relational** dynamics → better rollout
  generalisation than pixel/Dreamer baselines.
- Repo already has `v17/engine.py:object_features` and v17 documents the
  "colour perms + D4 + action remap" recipe as the unfinished lever.

## Experiments (ordered by ROI; cheap → structural)

### Exp A — augmentation (cheap, proven, DO FIRST)  [prototype: wm_augment.py]
Augment the harvested transitions `(frame, action, next_frame, reward)`:
- **Colour permutation** (primary): randomly permute the non-background colours
  1..15 consistently across frame & next_frame; action label unchanged (a colour
  relabel doesn't change what the action does). Forces **colour-invariant**
  dynamics — the single highest-confidence lever for ARC.
- **D4** (secondary): apply the 8 rotations/flips to frame & next_frame; **remap
  click x,y** under the transform (simple actions keep their label → the conv
  features become orientation-robust). Action *semantics* are game-specific and
  unknown, so we do NOT remap simple-action labels — we rely on the conv trunk
  learning orientation-equivariant effects.
Train an identical WM on augmented vs raw data, compare held-out chg-acc. If aug
lifts it, the plateau was overfitting-to-surface-form (cheap win). If not, the
bottleneck is architectural → Exp B/C.

### Exp B — object-centric input features  [next]
Add `v17/object_features` as auxiliary channels / a parallel slot encoder so the
WM reasons over *objects* (count, position, bbox per colour) not raw pixels.
Permutation-invariant over objects ⇒ transfers the *interaction* pattern.

### Exp C — relational (slot + GNN) dynamics  [structural, later]
Slot encoder + graph dynamics (Dyn-O/SSWM). Biggest generalisation potential,
biggest build. Only if A/B show representation (not data) is the ceiling.

## Success metric
Held-out chg-acc (WM_LOG / the A/B harness) **climbing above ~0.20 and not
degrading with epochs**. If Exp A moves it, fold augmentation into the loop's
training (the parallel session owns `train_wm_v19.py`; this stays a standalone
prototype until proven).

## Guardrail (the honest caveat, restated)
This is a *representation/data* fix, not a compute one. If augmentation +
object features still don't lift chg-acc with more data, the limiter is the task
+ architecture, and no GPU helps the score.

<!-- ExpA 06-15 11:25: baseline=0.149 aug(ncolor=3,d4=False)=0.130 lift=-0.020 -->
