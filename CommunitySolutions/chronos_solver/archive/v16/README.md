# v16 — PLAN ONLY (no code yet; gated on v15 success)

> Status: planning doc. Update this once v15's Mac gate passes
> (diag [C] fresh-episode acc >= 0.90 on ar25, above copy baseline).
> Nothing here gets built until then.

---

## Step 1 (immediately after v15 passes): train on ar25, test on ar25

This is the "can it master ONE game end-to-end" checkpoint before any
generalization work. All commands run in `../v15/` — no new code.

```bash
cd ../v15

# 1. ar25-only dataset, dense (expert replays included via v13 cache)
python gen_data.py --games ar25 --episodes-per-game 500 \
    --max-steps 150 --out /tmp/ar25_shards

# 2. train wm on ar25 only (reuse v14 tokenizer as in v15 NOTES)
cp ../v14/plm_weights.pt .
python train_wm.py --phase wm --shards /tmp/ar25_shards \
    --epochs 10 --steps-per-epoch 500 --bsz 64 --holdout ""

# 3. test on ar25: fresh-episode dynamics accuracy
python diag_mismatch.py --game ar25 --shards /tmp/ar25_shards

# 4. test on ar25: actual play — levels completed is the metric now
V15_REQUIRE_WEIGHTS=1 python play_game.py --game ar25 --fast
```

Gates for this step:

| Metric | v15 pilot | ar25-only target |
|---|---|---|
| diag [C] fresh-episode acc | >= 0.90 | >= 0.97 (single game, should saturate) |
| goose err during play | falls below 0.12 | falls below 0.12, stays there |
| levels completed (of 8) | n/a (pilot) | >= 1, ideally more |

If a single-game model can't complete ar25 levels with the planner,
the bottleneck is planner/reward-head, NOT the world model — fix that
before any v16 generalization work, because generalization multiplies
whatever the planner can do, it doesn't replace it.

---

## Step 2: v16 = generalization, fueled by arc-interactive

### Will https://github.com/theredbluepill/arc-interactive help? YES.

Why it matters for us specifically:

1. **It's the same engine contract.** Games are `ARCBaseGame` +
   `metadata.json` under `environment_files/<stem>/<version>/`. Our
   `gen_data.py:load_game()` already loads exactly this layout from
   `arc-prize-2026-arc-agi-3/environment_files/` — pointing it at a
   second env dir is a ~5-line change. No new harness needed.

2. **It attacks our actual failure mode.** v14's HELDOUT_tok_acc was
   0.252 — the model has only ever seen 25 official games, ~20 in
   train. 249 extra community games (Sokoban, flood fill, slide
   puzzles, memory match, mirror/symmetry, rule-switching...) is a
   10x increase in dynamics diversity. The held-out-games gap is a
   data-diversity problem as much as an architecture problem; this is
   the cheapest data-diversity lever available.

3. **It gives us a real held-out test bed.** Today our "zero-shot"
   eval is 5 official games we can't afford to also train on. With
   arc-interactive we can train on 200+ community games + 20 official,
   and hold out ALL remaining official games as a clean,
   competition-relevant test set.

4. **Curriculum exists.** Tutorial stems ez01–ez04 are deliberately
   trivial — useful as smoke tests and as the easy end of a difficulty
   curriculum for the world model AND for future test-time-training.

5. **Solvability is checkable.** `devtools/verify_level_solvability.py`
   means we can filter out broken/unwinnable community games before
   they pollute shards (community quality varies — this is the risk,
   and the repo ships the mitigation).

### What it does NOT give us

- No guarantee community dynamics match official ARC-AGI-3 mechanics
  distribution — treat it as augmentation, not ground truth. Keep the
  official-games holdout as the only score that counts.
- Expert replays: our 30% expert-replay policy mix relies on v13's
  cached solutions, which exist only for official games. Community
  games get random/curiosity policy only at first → fewer WIN
  transitions. Mitigation: many community games are BFS-solvable with
  the v13 solver — run it once per game to build a solution cache.

### v16 build order (each gated on the previous)

1. **Vendor the environments.** `git clone arc-interactive` next to
   `arc-prize-2026-arc-agi-3/`; add `--env-dirs` (plural) flag to
   `gen_data.py`. Run solvability filter; keep the pass list in a
   checked-in `community_games.txt`.
2. **Augmentation first (already queued in v15 NOTES #1):** color
   perms + D4 with action remap. Cheap, attacks the same gap, do it
   before scaling data so the comparison is clean.
3. **Scale data:** regenerate shards over official-train +
   community-pass-list. Retrain wm (tokenizer probably needs a
   retrain too — 249 games will surface new patch statistics; expect
   token-cache invalidation, that's fine).
4. **Measure the one number that matters:** HELDOUT_tok_acc on the 5
   held-out official games. v14: 0.252. v15 target: ~0.9 via copy
   path. v16 target: meaningfully above the copy baseline = real
   zero-shot dynamics transfer.
5. **Test-time training (v15 NOTES #3):** finetune on the episode's
   own transitions when goose err stays high. The diverse pretrain
   from step 3 is what makes TTT converge in few steps on an unseen
   game — these two compound.
6. **Only then:** Kaggle notebook / competition mode dry-run
   (`run_game.py --competition` in arc-interactive matches the real
   toolkit, so use it as the pre-submit rehearsal).

### Risks / open questions

- **Engine version drift:** arc-interactive pins its own `arcengine`
  via uv; ours comes from `ARC-AGI-3-Agents`. Verify same
  `ARCBaseGame` API before vendoring (likely fine, both target the
  official toolkit).
- **License:** MIT — clean for training data.
- **Compute:** 10x games ≈ 10x shard gen time but shard gen is
  CPU-parallel; wm training cost grows with steps, not dataset size.
  Budget one RTX run (~2.5h recipe from v15 NOTES) per iteration.
- **Quality skew:** cap episodes-per-game for community games (e.g.
  100 vs 400 official) so 249 mediocre games don't drown 20 real ones.

---

## Summary

v15 proves the architecture can learn dynamics instead of memorizing.
Step 1 proves the full stack (wm + reward + planner) can win one game.
v16 = augmentation + arc-interactive data scale + TTT to make it
transfer to games it has never seen — which is the entire competition.
