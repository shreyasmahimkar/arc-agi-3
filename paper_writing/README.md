# paper_writing/ — ARC Prize 2026 Paper Track submission

Everything needed to submit the **Chronos Solver** writeup documenting the v12→v19
ARC-AGI-3 approach. Built to the competition's six judging criteria.

## Files

| File | What it is | Maps to form field |
|------|-----------|--------------------|
| **`KAGGLE_WRITEUP_SUBMISSION.md`** | **Start here.** Field-by-field copy-paste pack for the *Create Writeup* form. | all |
| `PAPER.md` | The Project Description (body = **1,070 words**, ≤1500) + bibliography. | Project Description |
| **`Chronos_Solver_ARC_Prize_2026.pdf`** | Polished 8-page paper (abstract + all 8 figures + bibliography). | Public Project Link / Files |
| `make_pdf.py` | Regenerates the PDF from `PAPER.md` + `figures/`. | — (reproducibility) |
| `chronos_paper_charts.ipynb` | Public notebook — full writeup framing + 8 charts inline (executed). | Public notebook link |
| `make_figures.py` | Standalone, reproducible generator for every figure. | — (reproducibility) |
| `figures/thumbnail_560x280.png` | Card / thumbnail image (exact size). | Card / Thumbnail |
| `figures/fig0_cover.png` … `fig8_*.png` | Cover + 8 story figures. | Media gallery |

## How the work hits the 6 judging criteria

- **Accuracy** — the leaderboard story is the spine: live BFS **0.22** vs black-box
  **0.01** (Fig 1); 31 levels / 13 games solved from real caches (Fig 3).
- **Universality** — §8: "when a faithful simulator/verifier exists, search first and
  let learned models prioritise it" (AlphaZero/Searchformer/DeepCubeA); honesty
  scaffolding transfers to any sparse-reward, distribution-shift task.
- **Progress** — localises the field-relevant bottleneck to *representation* via a
  disproved ablation (Fig 7), and shows the genuine-search recipe that actually scores.
- **Theory** — every section argues *why* (genuine search generalises; chained
  baselines align the search space; breadth-death; residual "copy-by-default"), not
  just *how* (Figs 4–6).
- **Completeness** — full 19-version arc in three eras (Fig 2) + the v19 system
  diagram (Fig 8) + limitations + the 0.02 deployment-regression caveat.
- **Novelty** — honesty-as-method (held-out by game, changed-pixel metric, save-gate,
  reported negatives) and the BFS-first synthesis fusing both ARC-AGI-3 preview winners.

## Reproduce the figures

```bash
source .venv312/bin/activate
python paper_writing/make_figures.py          # writes figures/*.png
# or run the notebook end-to-end (self-contained; uses fallback data off-repo):
cd paper_writing && jupyter nbconvert --to notebook --execute --inplace chronos_paper_charts.ipynb
```

## Before you submit
1. Paste fields from `KAGGLE_WRITEUP_SUBMISSION.md`.
2. In the description, replace `<fill in>` with your **leaderboard submission ID**
   (the v12 baseline run that scored 0.22).
3. Project links: the **scored public notebook** is
   `https://www.kaggle.com/code/shreyas4/claude-code-v12-baseline` (primary) + the GitHub
   repo. Optionally publish `chronos_paper_charts.ipynb` as a supplementary figures notebook.
4. Upload `thumbnail_560x280.png` as the card and the figures to the media gallery.
5. Save Draft → **Submit**.
