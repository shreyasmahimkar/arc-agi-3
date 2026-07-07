# Kaggle Writeup — copy-paste pack (ARC Prize 2026 · Paper Track)

Fill the "Create Writeup" form field-by-field. Each section below = one form field.
Char/word limits noted. Everything here is ready to paste.

---

## ① Title  *(max 80 chars — pick one)*

**Recommended:**
```
Chronos Solver: Why Live Search Beats Black-Box Neural Agents on ARC-AGI-3
```
Alternatives:
```
BFS-First: Genuine Live Search for ARC-AGI-3 (a 19-iteration account)
```
```
Less Search, More Honesty: BFS-First Generalisation on ARC-AGI-3
```

---

## ② Writeup URL
Auto-generated from the title — leave as Kaggle sets it.

---

## ③ Subtitle  *(max 140 chars — one sentence)*
```
An honest 19-iteration account: live white-box search scores 0.22 while a black-box neural agent scores 0.01 on the same ARC-AGI-3 games.
```

---

## ④ Card and Thumbnail Image  *(560 × 280)*
Upload: **`paper_writing/figures/thumbnail_560x280.png`** (rendered at exactly 560×280).

---

## ⑤ Submission Track
**Main Track** (single track — auto-selected).

---

## ⑥ Media gallery  *(images that tell the story — upload in this order)*

| # | File | Suggested caption |
|---|------|-------------------|
| Cover | `figures/fig0_cover.png` | Chronos Solver — 19 iterations; live white-box BFS (0.22) vs black-box neural (0.01). |
| 1 | `figures/fig1_scores.png` | Genuine live search scores 22× the black-box neural agent on the same games. |
| 2 | `figures/fig2_timeline.png` | Three eras — LLM orchestration → symbolic search → model-based RL; each fixes the last era's bottleneck. |
| 3 | `figures/fig3_coverage.png` | What v12 solved from scratch: 31 levels across 13 games (read from the real BFS caches). |
| 4 | `figures/fig4_chaining.png` | The bug that unlocked 0.22 — chaining real level baselines fixed correctness and halved solutions. |
| 5 | `figures/fig5_wall.png` | The wall: deep levels die of breadth, not impossibility (ls20 L5). |
| 6 | `figures/fig6_wm_fix.png` | A textbook generalisation fix: make "copy" the default → fresh-episode 36.6% → >90%. |
| 7 | `figures/fig7_honest_negative.png` | Honesty as method: a disproved colour-augmentation hypothesis, reported as a negative. |
| 8 | `figures/fig8_architecture.png` | The v19 system: BFS-first routing + black-box fallback + the ExIt flywheel. |

---

## ⑦ Project Description  *(≤ 1500 words — body is 1,070 words)*
Paste the text **between the two `===` rulers** in **`paper_writing/PAPER.md`**.
The bibliography sits below the lower ruler — paste it too (it does not count toward
the limit). **Before submitting, replace `<fill in>` with your leaderboard submission ID.**

---

## ⑧ Attachments

**Project links** (click "Add a link" for each):
- **Public notebook — the scored 0.22 solution (primary):**
  `https://www.kaggle.com/code/shreyas4/claude-code-v12-baseline`
  *(This is the code that produced the best leaderboard submission — the v12 live
  white-box BFS baseline. It is the required "Attached Public Notebook".)*
- Open-source code (full v1→v19 history): `https://github.com/shreyasmahimkar/arc-agi-3`
- Companion figures notebook (supplementary): *published URL of `chronos_paper_charts.ipynb`*
  — publish it as below if you want the charts as a standalone notebook too.

**Files** (optional, ≤100 MB): you may also upload `chronos_paper_charts.ipynb` and the
`figures/` directly, but the scored public notebook above is the primary code deliverable.

**Optional — Public Project Link (PDF of the paper):** upload
**`paper_writing/Chronos_Solver_ARC_Prize_2026.pdf`** (8 pages, ~1.1 MB — title page,
abstract, all 8 figures with captions, bibliography). The competition lets you attach a
PDF version of the paper via the Public Project Link / Files; this is that PDF.

---

## How to publish the *companion* figures notebook (optional)
1. On Kaggle → **Create → Notebook → File → Import** and upload
   `paper_writing/chronos_paper_charts.ipynb`.
2. Run all (it is self-contained — uses documented fallback data if the repo caches
   aren't attached), then **Save Version → make Public**.
3. Add that URL as the supplementary link above.

> The **scored** notebook is already public:
> `https://www.kaggle.com/code/shreyas4/claude-code-v12-baseline`

> Note from the form's sample list ("Bibliography? 1500-word limit?"): the 1500-word
> limit applies to the **Project Description body** — handled (1,070 words). The
> **bibliography is separate** and included at the bottom of `PAPER.md`.

---

## Submission checklist (mirrors the form's 6 items)
- [x] **Title** — §①
- [x] **Subtitle** — §③
- [x] **Card / Thumbnail image** — §④ (`thumbnail_560x280.png`)
- [x] **Submission Track** — Main Track
- [x] **Project Description** — `PAPER.md` body (paste; add submission ID)
- [x] **Project Links** — GitHub + public notebook URL
- [ ] **Final: click Submit** (top-right) after Save Draft
