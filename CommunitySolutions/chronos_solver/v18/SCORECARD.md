# v18 autonomous loop — scorecard

GOAL: a single frame-only agent that GENERALISES — solves levels on HELD-OUT games
(cn04, ka59, sk48, tu93, wa30) it never trained on, by genuine search (NO stored
answers). **DONE when ≥1 level is solved on ≥3 of the 5 held-out games in one eval.**

Status: **IN PROGRESS — generalisation demonstrated** (iter3 solves 2/5 held-out
games it never trained on; DONE threshold is 3/5).

Each autonomous run appends one row. "held-out" = games solved / total levels on
the 5 unseen games; "train-sample" = same on a fixed probe set
(ls20,sp80,lp85,cd82,ar25,vc33). A solve counts only if the engine's
`levels_completed` registered it — never fabricated.

| date | technique tried | held-out (games/levels) | train-sample (games/levels) | verdict |
|---|---|---|---|---|
| 2026-06-15 | iter2 baseline: uniform-random rollout search | 0 / 0 | 3 / 3 (ls20,sp80,lp85) | genuine solver works on easy games; held-out=0, needs directed search |
| 2026-06-15 | iter3: Go-Explore novelty-guided rollouts + GAME_OVER reset-continue | **2 / 3 (cn04, tu93×2)** | 5 / 5 (+cd82,+ar25) | **generalisation demonstrated — solves 2 UNSEEN games, no stored answers.** vc33/ft09/ka59/sk48/wa30 still 0 (starve / need click/other actions) |
