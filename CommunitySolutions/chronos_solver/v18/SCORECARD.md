# v18 autonomous loop — scorecard

**DONE: generalised (held-out threshold reached).** iter4 solves **3/5 held-out
games (cn04, sk48, tu93; 4 levels)** — confirmed by the official harness
(episodes=2). The SAME game-agnostic search agent generalises to unseen games with
NO stored answers, NO per-game code, NO engine internals.

GOAL: a single frame-only agent that GENERALISES — solves levels on HELD-OUT games
(cn04, ka59, sk48, tu93, wa30) it never trained on, by genuine search (NO stored
answers). **DONE when ≥1 level is solved on ≥3 of the 5 held-out games in one eval.**

Status: **DONE (3/5)** — loop continues toward 4/5, 5/5, deeper levels, and a
learned forward model (cheaper imagination search). Honest caveats: 2/5 held-out
(ka59, wa30) still unsolved; search is action-expensive (~50k search actions/game,
though discovered solution paths are short: 65/93/82 actions).

Each autonomous run appends one row. "held-out" = games solved / total levels on
the 5 unseen games; "train-sample" = same on a fixed probe set
(ls20,sp80,lp85,cd82,ar25,vc33). A solve counts only if the engine's
`levels_completed` registered it — never fabricated.

| date | technique tried | held-out (games/levels) | train-sample (games/levels) | verdict |
|---|---|---|---|---|
| 2026-06-15 | iter2 baseline: uniform-random rollout search | 0 / 0 | 3 / 3 (ls20,sp80,lp85) | genuine solver works on easy games; held-out=0, needs directed search |
| 2026-06-15 | iter3: Go-Explore novelty-guided rollouts + GAME_OVER reset-continue | **2 / 3 (cn04, tu93×2)** | 5 / 5 (+cd82,+ar25) | **generalisation demonstrated — solves 2 UNSEEN games, no stored answers.** vc33/ft09/ka59/sk48/wa30 still 0 (starve / need click/other actions) |
| 2026-06-15 | iter4: component click-targets + ACTION5/7 + simple-first priority | **3 / 4 (cn04, sk48, tu93×2)** [official, ep=2] | +vc33 (pure-click, 2 lvls) | **DONE bar reached — 3/5 unseen games generalise.** Fixed an iter4a cn04 regression (click over-branching). Open: ka59, wa30 (near-miss), ft09 (click-target bug) |
