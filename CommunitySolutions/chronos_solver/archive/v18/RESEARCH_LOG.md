# v18 research log

## 2026-06-15 — directed exploration for ARC-AGI-3 (iter3 input)

- **ARC-AGI-3 is an Interactive Reasoning Benchmark**: agents must explore to
  discover what each action does (action semantics vary per game), build a
  generalizable world model, set goals without instructions, then plan. Frontier
  AI < 1% as of March 2026; humans 100%. So undirected search is expected to fail.
  (ARC-AGI-3 technical report, arXiv 2603.24621; arcprize.org/arc-agi/3)
- **Go-Explore** (archive promising states → RETURN to them → EXPLORE from them):
  decouples remembering from exploring; the key to sparse-reward hard-exploration.
  Our rollout search is already this shape (frontier = archive, reset+replay =
  return, rollout = explore) — but our EXPLORE step is uniform-random, which is
  the weakness. (referenced in Graph-Based Exploration for ARC-AGI-3)
- **Graph-Based Exploration for ARC-AGI-3**, arXiv 2512.24156 — builds a state
  graph and explores it systematically rather than randomly.

**Decision for iter3:** make the rollout EXPLORE step novelty/coverage-guided —
prefer actions not yet tried from the current observed frame (a global
(state,action) archive), instead of uniform-random — and stop the frontier from
starving on GAME_OVER. This is the cheapest black-box-legal lever pointed at the
held-out number.

Sources:
- https://arxiv.org/abs/2603.24621
- https://arxiv.org/pdf/2512.24156
- https://arcprize.org/arc-agi/3

## 2026-06-15 — click-target spatial coverage (iter5 input)

Engineering fix rather than new technique — web search skipped.

**Diagnosis via brute-force scan:** ft09's clickable positions are at
{36,44,52}×{36,44,52} on the 64×64 frame, but its 64 connected-component centroids
all land OFF those positions. A grid with step=H//5=12 ([6,18,30,42,54]) also misses
them; step=H//8=8 ([4,12,20,28,36,44,52,60]) hits all three target values.

**Key finding:** vc33 has 256 productive click positions (every pixel changes the
frame), so centroid-focus (~10 candidates from iter4) was the right branching factor;
64-pt grid is too diffuse. Next lever: adaptive branching (fewer candidates when
frame is click-rich, more when centroid-based candidates produce no change).
