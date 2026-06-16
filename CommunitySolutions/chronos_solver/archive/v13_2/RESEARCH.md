# v13_2 — RESEARCH: multi-pass search (anytime tighten + enforced hill climbing)

> Two ideas from the v13_1 600s benchmark post-mortem. Both are "passes":
> a fast pass that finds/commits something, and a disciplined pass that
> exploits it. Both shrink the searched space; neither chews it faster.

## 1. Anytime incumbent tightening (ARA*-style)

The canonical version is ARA* (Likhachev, Gordon & Thrun, NIPS 2003):
run Weighted A* with a large weight to get a fast suboptimal solution,
then rerun with progressively smaller weights, REUSING earlier search
effort, until w=1 proves optimality. Each pass's solution cost is a
provable suboptimality bound on the next.

Our simplification (no reusable OPEN list across heterogeneous rungs —
they hash/prune differently): **the first verified solution from ANY rung
becomes an incumbent upper bound L; if budget remains, run exact masked
BFS with `max_depth = L-1`** — it either finds something strictly shorter
or exhausts the bounded space (proving L optimal-in-model). The depth
bound turns an unbounded search into branch-and-bound with a ceiling:
nothing deeper than the incumbent is ever expanded.

Evidence this pays: v13's 600s ls20 L4 banked greedy's 78 actions and
stopped; the 44-action solution existed and was provably inside the
depth-78 bounded space (132k states — reachable in-budget). v13_1
recovered 44 only because its bfs-rest rung happened to get there;
the incumbent bound makes that recovery *systematic* and cheaper (the
bound prunes every branch that can't beat 78).

Costs/risks: on levels where the first solution is already optimal the
tighten pass burns leftover budget proving it (capped); scorecard-wise
shorter solutions are pure profit.

## 2. Enforced hill climbing (FF-style plateau chaining)

EHC is the search core of the FF planner (Hoffmann & Nebel, JAIR 2001):
from the current state, run a complete breadth-first search until ANY
strictly-better-heuristic state is found (possibly several steps away —
"basin flooding"), commit to it, discard the rest of the frontier, and
restart from there. Depth-d problems become chains of shallow plateau
searches: roughly k searches of b^(d/k) instead of one b^d. FF falls
back to complete best-first search when EHC dead-ends — EHC is
incomplete (committing can walk into traps).

Black-box adaptation (no heuristic value to improve): "strictly better" =
**a progress event the search has never achieved before** — a masked
color-histogram signature not in the committed-sig set (same signal
v13's greedy rung priorities on, used as a commitment rule instead of a
soft ordering). Plateau search = small exact BFS (with dynamic click
targets); win checked everywhere; commit appends the plateau path to the
running solution.

Relationship to existing rungs: EHC is waypoint decomposition with
subgoals discovered BY SEARCH instead of guessed from object centroids —
it should reach levels where the "objects" aren't visually identifiable
(where _detect_player/_frame_objs fail). Like waypoint, its solution is
a composition → must be replay-verified (v13_1's `_verify_from_snap`
already does this for non-exact rungs).

Failure modes: plateau exhausts with no new signature (su15-style space
exhaustion → rung returns None, ladder continues); commitment traps
(picked the wrong key) → mitigated by the ladder fallback, exactly FF's
EHC→best-first fallback structure.

## v13_2 ladder

`bfs sprint → iw1 → iw2 → EHC (new) → waypoint → astar → bfs-rest
(frontier-resumed, **depth-bounded if an incumbent exists**) → greedy →
rescues`, plus: any non-bfs win longer than ~12 actions with budget left
triggers the **tighten pass** before returning.

## Sources

- ARA*: Likhachev, Gordon, Thrun, "ARA*: Anytime A* with Provable Bounds
  on Sub-Optimality", NIPS 2003 —
  https://www.semanticscholar.org/paper/9dfd9554e948b95cc92a64f4d16c3369cdde82de
  (formal analysis: http://www.cs.cmu.edu/~ggordon/mlikhach-ggordon-thrun.ara-tr.pdf)
- Anytime heuristic search survey (AwA* vs ARA*): Hansen & Zhou, JAIR —
  https://arxiv.org/pdf/1110.2737
- EHC / FF: Hoffmann & Nebel, "The FF Planning System", JAIR 2001;
  Metric-FF description of EHC mechanics — https://arxiv.org/pdf/1106.5271
- EHC plateau analysis ("basin flooding", dead-end fallback): Wu & Givan,
  "Stochastic Enforced Hill-Climbing", JAIR —
  https://engineering.purdue.edu/~givan/papers/seh_JAIR_final.pdf
