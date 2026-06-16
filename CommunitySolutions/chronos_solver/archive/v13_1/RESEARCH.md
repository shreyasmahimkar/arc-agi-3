# v17 — RESEARCH ONLY (no code): do TSP / LP / NP-hard techniques help the BFS?

> Status: research note. Question: can classical combinatorial-optimization
> machinery (branch & bound, LP relaxation, TSP solvers) speed up the v13
> level search? Short answer: **the machinery mostly doesn't transfer
> directly, but four of its core ideas do** — bounds, relaxation,
> decomposition, dominance — and one algorithm family from the planning
> literature (width-based search, IW) was designed for *exactly* our
> setting and is the single best-fit import.

---

## 0. The constraint that decides everything: we are black-box

TSP and LP both assume a **white-box model**: you have the full distance
matrix / constraint matrix up front, so you can compute bounds *without
exploring*. v13 has a **black-box simulator** — the only way to learn what
an action does is to run it (0-4ms/sim). This kills any technique whose
power comes from analyzing the model symbolically, and keeps any technique
whose power comes from *organizing the exploration*.

Also: complexity class. These puzzle games are Sokoban-shaped —
reachability in compactly-described state spaces is PSPACE-hard
(Culberson '97 for Sokoban). No algorithm makes the worst case
polynomial. The only real wins are (a) heuristics/bounds, (b) abstraction,
(c) decomposition, (d) pruning. Everything below is sorted into those
buckets.

---

## 1. Concept-by-concept verdict

### Branch & bound (the exact-TSP workhorse) — idea transfers, bound is the problem
B&B = best-first search + an **admissible lower bound** that lets you prune
provably-hopeless subtrees. Our greedy heap (`-prog, depth`) is the
best-first half with no bound — it reorders but never prunes. To get the
pruning half we need a lower bound on "actions remaining to win" from a
state. Black-box, we can't derive one symbolically; but we CAN get one
geometrically: if winning requires the avatar/key to reach cell G,
`manhattan(pos, G) / max_step` is admissible. We already extract player
coords + key attrs via `hidden_fields` — the inputs exist. This turns
greedy into **A\***: same code shape, priority becomes
`depth + h(state)` instead of `-prog`. A* with even a weak admissible h
expands a fraction of BFS's nodes *and keeps the shortest-solution
guarantee* (greedy lost it).

**Verdict: YES — as A\*, not as literal B&B. Cheapest big win.**

### Linear programming — not directly; as a *bound generator*, eventually
LP itself solves continuous problems in polynomial time; our problem is
discrete and the difficulty is the integrality gap, not the LP. Where LP
earns its keep in TSP is producing **lower bounds via relaxation** (drop
integrality, solve the easy version, use its cost as a bound). The planning
literature industrialized this: delete-relaxation h+ / hFF (Hoffmann &
Nebel '01), LM-cut (Helmert & Domshlak '09), operator-counting LPs
(Pommerening et al. '14). All of them, however, require a **declarative
action model** (STRIPS-style "this action adds/deletes these facts") — we
don't have one. We'd have to *learn* the action model from rollouts first
(action X moves player +1 col; action Y toggles object Z), then relax it.
That's a v18+ project, not a v17 one.

**Verdict: NOT NOW. LP-style relaxation only pays after we have a learned
action model. The manual version of the same idea — "ignore walls, how far
is the goal?" — is just the Manhattan heuristic above, free today.**

### TSP itself — YES, as hierarchical decomposition (this is the big one)
Several games are "visit/collect k objects, then exit" — su15 branches 42
clicks, the selection games take clicks on object centroids. v13 searches
this at the **action level**: branching 42, depth ~k·(travel), space
42^depth. But the *structure* is two-level:

- **Macro level**: in what ORDER do we visit the k objects? That is
  literally TSP over k cities. k ≤ ~12 → Held-Karp exact in
  O(2^k · k²) — microseconds. k ≤ ~6 → brute-force k! is fine.
  And often order doesn't matter at all → it collapses to one macro plan.
- **Micro level**: BFS/A* *between consecutive waypoints* — branching 4-8
  movement actions, depth ~10-15 per leg. Each leg is trivial.

Space math: instead of one search of size b^(k·d), you do k! orderings
(or 2^k·k² with DP) × k searches of size b^d. For su15-type games this is
the difference between unreachable and seconds. The **dynamic-click-target
scanner already computes the waypoint set** (object centroids per frame) —
v13 built the macro vocabulary and is still searching it flat.

Caveat: legs aren't independent when the world changes after each pickup
(doors open, objects move). Fix: re-plan legs in sequence from the actual
post-leg state — exactly the pattern v13 already uses for chaining level
solutions (bug-fix #4). Order constraints (key BEFORE lock) make it
sequential-ordering-TSP — still tiny at our k; constraints only *shrink*
the ordering space.

**Verdict: YES — highest-leverage import on the list. It's the
"cut breadth, don't chew faster" pattern at the level above clicks.**

### Dominance pruning (B&B's other half) — YES, cheap upgrade to dedup
v13's visited-set prunes only *exact-equal* (masked-hash) states. B&B/TSP
solvers also prune **dominated** states: if state A has the same position
as B but strictly-no-worse resources (more keys collected, lower
countdown), B is hopeless and dies. We already fold public scalar attrs
into the hash — the same attrs define the dominance partial order
(position equal, collected-set ⊇, timer ≤). One dict from
position→best-resources-seen, prune on insert. Risk: wrong dominance
assumptions silently lose solutions (a game where *fewer* keys is better);
ship it like masked-hashing shipped — with an automatic
dominance-off retry on exhaustion.

**Verdict: YES — small code, multiplies with everything else.**

### Width-based search / IW(1) — YES, the literature's best match
Iterated Width (Lipovetzky & Geffner '12; applied to black-box Atari
sims in Lipovetzky, Ramirez & Geffner IJCAI'15 — *exactly* our regime:
pixels + black-box step function). IW(1): define boolean atoms over the
state (e.g. "pixel(x,y)=c", or object-level "key at (r,c)"); a new state
is kept **only if it makes at least one atom true for the first time in
the whole search**; otherwise pruned, even if unvisited. Number of kept
states is bounded by #atoms — *linear, not exponential*. IW(1) solves a
shocking fraction of Atari/puzzle domains because most game features
depend on few state variables ("low width"). If IW(1) exhausts without a
win, escalate to IW(2) (pairs of atoms), like the masked→unmasked retry
ladder v13 already has.

This is the principled version of v13's "progress events" histogram —
the histogram says "something changed vs my parent"; novelty says
"something changed *vs everything ever seen*", which is a far stronger
pruner. Natural atom set for us: per-cell color (frames are small), plus
the hidden scalar attrs.

**Verdict: YES — strongest single algorithmic idea here; designed for
black-box sims; composable with greedy ordering and the waypoint TSP.**

### Bidirectional search — NO (blocked)
Would turn b^d into ~2·b^(d/2) (square root of the space — bigger than
any constant-factor parallelism). Requires searching backward from the
goal, i.e. inverse dynamics or at least enumerable goal states. Our sim
is forward-only and the win state is unknown until reached.
Revisit only if a learned world model (v15/v16 track) becomes reliable
enough to invert.

### Approximation algorithms / local search (Christofides, 2-opt, annealing) — mostly NO
We need an *exact* win, not a 1.5-approximation of one. The one transfer:
**2-opt-style post-hoc solution shortening** — after a win, check whether
any state in the trace recurs later (splice out the loop) or whether
A* between trace states i and j beats the recorded segment. Only worth it
if action count affects the scorecard.

### Beam search — already half-built, formalize it
The 25k frontier cap on persist *is* a beam, applied accidentally and only
at checkpoint time. A deliberate beam (keep best W by priority every
expansion round) is the standard memory-bounded best-first compromise —
but it sacrifices completeness, so it belongs as a stage in the retry
ladder, not a default.

### IDA\* / frontier search — the answer to the RAM wall, not the speed wall
`--max-states 5M` and the disk guards exist because frontier+visited live
in RAM. IDA* stores O(depth) and re-expands instead — and re-expansion is
cheap at 0-4ms/sim. With a Manhattan h it's IDA*; with h=0 it's iterative
deepening (pure-BFS-equivalent, near-zero memory). Best fit for the deep
games (cn04 at 5700 sims/s slamming state caps).

---

## 2. What this means for v17, priority-ordered

1. **Waypoint decomposition (TSP layer)**: macro-search over object-centroid
   orderings (Held-Karp / brute force, k is tiny), micro A* per leg,
   re-planned sequentially from actual post-leg state. Targets: su15,
   selection games, any "collect-k" level.
2. **IW(1) novelty pruning** as a new strategy in the retry ladder:
   masked-BFS → greedy → **IW(1) → IW(2)** → unmasked. Atoms = per-cell
   color + hidden scalars. Replaces "histogram changed vs parent" with
   "first time ever".
3. **A\* upgrade to greedy**: priority `depth + manhattan(player/key, goal)`
   when a goal cell is identifiable; keeps optimality, prunes hard.
4. **Dominance pruning** on (position, collected-set, timers) with
   auto-retry off on exhaustion.
5. **IDA\*** variant for the RAM-capped deep games.
6. (Deferred to v18+): learned action model → LP/operator-counting bounds;
   bidirectional via world model.

Items 1-4 all *shrink the searched space*; none add cores. They compose:
TSP layer cuts depth, IW cuts breadth, dominance and A* cut both.

## 3. References

- Lipovetzky & Geffner, "Width and Serialization of Classical Planning
  Problems", ECAI 2012 (IW).
- Lipovetzky, Ramirez, Geffner, "Classical Planning with Simulators:
  Results on the Atari Video Games", IJCAI 2015 (IW on black-box pixels).
- Held & Karp, dynamic programming for TSP, 1962 (exact small-k orderings).
- Hoffmann & Nebel, "The FF Planning System", JAIR 2001 (delete relaxation).
- Helmert & Domshlak, "Landmarks, Critical Paths and Abstractions", ICAPS
  2009 (LM-cut, LP-flavored admissible bounds).
- Pommerening et al., "LP-Based Heuristics for Cost-Optimal Planning",
  ICAPS 2014 (operator counting — the real "LP meets search").
- Culberson, "Sokoban is PSPACE-complete", 1997 (why no silver bullet).
- Korf, "Depth-First Iterative-Deepening", AIJ 1985 (IDA*).
