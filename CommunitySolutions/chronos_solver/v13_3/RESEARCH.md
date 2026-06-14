# v13_3 — RESEARCH: Cracking Space-Limited Levels (Partial Observability)

> **Problem statement**: v13_2 exhausts the entire reachable state space on
> levels like ar25 L1 (~30-50k unique states) without finding a solution.
> More workers, more time, more budget: all useless. The solver has literally
> visited every state it can reach from the clean start — and the solution
> isn't in any of them.
>
> Root cause: the game is **partially observable**. Taking certain actions
> reveals hidden regions of the board — new objects, counters, triggers —
> that weren't visible in the initial frame. BFS hashes the *visible* state
> and declares exhaustion. The true reachable state space, including hidden
> regions unlocked by exploratory actions, may be 10x larger. The solution
> lives there. BFS never gets there because the "revealing" moves look neutral
> (no score change, no obvious progress) and IW prunes them.

---

## 1. Why IW makes it worse: the atom-set problem

IW(k) (Lipovetzky & Geffner, IJCAI 2012) prunes any child state whose
novelty is greater than k. Novelty is defined as: *the size of the smallest
tuple of atoms that are true in this state and were false in all previously
visited states*. The key guarantee — IW solves any problem of width k in
O(b^k) states — holds **only if the atom set captures the relevant structure
of the problem**.

v13_2's IW atoms:
- `(x, y, color)` — is cell (x,y) this color for the first time?
- public scalar attrs: score, level counter, etc.

Neither of these captures "this action revealed a previously hidden pixel."
So when BFS takes an exploratory move that peeks behind a wall or triggers
a hidden object:
- No new color atoms (the revealed region wasn't in the hash before, so its
  pixels register as "first time" — actually this *should* fire, see §3)
- No new scalars
- IW prunes it as low-novelty

**The real issue**: if a region is hidden (all zeros / background), every
cell in it is "seen" at color=0 immediately at depth 0. When the region
reveals, the cells change to non-zero colors — but those cells at those
colors ARE new atoms. So IW(1) *should* keep them... unless the revealing
move is deep enough that IW has already pruned the path leading to it.

This is IW's fundamental incompleteness: it commits to a novelty frontier
at each depth level. If the only path to a revealing action goes through
states that IW considers boring (correct color histogram, no new atoms at
shallow depth), IW cuts that path before depth reaches the reveal. The
solution — reachable in principle — becomes unreachable in practice.

**Source**: Lipovetzky & Geffner, "Width and Serialization of Classical
Planning Problems", ECAI 2012; survey in "Planning for Novelty: Width-Based
Algorithms for Common Problems in Control, Planning and Reinforcement
Learning", arXiv 2106.04866.

---

## 2. The contingent planning framework

Contingent planning (Hoffmann & Brafman, ICAPS 2005 — Contingent-FF) splits
actions into two categories:

- **Actuation actions**: move toward the goal (collect object, push block)
- **Sensing / information-gathering actions**: reveal hidden state (open a
  door, step on a pressure plate, click an obscured region)

Classical BFS treats both identically. Contingent planners explicitly model
sensing actions and reward information gain. The key idea: *you must gather
information before you can act on it*. The planner maintains an **implicit
belief state** — a compact formula describing which world-states are still
consistent with observations — and searches over belief states, not raw
states.

For ARC games, sensing actions are any action that causes `frame[x,y]` to
change from background to non-background for pixels that were not previously
tracked. The solver currently has no concept of "I revealed something" as a
goal-relevant event.

**Contingent-FF result**: on partially observable planning benchmarks,
explicit sensing-action reasoning reduces search space by orders of magnitude
vs. treating all actions uniformly. The planner knows *which actions to take
to reduce uncertainty* before attempting goal achievement.

**Source**: Hoffmann & Brafman, "Contingent Planning via Heuristic Forward
Search with Implicit Belief States", ICAPS 2005.
https://aaai.org/papers/icaps-05-008-contingent-planning-via-heuristic-forward-search-with-implicit-belief-states/

---

## 3. RolloutIW and pixel-domain novelty (Atari analogy)

Bandres, Bonet & Geffner (AAAI 2018) introduced **RolloutIW** for Atari:
IW running directly on pixel frames. Crucially, they use **B-PROST boolean
features** extracted from pixel patches — not raw pixel values. A feature
fires when a specific visual pattern appears for the first time anywhere in
the frame. This is much richer than per-cell-color atoms.

Follow-up work (arXiv 1904.07091, "Deep Policies for Width-Based Planning
in Pixel Domains") showed that VAE-learned latent features used as novelty
atoms outperform hand-coded pixel features, because the learned atoms capture
semantically meaningful patterns (object presence, object state) rather than
raw pixels.

**Lesson for ARC**: our current IW atoms are essentially per-cell color atoms
— one step above raw pixels. What we're missing is **object-level atoms**:
- "object of color C is now visible" (object appeared)
- "region R transitioned from all-background to non-background" (area revealed)
- "N previously-hidden pixels are now non-zero" (reveal magnitude)

These atoms fire precisely when information is gained, making the revealing
move novel regardless of whether any individual cell is a new color.

**Source**: Bandres, Bonet & Geffner, "Planning With Pixels in (Almost) Real
Time", AAAI 2018. https://bonetblai.github.io/reports/AAAI18-pixels.pdf
Deep Policies for Width-Based Planning: https://arxiv.org/pdf/1904.07091

---

## 4. Sketch decompositions: bounded-width subproblems

Drexler, Seipp & Geffner (ICAPS 2022, IJCAI 2025) introduced **sketches**:
collections of rules `C ↦ E` over features that decompose a hard problem
into a sequence of bounded-width subproblems. A sketch rule says "from any
state satisfying condition C, achieve the qualitative change E." Each
subproblem is solved by IW(k); the full solution is the chain.

For ARC levels where the full problem has high width (IW(1) fails), the
problem might still decompose into width-1 subproblems:
1. Reveal all hidden regions (sensing phase)
2. Collect all objects of type A (actuation phase)
3. Reach the goal (final phase)

Each subproblem is narrow; the chain is solvable. This is exactly the
structure ar25 L1 likely has — there's a sensing phase (click/step to reveal
the hidden part of the board) followed by a normal collection phase.

The EHC rung in v13_2 is a rudimentary version of this: it chains plateau
BFS segments, committing to progress signatures. The difference: EHC doesn't
distinguish sensing from actuation, so it can commit to a "progress event"
that was actually just a neutral exploration move, trapping the search.

**Source**: "Learning Sketches for Decomposing Planning Problems into
Subproblems of Bounded Width", ICAPS 2022. https://arxiv.org/abs/2203.14852
IJCAI 2025 follow-up: https://www.ijcai.org/proceedings/2025/0938.pdf

---

## 5. POMCP: belief-state MCTS (heavyweight option)

POMCP (Silver & Veness, NIPS 2010) is Monte Carlo Tree Search over belief
states for POMDPs. Instead of searching over world states, it maintains a
particle filter — a set of sampled possible world states consistent with
the action-observation history. UCB1 selects actions; observations narrow
the particle set; rollouts estimate value.

POMCP scales to large state spaces because it never represents the full
policy — only the part of the belief tree actually visited. It handles
partial observability natively: the particle filter IS the belief state.

For ARC this would mean:
- Particles = sampled completions of the hidden portions of the frame
- Observations = the actual frame returned after each action
- UCB rollouts guide toward revealing actions naturally (particles that
  don't get resolved by an action contribute high variance → UCB explores)

The cost: particle filter maintenance, rollout simulation, and the need for
a generative model (we have one: the game engine itself). This is the most
principled solution but the most expensive to implement.

**Source**: Silver & Veness, "Monte-Carlo Planning in Large POMDPs", NIPS
2010. POMCPOW extension for continuous spaces:
https://github.com/JuliaPOMDP/POMCPOW.jl

---

## 6. Proposed approaches for v13_3 (ordered by implementation cost)

### 6.1 Reveal-novelty atoms in IW [LOW COST, HIGH LEVERAGE]

Add a new IW atom class: `revealed(region_id)` — fires when a cluster of
previously-background pixels becomes non-background after an action. Detect
regions via connected-component labeling on the diff `new_frame - old_frame`
where old_frame was all-zero at those positions.

Any child state that makes at least one `revealed(r)` atom true is kept
regardless of other novelty checks. This costs a single frame diff per
expanded node and prevents IW from pruning exploratory moves.

Implementation: in `_guided_search()`, after computing the child frame,
diff against a "background mask" maintained from the initial frame. If any
pixel transitions background→non-background, add `('revealed', region_id)`
to the child's atom set before the novelty check.

### 6.2 Sensing prepass rung [MEDIUM COST, HIGH LEVERAGE]

New ladder rung `sense` (before IW): a short BFS (10-20s, capped) whose
only objective is **maximizing revealed pixels** from the initial frame. No
solution check needed; just explore. Records the top-K "revelation states"
— states where the most new pixels were uncovered.

After the sensing pass, IW/BFS restarts from each revelation state rather
than the clean initial frame. The reachable state space from a revelation
state includes all the hidden regions, so IW and BFS can now find the
solution.

This is the Contingent-FF sensing-action idea adapted to our architecture:
a dedicated phase to reduce uncertainty before solution search.

### 6.3 Sketch-aware EHC [MEDIUM COST, MEDIUM LEVERAGE]

Modify EHC to distinguish sensing commits from actuation commits. A commit
is a "sensing commit" if it increased the revealed pixel count; an
"actuation commit" if it changed score, histogram, or scalar attrs. The
EHC plateau restarts from sensing commits with a larger plateau budget
(they opened new territory), but discards actuation commits that reach dead
ends more aggressively.

This prevents EHC from trapping itself in a committed sensing path that
doesn't lead to the solution — it keeps sensing commits and backtracks
actuation commits.

### 6.4 Belief-state BFS [HIGH COST, MOST PRINCIPLED]

Maintain K particle states (K=8-16) per BFS node. Each particle is a
completion of the hidden pixels. Actions are applied to all particles;
the observation (actual frame returned) is used to filter particles. BFS
searches over particle-sets rather than single states. Expensive but sound
for any POMDP.

---

## 7. Verdict

**v13_3 should implement 6.1 + 6.2** — reveal-novelty atoms are a one-file
change to IW, and the sensing prepass is a new ladder rung that slots into
the existing architecture. Together they address the root cause: the solver
was blind to information-gathering moves. The sketch-aware EHC (6.3) is a
good follow-on if 6.1+6.2 don't fully crack ar25 L1.

6.4 (belief-state BFS) is the thesis-level solution and probably v14 or
later — it requires rethinking the state representation from scratch.

---

## Sources

- [Planning for Novelty: Width-Based Algorithms (Lipovetzky, 2021)](https://arxiv.org/abs/2106.04866)
- [Contingent Planning via Heuristic Forward Search (Hoffmann & Brafman, ICAPS 2005)](https://aaai.org/papers/icaps-05-008-contingent-planning-via-heuristic-forward-search-with-implicit-belief-states/)
- [Planning With Pixels in (Almost) Real Time — RolloutIW (Bandres et al., AAAI 2018)](https://bonetblai.github.io/reports/AAAI18-pixels.pdf)
- [Deep Policies for Width-Based Planning in Pixel Domains (2019)](https://arxiv.org/pdf/1904.07091)
- [Learning Sketches for Decomposing Planning Problems — Sketch decompositions (ICAPS 2022)](https://arxiv.org/abs/2203.14852)
- [Sketch Decompositions via Deep RL (IJCAI 2025)](https://www.ijcai.org/proceedings/2025/0938.pdf)
- [POMCPOW — Online POMDP solver with continuous spaces](https://github.com/JuliaPOMDP/POMCPOW.jl)
- [Heuristics for Partially Observable Stochastic Contingent Planning (2024)](https://arxiv.org/abs/2410.05870)
- [Probabilistic Contingent Planning with HTN (2025)](https://www.mdpi.com/1999-4893/18/4/214)
- [Is Policy Learning Overrated? Width-Based Planning and Active Learning for Atari (2021)](https://arxiv.org/pdf/2109.15310)

---

## 8. Reactive / co-moving objects (ls20 L5 observation)

> **Observed mechanic**: when the player moves in ls20 L5, other blocks in
> the level also react — color-changing blocks shift color, rotation blocks
> rotate. The player's single action has **multiple simultaneous state
> changes** across multiple objects. This is a fundamentally different
> problem class from pure navigation.

### Why this explodes the state space

In a plain movement game, state = player position. With K reactive blocks
each having M possible states, state = (player_pos, block1_state, ...,
blockK_state). For ls20 L5 with 5 waypoints and say 3-state reactive
blocks: 3^5 = 243x more unique states per player position. BFS with a
transient mask that only covers the player's rows will hash all of these as
distinct states. The ladder hits 235k explored on M1 and still has 10k
frontier — because it's not exploring player positions, it's exploring
(player_pos × reactive_block_config) tuples, most of which are dead ends.

Additionally, the **solution requires choreography**: the player must reach
the goal configuration while simultaneously driving the reactive blocks into
the correct states. This is a constraint-satisfaction problem embedded inside
a search problem.

### Analogy 1: PushWorld (Google DeepMind, 2023)

PushWorld (Krasinski et al., arXiv 2301.10289) is a benchmark of exactly
this structure: the agent pushes blue objects which transitively push red
objects, all of which must reach goal positions simultaneously. Classical
planners and RL both perform below human level. Key findings:

- The coupled state transitions (push A → A pushes B → B pushes C) make
  the problem **non-Markovian from the player's perspective**: you can't
  plan the player's path without also planning the block trajectories.
- The best performing approach uses **factored state representations** —
  planning over (player, object1, object2, ...) as separate variables,
  exploiting independence where it exists, rather than treating the joint
  state as a monolithic hash.
- Landmark-based heuristics (identify necessary intermediate configurations
  of sub-objects) dramatically reduce search depth.

**Lesson for ls20 L5**: The reactive blocks are essentially the "red objects"
in PushWorld. The solver needs to reason about block states as first-class
planning variables, not as incidental pixel noise.

**Source**: https://arxiv.org/abs/2301.10289 |
GitHub: https://github.com/google-deepmind/pushworld

### Analogy 2: Keke AI / Baba is You (2022)

Keke AI (arXiv 2209.04911, IEEE CoG 2022) is a competition for solving
"Baba is You" — a puzzle game where player actions can change the game's
own rules (push "BABA IS YOU" blocks to make a new object be the player,
change win conditions, etc.). Moving blocks doesn't just change position —
it changes the *transition function* itself.

ls20 L5's color-changing and rotation blocks are a milder version of this:
they don't change the rules, but they change environment state as a side
effect of the player's move. The Keke AI research shows:

- **Dynamic mechanic space**: the effective action model is not fixed — the
  same player action has different effects depending on which reactive blocks
  are adjacent. BFS must re-evaluate action effects at each state, not
  precompute them.
- **Rule/mechanic tracking as atoms**: the best Keke solvers track mechanic
  states (which rules are currently active) explicitly as part of the novelty
  atom set. For us: track `(block_id, rotation_state)` and `(block_id,
  color_state)` as IW atoms, not their constituent pixels.
- **Default agent strategy**: exhaustive BFS with a heuristic that
  prioritizes states where more rule-changes have been achieved — analogous
  to our progress signature, but sensitive to reactive-block state changes.

**Source**: https://arxiv.org/pdf/2209.04911 |
"Baba is LLM" (2025, reasoning about dynamic rules):
https://arxiv.org/html/2506.19095v1

### Analogy 3: Concurrent effects planning (PDDL)

Formally, ls20 L5's mechanics are a **concurrent effects** planning problem.
Each player action `move(dir)` has effects:
- `player_pos ← new_pos`
- `block_i_color ← f_i(player_pos, block_i_color)` for each reactive block i
- `block_j_rotation ← g_j(player_pos, block_j_rotation)` for each rotation block j

Classical PDDL planners handle this with **conditional effects**:
`when (player_at X) do (block_color ← new_color)`. The key insight from
concurrent-effects planning literature (arXiv 1906.08157): you can split
the joint planning problem into a **relaxed problem** (ignore block coupling)
to get a fast heuristic, then use that heuristic to guide the full search.
The relaxed problem is solvable in polynomial time; its solution length is
an admissible lower bound on the full problem.

**Source**: "Solving Multiagent Planning Problems with Concurrent Conditional
Effects", arXiv 1906.08157. https://arxiv.org/pdf/1906.08157

### Analogy 4: Lights Out (linear algebra approach)

Lights Out is the canonical reactive-tile puzzle: pressing a cell toggles
it and its 4 neighbours. It looks like a search problem but has an exact
algebraic solution — represent the board as a vector over GF(2), pressing
cell i is matrix multiplication, and the solution is the null space of the
transition matrix. For ls20's rotation/color blocks:

- If the reactive rule is **deterministic and position-dependent** (e.g.,
  block always rotates 90° clockwise when player is in row R), the reactive
  block's state is a **function of the player's path**, not an independent
  variable. The search space collapses: instead of tracking block state
  separately, compute it from the player path.
- If the reactive rule is **proximity-triggered** (block changes when player
  is within N cells), the state space is still (player_pos × block_states)
  but there are strong constraints between them — constraint propagation
  can prune most combinations before search begins.

**Lights Out solver**: https://www.dcode.fr/lights-out-solver

### What v13_3 should do for reactive blocks

**Short term — better atoms for IW**:
Track each reactive block's coarse state (color index 0-9, rotation index
0-3) as explicit IW atoms: `('block_color', block_id, color_idx)` and
`('block_rot', block_id, rot_idx)`. Currently IW tracks only per-pixel
colors — a rotating block generates 16 new (x,y,color) atoms per rotation,
overwhelming the novelty table with noise. One coarse atom per block per
rotation step is far more informative and keeps IW's width guarantee tight.

**Medium term — reactive-block heuristic for A***:
Detect reactive blocks (objects whose pixels change without the player
touching them). Define a joint heuristic: `h = manhattan(player, goal) +
sum_i(steps_to_correct_state(block_i))`. This guides search toward player
paths that simultaneously route toward the goal AND drive reactive blocks
into required configurations.

**Long term — constraint propagation prepass**:
Before search, characterize each reactive block's transition function
(what player positions/paths trigger what block-state changes). Build a
constraint graph: "to reach goal configuration, block_i must be in state S,
which requires the player to have passed through cell C in direction D."
Propagate constraints backward to prune the search space before BFS starts.
This is the Lights Out / PDDL conditional-effects approach applied to ARC.

### Sources (reactive/co-moving objects)

- [PushWorld: Manipulation planning with transitive push effects (DeepMind, 2023)](https://arxiv.org/abs/2301.10289)
- [Keke AI Competition: Baba is You with dynamic mechanics (IEEE CoG 2022)](https://arxiv.org/pdf/2209.04911)
- [Baba is LLM: Reasoning about dynamic rules (2025)](https://arxiv.org/html/2506.19095v1)
- [Solving Multiagent Problems with Concurrent Conditional Effects (2019)](https://arxiv.org/pdf/1906.08157)
- [Search-Based Path Planning among Movable Obstacles (2024)](https://arxiv.org/pdf/2410.18333)
- [Conflict-Based Search for Multi-Agent Path Finding with Movable Obstacles (2025)](https://arxiv.org/html/2509.26050v1)
