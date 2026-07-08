# Chronos v21 — Teacher/Student Ensemble (Epic C)

*The solvers are teachers around a table; a toddler (the learned intuitive prior)
watches them all. They teach each other — and the toddler — by leaving lessons on
one shared scratchpad (`brain/blackboard.py`), not by calling each other directly.*

This is the unifying architecture that lets us build the three deep tracks **in
parallel** without three disconnected efforts: Go-Explore, the world model, and the
intuitive brain all become **teachers and students around one blackboard**.

---

## 1. The shared scratchpad (built — `brain/blackboard.py`)

One persistent JSON per game (`brain/blackboard/<gid>.json`) holding the shared
"understanding of the world," with provenance on every lesson:

- **action_effects** — per action: tried / changed / won counts (the toddler's raw sense of "what does this button do").
- **fragments** — verified plans that reached a subgoal (Go-Explore / macro seeds), keyed by level + perceptual features.
- **dead_ends** — counterexamples to avoid (DREAMTEAM's "negative constraints").
- **cells** — Go-Explore archive: downsampled-frame signature → shortest path seen.
- **world_facts** — dynamics the model teachers learned, in words.

Teachers call `teach_*`; students call `hints(level)` → `{action_order, seed_plans, avoid, ...}`. `consolidate()` is the sleep teacher (dedup/compress/bound). Verified + offline-tested.

## 2. Who teaches what (the teaching links)

| Teacher | Writes to blackboard | Reads (as student) |
|---|---|---|
| **BFS / blitz** (search) | action_effects, verified fragments, dead_ends | action_order (try effective actions first) |
| **Go-Explore planner** (T1) | cells (novelty archive), fragments (seeds) | cells + seed_plans (return to promising cell, then explore) |
| **World model** (T2) | world_facts, predicted-transition fragments | fragments as few-shot; dead_ends as constraints |
| **Runtime coder** (Qwen) | world_facts (its hypotheses), fragments on a win | perception digest + seed_plans + world_facts in-prompt |
| **Intuitive brain / toddler** (T3) | — (it IS the student) | the WHOLE blackboard → distilled into `action_order`/scoring |
| **Evolve / consolidation** | compresses + promotes robust lessons | everything (wake-sleep) |

The toddler closes the loop: distilled from every teacher, it hands its learned
`action_order` / action-scoring back to guide all the searches next cycle. Searches
get smarter → produce better lessons → the toddler learns more. That's the flywheel.

## 3. The three parallel tracks (all read/write the blackboard)

**T1 — Go-Explore intuitive search (the ls20 L5 lever).** Upgrade the macro-BFS
planner (`ladder.py`) into a real Go-Explore: cell archive (`blackboard.cell_key`)
+ "return to the most promising/novel cell, then explore with macros," guided by the
toddler's `action_order`. This is what tames L5's 19k-state blowup that blind BFS +
plain macro-BFS could not. Env `V21_GOEXPLORE`.

**T2 — Executable world model (the generalization spine).** Generalise
`runtime_coder` into a *persistent, verified* per-game model (`brain/world_model.py`
→ `brain/wm/<gid>/`) that must reproduce recorded transitions, MDL-refactored, reused
across runs. Plans inside it (unscored) via the planner; teaches `world_facts` +
predicted fragments. Pays off most on unseen black-box games. Env `V21_WORLD_MODEL`.

**T3 — Intuitive brain / the toddler (the learned prior).** A frame-change / action
scorer (StochasticGoose-lite CNN, R11 — or start self-supervised online from
`action_effects`) that predicts which action meaningfully changes the frame / moves
toward novelty. Replaces the frequency prior behind the fixed `IntuitionPrior.order_actions`
interface, and guides T1. Env `V21_TODDLER`. TRM (R9) is the recursive-refinement variant.

All three are **additive + env-gated + verify+shortest-gated + offline-covered** —
same invariants as the rest of the brain. They can be built in any order because they
only ever meet at the blackboard.

## 4. Build protocol for the 4h loop

Each loop cycle picks the highest-value unblocked step across T1/T2/T3, writes it
pure + offline-tested + committed + env-gated OFF, then a Mac cadence proves it and
flips its flag on. The blackboard is the integration point, so tracks never block
each other. Success metrics: **ls20 L5 solved** (T1), **held-out game solved by a
retrieved concept** (T2/blackboard generalisation), **toddler-guided search solves a
wall in fewer states than blind** (T3).

Wiring order (lowest-risk first): (a) teachers START populating the blackboard from
the existing cascade (BFS/blitz/planner write action_effects + fragments + cells) →
(b) searches READ `action_order`/`seed_plans` before expanding → (c) T1 Go-Explore
consumes the cell archive → (d) T3 toddler distilled from action_effects → (e) T2
persistent world model. Steps (a)/(b) are cheap and make every later track better.
