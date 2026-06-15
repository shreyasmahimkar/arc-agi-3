"""Regenerate ITERATIONS.md from v17_results.json for all iterations."""
import json, os
HERE = os.path.dirname(__file__)
d = json.load(open(os.path.join(HERE, "v17_results.json")))["iterations"]


def row(i):
    it = d.get(str(i))
    if not it:
        return None
    l5 = it.get("l5") or it.get("l5_round1") or it.get("l5_probe") or {}
    bench = it.get("bench", {})
    bp = it.get("best_progress", l5.get("best_progress", "-"))
    status = "**SOLVED**" if it.get("L5_SOLVED") else l5.get("status", "-")
    return (i, it.get("name", "-"), bench.get("levels_completed", "-"),
            bench.get("rhae", "-"), status, l5.get("best_depth", "-"),
            bp, l5.get("expansions", "-"))


md = ["# v17 — iteration log (auto-generated, 25 iterations)\n",
      "Target: **ls20 L5** — the dual-key level. v13 solved L0–L4 "
      "(13/45/39/43/44 actions) and breadth-died on L5 (~80k states).\n",
      "L0–L4 are re-verified through the REAL engine every iteration "
      "(`lc=5` means all five replay correctly; no self-reported wins).\n",
      "`depth` = deepest action sequence searched. `prog` = v17 progress "
      "signal = # of engine scalar-attrs (key colour/rotation, goal-match "
      "flags) moved vs level start; the dual-key WIN needs the full set.\n",
      "| iter | approach | L0–L4 | RHAE | L5 status | depth | prog | expansions |",
      "|---|---|---|---|---|---|---|---|"]
for i in range(1, 31):
    r = row(i)
    if not r:
        continue
    md.append(f"| {r[0]} | {r[1]} | lc={r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[7]} |")

md += [
    "\n## Iterations 26–30 — one web-researched technique each\n",
    "| iter | technique | paper | effect on L5 |",
    "|---|---|---|---|",
    "| 26 | Subgoal Search / landmark re-root | Czechowski et al. 2021; HIGL (Kim et al. 2021) | reliably chains progress 1→2 by re-rooting at a landmark |",
    "| 27 | Best-First Width Search (novelty bonus) | Lipovetzky & Geffner 2012/2017 | reaches progress 2 in a single phase (greedy stalled at 1) |",
    "| 28 | Macro-actions (move-until-wall) | Botea et al.; options framework | one expansion covers depth ~230; progress 2 |",
    "| 29 | Type-based exploration (multi-queue) | Xie, Müller, Holte, Imai 2014 | progress 2 in 85 expansions (vs 2 rounds before) |",
    "| 30 | full stack (all of the above) | — | progress 2 reliably; progress 3 is the new wall |",
]

md += [
    "\n## Reading the trajectory\n",
    "* **1 → 3**: plain BFS depth 8 → ForgeNet-A\\* depth 38. The learned "
    "cost-to-go converts breadth-death into a goal-directed dive (200× fewer "
    "expansions than v13's breadth search).",
    "* **4**: TRM-PUCT depth 172 but unguided — policy overfit to 184 expert "
    "states drives one corridor. Cautionary: the value/policy prior must "
    "*reorder*, not *dominate*.",
    "* **6–11**: progress-shaping sweep. `greedy+progress` (iter 9) is the "
    "deepest single-shot at **depth 43** — pure heuristic+progress, no depth "
    "penalty.",
    "* **12–19**: expert iteration (SoS/ExIt). Harvest the most-progressed "
    "path → bootstrap cost-to-go labels → retrain ForgeNet → re-search. Holds "
    "depth 43 with fewer wasted expansions as the heuristic learns the "
    "progressed region.",
    "* **20–25**: scaled budgets. Iter **24 reaches progress 2** — the first "
    "time the search engages the *second* key of the dual-key lock. The "
    "ceiling is budget-reachable, not architectural.",
    "\nSee `logs/iterN.log` for the full per-iteration trace.",
]
open(os.path.join(HERE, "ITERATIONS.md"), "w").write("\n".join(md))
print("wrote ITERATIONS.md with", sum(1 for i in range(1, 31) if d.get(str(i))), "iterations")
