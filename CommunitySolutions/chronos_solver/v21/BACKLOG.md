# v21 autonomous-coder BACKLOG (worked top-down by the Opus 4.8 loop)

The between-rounds coder (Opus 4.8) picks the highest-priority **unblocked** item each
cycle, implements it in `v21/` only, compile+mock-tests it, commits, and logs the attempt
in `ITERATION_LOG.md`. Wall analysis (official baselines): solved = ls20 L0–L4, ft09 L0–L1,
vc33 L0–L3. **Walls to crack: ls20 L5–L6, ft09 L2–L5, vc33 L4–L6.**

Rules the coder follows: edit only under `v21/`; never touch `v19/`/`v20/`; always
`py_compile` + run `test_offline.py` before commit; never delete verified solutions
(append-shorter only); one item per cycle unless trivial.

## P0 — unblock the loop (do first)
0. **Runner stale-run self-heal.** [DONE 2026-07-10 — offline-verified] `run_cadence.sh`
   pre-flight now reaps a `cadence_runner.py` older than `V21_STALE_SECS` (3h) and clears a
   stranded `logs/.cadence.lock` so a hung stage no longer needs a MANUAL
   `pkill+rm+launchctl start` (that class stalled the runner ~7h twice: 164123Z + the
   08:53->14:48 gap). Live runs (flock-guarded) untouched. `test_reaper.sh` 7/7 PASS.
   *Remaining:* a human still restarts the Mac ONCE to clear the current stall; then confirm
   a future stage-hang auto-recovers on the next launchd tick.
   [HEARTBEAT DONE 2026-07-10 — offline-verified] `run_cadence.sh` now writes epoch-stamped
   `logs/.last_start` (on start) and `logs/.last_end` (on finish, with exit code) so the health
   check reads liveness from one line instead of parsing cron_*.log names + converting local
   mtimes to UTC. `.last_start` newer than `.last_end` by >~90m => hung/died mid-pass;
   `.last_start` older than the launchd interval => scheduler not ticking. test_reaper.sh 8/8.
   *Note:* the 02:01 UTC stall was launchd NOT ticking (16:41 cadence SIGKILLed, no live proc) —
   the reaper only fires on a tick, so a dead scheduler still needs one manual restart.
   [COMMIT-HELPER HARDENED 2026-07-11 — offline-verified] `git_safe_commit.sh` no longer loses
   commits to the THREE failure modes that stranded ~9 cycles of verified work (Jul 9-11):
   (a) the network `git push` now runs **DETACHED** (`( git push … ) &`) so add+commit always
   land within the caller's exec cap even when the push is slow/blocked (the ~45s sandbox cap
   used to time out the whole call and drop the commit); (b) when `.git/index.lock` is present
   but un-removable (mounted `.git` EPERM in the sandbox), it commits via **PLUMBING** through a
   private `GIT_INDEX_FILE` (read-tree HEAD → `add -A -- v21` → write-tree → commit-tree →
   update-ref), never touching `.git/index`; (c) when `.git/HEAD.lock` is ALSO un-removable
   (a prior interrupted `git commit` left one — this defeats even `update-ref HEAD`, since
   moving the checked-out branch needs HEAD.lock), it lands the commit on a **SIDE ref**
   `refs/heads/v21-auto-sync` (fresh, creatable lock) and prints the one-line fast-forward the
   Mac runs after the stale lock clears (`git update-ref refs/heads/main <sha>`). Modes (b)+(c)
   automate by hand-free code the GIT_INDEX_FILE + plumbing recovery the last three cycles did
   manually. Env: `V21_FORCE_ALT_INDEX=1` forces plumbing, `V21_FORCE_SIDE_REF=1` forces the
   side ref, `V21_NO_PUSH=1` skips push (all for the test). New `test_git_safe_commit.sh` builds
   a throwaway repo and covers all three paths + no-op + detached-push contract: **11/11 PASS**.
   test_offline 174 + test_reaper 16/16 still green. *This cycle's own commit landed on
   `refs/heads/v21-auto-sync` — the Mac must `git update-ref refs/heads/main <sha> && git push`
   (or just fast-forward) after restart clears the stale HEAD.lock/index.lock.*
   [ONE-COMMAND RECOVERY DONE 2026-07-12 — offline-verified] `resume_loop.sh` collapses the
   whole recurring recovery (reap hung `cadence_runner` -> clear `.cadence.lock` -> clear only
   genuinely-stale git locks, age+live-git guarded -> **fast-forward `main` to `v21-auto-sync`
   iff it is a strict ancestor & the tree is clean** -> push -> `launchctl start
   com.chronos.v21.cadence`) into `bash resume_loop.sh`. Never merge-commits/resets/clobbers a
   dirty tree; `DRY_RUN=1` previews. New `test_resume_loop.sh` 10/10 (ff/idempotent/diverged/
   dirty/stale-lock/DRY_RUN/no-side-ref). A DRY_RUN vs the REAL repo found SIX stranded locks
   (index/HEAD/packed-refs/main/v21-auto-sync/v21-scratch-probe) & `main` 17 behind `v21-auto-sync`.
   *Remaining:* the human runs it ONCE on the Mac after a restart to relight the loop + land the
   17 stranded commits.
1. **P2 config-aware evolve evaluator.** [CODED + offline-verified — live probe env-gated]
   `evolve.config_aware_eval_fn` scores corpus-floor + wall RHAE under the challenger's config;
   `cadence_runner._make_evolve_probe` applies `blitz_K`→BFS budget on the real engine.
   *Remaining:* run a Mac cadence with `V21_EVOLVE_PROBE=1` (+ enough `--bfs-timeout`) so a
   challenger that raises `blitz_K` actually solves a budget-gated wall and PROMOTES.
2. **Live blitz Stage-0.** [CODED + offline-verified — live effect on next Mac run]
   `blitz.py` (`blitz_solve` + `blitz_for_solver`) ports "race cheap wins on the fork first"
   (each action once, repeat-action×K, click-each-object); wired into `cadence_runner.solve_game`
   as Stage-0 for UNSOLVED levels only (verify + shortest-gated, env `V21_BLITZ`). Commit a962f8c.
   *Remaining:* a Mac cadence to confirm it commits a cheap win on ft09 L2–L5 / vc33 L4–L6.
3. **Wire `runtime_coder` as cascade Stage 3.5.** [CODED + offline-verified — live effect on
   next Mac run] `cadence_runner._runtime_coder_for_solver` builds obs from `_make_start_state`
   + one-step transitions and calls `RuntimeCoder.solve_level` with a fork-replay `try_plan`
   (`runtime_coder.replay_wins`); wired into `solve_game` as the LAST stage for UNSOLVED walls
   only, env `V21_RUNTIME_CODER` (default OFF). Verify + shortest-gate + exploit-refusal all
   preserved. Commit cc493d7. *Remaining:* a Mac cadence with `V21_RUNTIME_CODER=1` (local Qwen
   pulled) that solves a BFS/blitz-blocked wall end-to-end via generated code.

## P1 — crack the walls
   [R15 DE-BLIND DONE 2026-07-10] `brain/teacher._call` now surfaces the API's JSON error
   body (helper `_http_error_detail`) on any HTTP failure, re-raising an HTTPError with the
   same `.code`. Motivated by run 050254Z's fresh `HTTP 400` on the ft09 L2 teacher/WM calls
   (0 in the prior 5 runs) that was invisible.
   [R16 ROOT-CAUSE READ 2026-07-10] Run 065844Z revealed the exact 400 on EVERY wall:
   `Your credit balance is too low ... go to Plans & Billing`. NOT a prompt-shaping bug —
   a BILLING block. `brain/teacher` now latches `_CLOUD_DISABLED` on the first credit-
   exhausted 400 (`_note_cloud_error`), flips `available()` to False so teacher/WM/arch
   skip cleanly (no wasted per-wall grounding prep, no repeat warnings), logs ONE loud
   banner, and auto-recovers next run once credits are topped up. +9 offline checks.
   ⛔ **OPS BLOCKER: the cloud teacher stays dark until API credits are added — this is
   now the single blocker on all 3 walls.** *Next cycle (while cloud is down): push the
   LOCAL levers — confirm blitz/brain_planner/go-explore/runtime_coder actually fire on
   the walls (they log only on success today) and deepen the strongest one.*
   [R17 OBSERVABILITY DONE 2026-07-10] `cadence_runner._local_stage_note` (pure) is now
   the `else:` branch of all four local stages' verified-win `if` — a wall every local
   lever silently failed now logs a per-stage `STAGE fired: no candidate` /
   `STAGE fired: candidate len=N failed verify/shortest gate` line instead of a blank
   gap. +4 offline `obs note:` checks (141 PASS).
   [R18 BLITZ BREADTH DONE 2026-07-10] `blitz.blitz_breadth_note` + `blitz_for_solver(stats=)`
   + `_local_stage_note(extra=)` now append `| macros=N simple=N clicks=N tier=..` to the
   BLITZ miss line — R17's note said only the candidate length; this says WHY (e.g.
   `clicks=0` on a vc33 wall = no target enumerated vs `clicks>0` all-failed). Pure/offline;
   +4 checks (145 PASS).
   [C1+ CLICK-TARGET PLUMBING DONE 2026-07-10] Run 144827Z (first with R18 breadth)
   was diagnostic: vc33 L4 had BLITZ `clicks=30` yet BRAIN_PLANNER + GOEXPLORE fired
   'no candidate' in <1s. Root cause: `_goexplore_for_solver`/`_brain_planner_for_solver`
   passed NO `click_targets`, so both white-box planners searched an inert simple-only
   action set on the click-driven vc33 walls. New pure `_scan_click_targets` helper
   (mirrors `blitz_for_solver`: `_scan_actions` a==6 + optional B1 centroids under
   `V21_BRAIN_PERCEPTION`) now feeds both `plan_in_model_macro`/`_goexplore`. +3
   offline `click-only wall` checks (153 PASS); ls20/ft09 keyboard walls unaffected.
   [C1++ STAGE WATCHDOG DONE 2026-07-10] Run 164123Z exposed a starvation bug: ls20 L5's
   RUNTIME_CODER went SILENT 66 min (normally ~1 min, cf. 144827Z 11:02:37→11:03:31) — a
   hung local model held the whole sweep, so ft09 L2–L5 / vc33 L4–L6 never got a turn AND
   the C1+ click-target fix couldn't be observed on vc33. New pure `_call_with_deadline(fn,
   deadline)` (daemon-thread watchdog, mirrors OllamaBackend.complete one level up) wraps the
   RUNTIME_CODER call; env `V21_RUNTIME_CODER_BUDGET` (default 300s, <=0 = legacy inline).
   Timeout → `RUNTIME_CODER abandoned` note + move to next wall; corpus write stays main-thread
   post-verify (no daemon corruption). +5 offline `stage deadline` checks (158 PASS).
   [C1++++ CLICK-CAP DONE 2026-07-10] Run 164123Z's diagnostic showed ls20 L5 BLITZ `clicks=32`:
   C1+ fed those off-solution ACTION6 targets to the white-box planners too, inflating their
   branching 4->36 on the ls20 walls (BFS solves ls20 with 4 simple actions). New pure
   `_planner_click_cap(gid)` suppresses clicks for keyboard-tier games (ls20 -> 0) while keeping
   them UNLIMITED for the click/reflex tiers (vc33/ft09, C1+ preserved); explicit int
   `V21_PLANNER_CLICK_CAP` overrides any tier. `gid` threaded through both planner helpers.
   +7 offline `planner click-cap:` checks (168 PASS). *Watch:* next Mac run's ls20 L5-L6
   BRAIN_PLANNER/GOEXPLORE reach deeper on the tight 4-action basis.
   [C1+++++ CODER OOM GUARD DONE 2026-07-11 — offline-verified] Root-caused the recurring
   ~13h stall: run 164123Z's coder stage was `Killed: 9` (SIGKILL/OOM) on ls20 L5 right after
   a 59k-state BFS left ~20k unique frames resident — the OOM ended the WHOLE pass mid-sweep and
   stranded `.cadence.lock`. C1++'s `_call_with_deadline` guards WALL-CLOCK, not MEMORY. New pure
   `_coder_mem_skip(rss_mb, ceiling_mb)` + `_process_rss_mb()` (platform-correct ru_maxrss: bytes
   on darwin / KiB on Linux) SKIP the memory-heavy ollama coder when RSS>=ceiling so the pass
   exits clean (lock released) instead of OOM-killing the sweep. Env `V21_RUNTIME_CODER_MAX_RSS_MB`
   (default 0 = OFF; suggest ~6500 on the 16GB M1 Pro). Wall stays UNSOLVED either way (no
   regression). +6 offline `coder mem-guard:` checks (174 PASS). *Remaining:* set the ceiling env
   on the Mac + confirm a future big-BFS wall logs `RUNTIME_CODER skipped: ... (OOM guard)` and the
   pass writes `cadence exit=0` instead of dying.
   *Remaining (deepen the strongest):* now that the sweep reaches ALL walls, read the next Mac
   run's `*_fired:*` lines (incl. BLITZ breadth) on ls20 L5–L6 / ft09 L2–L5 / vc33 L4–L6,
   pick the lever whose candidate gets closest to a win, and deepen it (e.g. raise
   `V21_PLANNER_STATES`/tune `V21_GOEXPLORE_BINS`, or seed the L4 end-state suffix-BFS).
4. **ls20 L5–L6 (LADDER / Go-Explore).** Variant re-root + TTRL suffix-BFS from the L4 end state.
   [PARTIAL — offline Go-Explore seed CODED] `blitz.blitz_macros` replays solved sibling-level
   plans (shortest winning prefix) as a Tier-0 wall seed, wired into `blitz_for_solver`. Commit
   9f69b20. *Remaining:* suffix-BFS from the L4 end state on the live engine; a Mac cadence to
   confirm a sibling macro (or seeded BFS) registers `levels_completed>=6`.
   [BUDGET-RESERVE DONE 2026-07-07] `cadence_runner._should_resolve` now skips re-BFS of
   already-solved+verified corpus levels (env `V21_RESOLVE_SOLVED`, default OFF), so L0–L4 replay
   in seconds and the ls20 BFS reaches L5 with the full per-level budget instead of ~0 (run
   220311Z burned ~1686s re-solving L0–L4 before L5 even started). See ITERATION_LOG for commit.
   *Still needed for the win:* L5 depth exceeds plain BFS — pair the reserved budget with
   `V21_RUNTIME_CODER=1` (Go-Explore/coder) or a seeded suffix-BFS from the L4 end state.
   [GO-EXPLORE STAGE-3.4 DONE 2026-07-08] `brain/planner.plan_in_model_macro` (→ committed
   `ladder.macro_bfs`: corridor-collapsing macro edges + single-step precision) wired as
   `cadence_runner._brain_planner_for_solver` Stage-3.4 over the white-box engine for UNSOLVED
   walls, env-gated `V21_BRAIN_PLANNER` (default OFF), verify+shortest-gated. +2 offline checks.
   Confirmed by run 011103Z: plain BFS timed out on ls20 L5 (57k states/600s) — macro reach is
   the missing ingredient, not depth.
   [RUNNER-WIRED 2026-07-08] Root-caused the 2nd FLAT run (025330Z): the Stage-3.4 planner was
   coded but `V21_BRAIN_PLANNER` was never exported in `run_cadence.sh`, so it never fired on the
   Mac. Added `export V21_BRAIN_PLANNER=1` to the runner's wall-cracking block (before RUNTIME_CODER).
   Budget-reserve (skip-re-BFS of solved L0-L4) is already active by default, so V21_RESOLVE_SOLVED
   is intentionally left OFF. *Still needed for the win:* the NEXT Mac cadence should now actually
   run the macro planner on ls20 L5/L6 and register `levels_completed>=6`; if still flat, add the
   L4-end-state suffix-BFS seed or the R13 Opus teacher.
5. **ft09 L2–L5.** Investigate mechanics (these aren't blind like L0); deepen BFS/graph budget,
   add object-aware click targets. *Done when:* ≥1 of L2–L5 solved+verified.
   [ADAPTIVE TODDLER EPOCHS DONE 2026-07-12 — offline-verified, cloud-free] ft09's world model is
   the loop's single weakest measured component (opus_arch champion_acc=0.8526 vs ls20/vc33=1.0),
   and BRAIN_PLANNER/GOEXPLORE plan AGAINST that WM on ft09's 4 walls. `brain/toddler_net`
   `last_champion_acc(game)` (reads newest champion_acc off `opus_arch.jsonl`, no cloud) +
   `adaptive_epochs(acc)` (ft09 0.8526→14 epochs, 1.0→8, None→8=today) now drive the
   `cadence_runner` toddler-train loop (`train(epochs=adaptive_epochs(last_champion_acc(gid)))`).
   +11 offline checks (187 PASS) incl. a source-introspection wire guard. *Remaining for the win:*
   a Mac cadence where ft09's deeper-trained WM lifts champion_acc→~1.0 and a white-box planner then
   solves ≥1 of L2–L5 against the now-trustworthy model.
   [TEACHER-GROUNDING CODED + offline-verified 2026-07-09] The Opus teacher's FIRST-round prompt
   now carries the level-start valid ACTION6 click targets (B1 perception centroids) via
   `_teacher_click_note` in `_opus_teacher_for_solver`, env `V21_TEACHER_GROUND` (=1 in
   run_cadence.sh). Root cause (cron 152556Z): the teacher was guessing x,y from source →
   round-1 first click a no-op on empty space. +6 offline checks. *Remaining:* a Mac cadence
   where the grounded round-1 plan shows non-zero delta past action index 0 on ft09 L2.
   [WM-EXTRACTION FIX 2026-07-10] cron 011206Z/231705Z both logged `ft09 L2 opus WM exec
   failed: invalid syntax (<world_model>, line 1)` — `brain/teacher._strip_module` only
   stripped a ``` fence when the reply STARTED with it, so an Opus prose preamble / unclosed
   fence reached compile(). Now extracts the first fenced block anywhere, tolerates an
   unclosed fence, and drops a leading prose preamble (commit 86ba254, +11 offline strip:
   checks). *Remaining:* a Mac cadence where ft09 L2 logs a built WorldModel + candidate_plans
   attempt instead of `exec failed`.
   [PROBED-CLICK GROUNDING 2026-07-10] cron 030701Z: ft09 L2 WM now execs (strip fix landed)
   but round-2 STILL led with a no-op ACTION6 (`first no-op at action index 0 (6)`, delta 0).
   Root cause: R8/B1 handed Opus unverified perception CENTROIDS. `cadence_runner._probe_click_targets`
   now forks the level-start state and actually PERFORMS ACTION6 at each centroid (≤8 probes),
   and `_format_click_note` recommends only VERIFIED-effective targets ("prefer these") while
   flagging verified no-ops ("never lead"); `_teacher_effective_click_note` wires it in and falls
   back to the static note offline (+6 offline `probed grounding:` checks). *Remaining:* a Mac
   cadence where ft09 L2 round-1/2 no longer opens with a dead click.
6. **vc33 L4–L6.** Click-orchestration: better connected-component click-target selection in
   `graph_explore`. *Done when:* ≥1 of L4–L6 solved+verified.
   [TEACHER-GROUNDING CODED 2026-07-09] Same `V21_TEACHER_GROUND` grounding fixes vc33 L4's
   round-1 no-op click (152556Z: "first no-op/failure at action index 0 … delta 0 cells"). The
   teacher now sees the L4 start scene's object centroids. *Remaining:* a Mac cadence confirming
   round-1 progress; for mixed-coord walls, still needs better click-target ordering.
   [PARTIAL — blitz click-REPEAT tier CODED] `blitz.blitz_solve` now repeats a single ACTION6
   coord ×K (shortest-gated), matching vc33's "hammer one component" endings (commit e049348).
   *Remaining:* a Mac cadence to confirm it commits a wall (L4–L6) whose win is a fixed-coord
   repeat; for mixed-coord walls, still needs better click-target ordering/selection.

## P2 — optimality & generalization
7. **ls20 L1 tighten 45→≤41** (only sub-1.0 solve). Masked/A* BFS or suffix trim. *Done when:* RHAE(L1)=1.0.
   [DONE] Corpus `solutions/ls20.json` L1 is 41 actions → RHAE(L1)=1.0 as of Mac run 20260706T194329Z.
8. **Trained intuition prior.** Replace corpus-frequency prior with a small policy net over
   frame features; keep the `order_actions` interface. *Done when:* held-out solve-rate improves.
   [ELEVATED by user steer 2026-07-08 → see R9 (TRM/HRM tiny recursive scorer) and R11 (Tufa
   StochasticGoose CNN frame-change head) for the concrete open-model implementations.]
9. **Cross-game macro retrieval (Stage 1b).** Use `intuition`/macro bank to seed BFS on a
   *similar* held-out game. *Done when:* a macro from one game solves a level of another.
   [PARTIAL — within-game macro replay CODED] `blitz.blitz_macros` replays a solved level's plan
   on another (wall) level of the SAME game (commit 9f69b20). *Remaining:* extend the macro source
   to `v21_macro_bank.json` / cross-GAME retrieval and seed BFS (not just full-plan replay).

## P3 — infra / submission
10. **Stall alarm.** Reporter pings if no cron_*.log in 8h.
    [DETECTION DONE 2026-07-11 — offline-verified] `health_check.sh` reads last cycle's
    `logs/.last_start`/`.last_end` heartbeat and prints ONE verdict line
    (`RUNNER: HEALTHY|RUNNING|HUNG|STALLED|UNKNOWN | detail`), replacing the per-cycle
    cron-name + sandbox-mtime→UTC hand-conversion that has produced off-by-4h judgements.
    Branches: start-after-end within 90m = RUNNING; start-after-end > `V21_HEALTH_HUNG_SECS`
    (5400s) = HUNG (died/SIGKILLed mid-pass); last tick > `V21_HEALTH_STALL_SECS` (9000s) =
    STALLED (launchd not ticking); no heartbeat = cron-mtime fallback; empty = UNKNOWN.
    Drops `logs/.stall_flag` (reason+epoch, gitignored) when stalled/hung and clears it when
    healthy — the one file a future *ping* half can act on. Pure/offline; exit 0/1/2.
    `test_reaper.sh` +8 `health:` checks (16/16 PASS). Validated on the real logs this cycle:
    correctly returned STALLED (newest cron 673m ago). *Remaining:* the PING side — have the
    reporter/an alarm consume `.stall_flag` (or wire a launchd `WatchPaths`/`StartInterval`
    that isn't blocked by the same dead scheduler).
11. **Kaggle offline notebook.** Bundle Qwen2.5-Coder as a dataset, `HF_HUB_OFFLINE=1`, embed
    agent+engine+cache; verify it runs network-off on a T4.
12. **Config-aware `MyAgent` load** of `champion.json` (blitz_K/action_order/heuristics).

# =====================================================================
# EPIC B — Cognitive ("brain") layer (game-general agent)
# Full design + rationale + research refs: BRAIN_ARCHITECTURE.md
# =====================================================================
# Goal: stack a cognitively-inspired layer (perception → executable world model
# → hypotheses → planner → memory → goal → consolidation) on top of the proven
# blitz→BFS→runtime_coder cascade, so the loop both cracks the remaining walls
# AND grows toward transfer to UNSEEN ARC-AGI-3 games. Spine = executable /
# program-synthesis world models (Rodionov 2026), NOT neural latent (that's B8).
# INVARIANT: the brain is additive — the proven cascade stays the fallback,
# every brain plan is still verify_solution + shortest-gated, each subsystem is
# wired live only behind its own env flag (default OFF) AFTER a Mac cadence
# proves it, and the offline submission guard is never disabled. All `brain/`
# code is pure/dependency-free at import and covered by test_offline.py.

## Epic B — phased build (each phase: green py_compile + test_offline, committed, env-gated OFF)
B1. **Perception scene-graph.** [DONE this session] `brain/perception.py`: connected-component
    objects (colour/size/bbox/centroid), frame-diff, and ACTION6 `click_targets` (one per
    component — fixes v19's per-colour-median clicks). 6 offline checks. [WIRED — env-gated,
    offline-verified] `blitz.merge_click_targets` fuses perception centroids with the scanned
    clicks (scan-first, deduped) in `blitz_for_solver` behind `V21_BRAIN_PERCEPTION` (default OFF);
    +3 offline checks; commit 77b5e69. *Remaining:* a Mac cadence with `V21_BRAIN_PERCEPTION=1` to
    confirm per-component click coords crack a vc33 L4–L6 wall the median-scan misses.
B2. **Executable world-model persistence + verifier.** [PARTIAL — `brain/world_model.py` verifier
    core + template DONE] Generalise `runtime_coder` to a per-game model on disk
    (`brain/wm/<game>/`) that must reproduce recorded transitions (`verify_model`, `is_trusted`);
    add an MDL refactor pass. *Done when:* a persisted model reproduces a solved level's
    transitions offline and the loop reuses it next run.
B3. **MPC plan-executor.** [PARTIAL — `brain/planner.py` `plan_in_model` + `execute_and_verify`
    cores DONE] Wire to the real engine: plan inside the trusted model (unscored), execute with
    step-wise frame-mismatch abort; scored actions only on verified plans. *Done when:* a level is
    solved via model-planned + executor-verified actions on a Mac cadence.
B4. **Hypothesis manager (anti tunnel-vision).** [PARTIAL — `brain/hypotheses.py` `falsify` +
    `most_discriminating_action` cores DONE] Seed 2–3 competing WorldModels; spend scored actions
    on the most-discriminating move; falsify on mismatch. *Done when:* on a wall, discriminating
    exploration reaches a trusted model in fewer scored actions than greedy.
B5. **Goal induction.** [PARTIAL — `brain/goal.py` score-signal inducers DONE] Add frame-motif
    goal induction (perception motif + memory). *Done when:* induced goal drives a solve with no
    hand-coded goal.
B6. **Cross-game concept library.** [PARTIAL — `brain/memory.py` perceptual key + retrieval DONE]
    Persist macros + WM fragments + motifs to a bank keyed by perceptual signature; retrieve to
    seed a DIFFERENT game's search. *Done when:* a concept learned on one game solves a level of
    another (the Epic-B success metric). Subsumes/extends legacy #9.
B7. **Wake-sleep consolidation.** Extend `evolve`: replay solved trajectories, compress/refactor
    the library, re-distil the intuition prior. *Done when:* held-out solve-rate improves after a
    consolidation pass.
B8. **(Optional, far) Neural latent world model.** H-JEPA/Dreamer-style latent predictive model
    behind the same planner/goal interfaces. Blocked on a GPU training path; not offline-verifiable
    in 4h increments — do NOT start until B2–B7 are solid and a training route exists.

# =====================================================================
# RESEARCH FEED — R1–R5 integrated 2026-07-07; R6 added (RESEARCH-2); R7–R8 added (RESEARCH-3)
# Latest ARC-AGI-3 literature scanned (arXiv / ARC Prize / Kaggle). Each item is
# ADDITIVE, env-gated when shipped, and slots into the proven cascade + brain.
# These VALIDATE the current direction (executable world models + explore-first)
# and add concrete, prioritized mechanisms. Do not duplicate B2–B6 — these refine
# them with specific techniques from the papers.
# =====================================================================
R1. **Explore→Verify→Plan with a belief-entropy COMMIT GATE** (AERA, arXiv:2605.25931,
    "Explore Before You Solve"). Strongest single finding: what enables non-zero RHAE on
    hidden-rule games is maintaining explicit world-model HYPOTHESES and *gating the switch from
    exploration to planning on a proxy for belief entropy* (uncertainty over models). Also gives a
    concrete budget heuristic: spend ≈40% of the human baseline on exploration before committing.
    Their public-set taxonomy explicitly places our walls — ft09 = blind-ACTION6 reflex; ls20 =
    budget-constrained repeated-action (50–200 steps); vc33 = probe-then-ACTION6. *Action:* sharpen
    B4 into a real commit gate — in `brain/hypotheses.py`, don't hand off to the planner until the
    surviving-hypothesis set collapses (entropy proxy below threshold) OR the ≈40%-of-baseline
    explore budget is spent; expose the budget as an env knob. Pure/offline-testable (entropy proxy
    + gate over injected hypotheses). *Done when:* on a wall, the gate reaches a single trusted model
    in fewer scored actions than greedy BFS. NOTE: their 55-game code-track entry is "BFS + offline
    pre-solve cache" at RHAE 0.30 — i.e. OUR architecture — good external validation.
R2. **Verify → MDL-refactor → plan-through-model** (Executable World Models, arXiv:2605.05138).
    The verifier-driven executable-WM loop (verify against observations → refactor toward SIMPLER
    abstractions as an MDL proxy → plan through the model before acting) is exactly B2/B3; the paper
    reports 15/25 games solved at RHAE 58% with a strong coder model. Two concrete adds: (a)
    prioritize the **MDL refactor pass** in `brain/world_model.py` (shorter program that still
    reproduces all recorded transitions → better generalization), and (b) the paper flags that WM
    quality "varies substantially across runs" → add **best-of-N / multi-hypothesis** WM synthesis
    (ties into R1's competing hypotheses). *Action:* bump B2's MDL-refactor to the next CODE-branch
    item once a Mac run gives runtime_coder live signal; keep it env-gated + verifier-gated.
R3. **Graph-based level explorer + frame processor** (arXiv:2512.24156). Method = Frame Processor
    (image segmentation → status-bar detection & MASKING → priority-based action grouping → STATE
    HASHING) feeding a Level Graph Explorer (explicit state-graph, action-selection strategy,
    FRONTIER MANAGEMENT). We already have transient/status-bar masking and connected-component
    perception (B1); the new, directly-usable pieces are **state hashing for dedup** and **frontier
    management** to make BFS explore unique states instead of re-expanding, plus **priority-based
    action grouping** to order ACTION6 targets. *Action:* fold state-hash dedup + priority action
    grouping into the BFS/planner for vc33 L4–L6 (item #6); offline-testable on a mock state graph.
R4. **Speed–Depth / RHAE is quadratic** (2605.25931, §3). RHAE = (human/AI actions)², so a solve
    that uses 2× human actions earns only 25% credit; budget-constrained repeated-action wins
    (ls20-style) are penalized hard for length. *Action:* keep the shortest-plan gate strict and add
    a suffix-trim/A* optimality pass for repeat-heavy solves (extends legacy #7/#8) — a solved-but-
    long wall should be revisited to SHORTEN, not just left at RHAE<1.
R5. **Test-time training on a tiny model** (NVARC 2025 winner: Qwen-4B + TTT + synthetic data,
    24% on ARC-AGI-2, ARC Prize 2025 report arXiv:2601.10904; TRM test-time adaptation
    arXiv:2511.02886). This is the static-grid (v1/v2) recipe; less direct for interactive v3 but
    relevant to a future learned intuition prior (item #8 / B8). *Action:* park as a B8 reference —
    do NOT start (needs a GPU training route); revisit only after B2–B4 are live.
R6. **Orchestrator + subagents with COMPRESSED-summary context control** (Symbolica *Arcgentica*,
    open-source `symbolica-ai/arcgentica` + `symbolica-ai/ARC-AGI-3-Agents`; blog
    https://www.symbolica.ai/blog/arc-agi-3 ; scores 36.08% on the 25-game public set = 113/182
    levels, 7/25 games, and solves all 3 public envs incl. ours). Integrated 2026-07-07 RESEARCH-2
    cycle — NEW vs R1–R5. Key mechanism: a top-level orchestrator never touches the environment; it
    delegates to specialised subagents that return **compressed textual summaries**, which caps
    context growth so a long exploration transcript never blows the coder's context window. This is
    directly relevant to our `runtime_coder` Stage-3.5 and the brain planner on LONG walls (ls20
    L5–L6, ft09 L2–L5): today we feed raw obs/one-step transitions into the coder model, which grows
    unbounded as exploration deepens. *Action:* add a pure `brain/summarize.py` (or a helper in
    `runtime_coder`) that compresses recorded transitions into a fixed-size structured digest
    (object/scene deltas + tried-action → outcome table) BEFORE the coder prompt, env-gated
    `V21_CODER_DIGEST` (default OFF); slots between obs-build and `OllamaBackend.complete`. Keep it
    additive + offline-testable (digest is deterministic over a mock transition log; assert bounded
    length + lossless action→outcome recall). *Done when:* a Mac cadence shows the coder step
    completing (not stalling/OOM) on a deep wall where it previously timed out — ties into the
    160000Z stall root-cause. Do NOT adopt their multi-agent SDK wholesale (heavy dep, network);
    port only the summary-compression idea into the existing single-coder path.
    [CODED + offline-verified 2026-07-08] `brain/summarize.py::digest()` (pure, imports only
    brain.perception) + `runtime_coder._obs_block` swap the raw-grid `{obs}` block for a bounded,
    deterministic, perception-first digest behind `V21_CODER_DIGEST` (default OFF) — implements R6
    (bounded context) AND R8 (perception-first schema) in ONE env flag. 7 offline checks; see
    ITERATION_LOG. *Remaining:* a Mac cadence with `V21_CODER_DIGEST=1` (+ `V21_RUNTIME_CODER=1`,
    local Qwen) that yields a coder plan on ls20 L5 / ft09 L2 / vc33 L4 where the raw-grid prompt
    did not.
R7. **Workspace optimization: evolve the persistent substrate, not the weights** (NVIDIA/Technion
    *DREAMTEAM*, arXiv:2605.09650, "Workspace Optimization: How to Train Your Agent"; abs
    https://arxiv.org/abs/2605.09650 ). Integrated 2026-07-07 RESEARCH-3 cycle — NEW vs R1–R6, and
    currently the highest public-set score we've seen: **38.4% on the 25-game set (up from Symbolica's
    36.08%) using 31% FEWER environment actions per game.** Core idea maps almost 1:1 onto our loop:
    since a frozen frontier/coder model can't be weight-trained, treat the agent's *workspace* (the
    structured files it reads/writes/tests) as the trainable object, mirroring weight-space training —
    **artifacts↔parameters, evidence↔data, counterexamples↔losses, textual feedback↔gradients.** Our
    corpus/`champion.json`/`intuition_prior.json`/macro-bank ARE the artifacts, the scorecard +
    recorded transitions ARE the evidence, the regression gate + UNSOLVED walls ARE the
    counterexamples, and ITERATION_LOG/cron notes ARE the textual-feedback "gradient." *Action (two
    concrete, additive, env-gated adds):* (a) after each Mac run, `evolve` writes failed-wall
    transcripts as **counterexample artifacts** (`brain/wm/<game>/counterexamples.jsonl`: the wall,
    the tried plan, why it failed) that the NEXT cycle's `runtime_coder`/planner reads as
    negative-constraint context ("don't repeat these dead ends") — env `V21_WORKSPACE_COUNTEREX`
    (default OFF); pure + offline-testable (assert the coder prompt excludes a recorded dead-end
    action). (b) Adopt their **action-frugality objective**: in the shortest-plan/evolve scoring,
    tie-break challengers by env-actions-per-solve so the workspace evolves toward the 31%-fewer-
    actions regime (extends R4's quadratic-RHAE pressure). Slots into `evolve` + `runtime_coder`
    Stage-3.5 + brain `memory`; do NOT pull the DREAMTEAM multi-agent harness (network/heavy) — port
    only the workspace-as-trainable-substrate discipline. *Done when:* a Mac cadence shows a wall the
    coder previously failed is not re-tried down the same dead end (counterexample recall), or a
    challenger promotes on equal-RHAE-but-fewer-actions.
    [CODED + offline-verified 2026-07-09] R7(a) DONE: `_counterex_open/_counterex_notes/_counterex_record` in cadence_runner persist each FAILED Opus-teacher plan as a blackboard dead_end and feed prior-run dead_ends back as a 'do NOT repeat' note (`_opus_teacher_for_solver` notes + `_try_plan`); env `V21_WORKSPACE_COUNTEREX` (exported =1). +8 offline checks.
    [R7(b) DONE 2026-07-10] Action-frugality tie-break landed in `evolve.evolve_step` (optional `cost_fn=None`): at EQUAL held-out RHAE a challenger solving the walls in STRICTLY fewer env-actions promotes (`PROMOTE(frugal)`), preserving the generalization + strict-RHAE gates. Companion `evolve.config_aware_cost_fn(walls, probe_fn, miss_penalty)` totals same-probe env-actions (unsolved walls charged a penalty), degrading to 0.0 offline so nothing promotes on noise. +5 offline `frugality:` checks (150 PASS).
    [WIRED 2026-07-12 — offline-verified] `cadence_runner.main` now builds `cost_fn =
    evolve.config_aware_cost_fn(walls_by_game, probe_fn)` (the SAME live wall probe as the eval)
    and passes `cost_fn=cost_fn` into the `evolve_step` call, so a challenger that TIES held-out
    RHAE but solves the walls in fewer env-actions PROMOTES(frugal) on the Mac. Degrades inert
    offline (probe_fn None -> cost 0.0 for every config -> frugal branch can't fire); gen +
    strict-RHAE gates untouched. +2 offline `frugality wire:` source-introspection checks (176
    PASS) guard the wiring against removal. *Remaining:* a Mac cadence with `V21_EVOLVE_PROBE=1`
    where a cheaper equal-RHAE challenger logs `PROMOTE(frugal)`. *Also remaining (unchanged):* a
    Mac cadence where the teacher avoids last run's ls20 L5 near-miss.
R8. **Perception is the real bottleneck — feed the coder a symbolic scene description, not raw grids**
    (CMU/UMich/UCSD/UIUC, arXiv:2512.21329, "Your Reasoning Benchmark May Not Test Reasoning:
    Revealing Perception Bottleneck in Abstract Reasoning Benchmarks"; abs
    https://arxiv.org/abs/2512.21329 ). Integrated 2026-07-07 RESEARCH-3 cycle — NEW vs R1–R7.
    Controlled two-stage experiments show **~80% of VLM ARC failures are PERCEPTION errors, not
    reasoning**, and inserting a dedicated perception→natural-language stage before the reasoning
    model gives +11–13pp (up to 2.5× on Mini-ARC). Strong external VALIDATION of our design choice:
    v21 already parses frames with exact connected-component perception (B1) instead of a VLM, so we
    *sidestep* the perception bottleneck entirely — the paper argues that's exactly where the wins
    are. *Action:* make R6's `V21_CODER_DIGEST` digest **perception-first** — the digest the coder
    reads should be a structured scene description built from B1's `perception.py` objects
    (per-object colour/size/bbox/centroid + frame-diff deltas + tried-action→object-change table),
    NOT a serialized raw grid (the paper shows serialized grids are the hard format even for humans).
    This is a refinement of R6 (same env flag `V21_CODER_DIGEST`), not a new subsystem: swap the
    digest's representation from raw-grid to perception-object schema. Pure + offline-testable (assert
    the digest names each component and its post-action delta over a mock transition log). *Done when:*
    on a wall, the perception-object digest lets the coder reference objects by identity and the
    coder step yields a valid plan where the raw-grid digest did not.

# ---------------------------------------------------------------------
# STRATEGIC STEER (2026-07-08, user): raw BFS provably cannot crack ls20 L5/L6
# (117k states, 30k unique, still timed out at 1200s) — budget lowered 1200->600
# so passes turn over faster. MAIN FOCUS moves OFF search depth and ONTO: latest
# world-model research, small/intuitive OPEN models (TRM-style), Tufa Labs'
# interactive-v3 work, Opus-4.8 as a teacher on the network-enabled cadence box,
# and (long-horizon) an ensemble that combines them. Items R9–R14. All additive +
# env-gated; the offline submission corpus is never risked.
# ---------------------------------------------------------------------
R9. **Tiny Recursive Models (TRM) / HRM as the learned intuition prior** (elevates item #8; the
    user's "small intuitive open models" steer). TRM = 7M-param single tiny net that recurses on a
    latent "scratchpad" and backprops through ALL recursive steps; 45% ARC-AGI-1 / 8% ARC-AGI-2,
    beating DeepSeek-R1/o3-mini/Gemini-2.5-Pro at <0.01% of their params (Jolicoeur-Martineau,
    "Less is More", arXiv:2510.04871). Predecessor HRM (27M, 40.3% ARC-AGI-1). OPEN WEIGHTS on HF:
    `wtfmahe/Samsung-TRM`, `SamsungSAILMontreal/TinyRecursiveModels`, `domus-magna/trm-repro`.
    Refinements: TTT-of-TRM (arXiv:2511.02886), Mamba-2 hybrid (arXiv:2602.12078), identity-cond +
    test-time compute (arXiv:2512.11847). *Action:* replace the corpus-frequency `intuition_prior`
    with a TRM-style tiny recursive scorer behind the existing `order_actions` interface, env
    `V21_TRM_PRIOR` (OFF). CAVEAT: TRM is a STATIC-grid solver — for interactive v3 reframe it as a
    per-frame "rank next action / predict frame-delta" scorer, not a puzzle solver. *Done when:* the
    TRM prior beats the frequency prior on held-out action ordering offline. **BLOCKER: GPU
    training/TTT route** (not offline-verifiable in a 4h sandbox); bundle a quantized checkpoint for
    offline inference once trained.
R10. **Mine the now-PUBLIC ARC-AGI-3 reference code** (validates R2/R3/B2, gives real impls to port).
    (a) Executable World Models baseline `github.com/astroseger/arc-3-agents-baseline1` (Rodionov
    2605.05138: coder builds+verifies+MDL-refactors a Python WM, plans through it; 15/25 games, RHAE
    58%) → seeds B2/B3 authoring loop. (b) Graph explorer `github.com/dolphin-in-a-coma/arc-agi-3-just-explore`
    (frame processor + state hashing + frontier mgmt) → seeds R3/#6. (c) Interactive agent
    `github.com/ssppsy/arc-agi-3`. (d) Official API/toolkit `github.com/arcprize/arc-agi`. *Action:*
    port the WM-authoring loop (a) and frontier-managed state-graph (b) into `brain/world_model.py`/
    planner + BFS, pure/offline + env-gated; do NOT vendor network/agent-SDK deps. *Done when:* a
    ported WM-authoring or frontier-dedup step reproduces a solved level offline.
R11. **Tufa Labs "StochasticGoose" — CNN+RL frame-change predictor** (won the ARC-AGI-3 Agent
    Preview at 12.58%, 18 levels: a 4-layer conv over 64×64 frames predicting which actions cause a
    frame change; ARC Prize 2025 report arXiv:2601.10904 + Tufa preview writeup). INTERACTIVE-v3-
    NATIVE and the cleanest attack on **ft09** (blind-ACTION6 reflex): a small conv "will this action
    change the frame?" head is exactly ft09's missing intuition. Ties to R9 (both are the learned
    prior). *Action:* prototype a tiny frame-change-predictor head (numpy inference, bundled weights)
    that reorders/gates ACTION6 candidates, env `V21_FRAMECHANGE_PRIOR` (OFF). *Done when:* it cuts
    scored actions on ft09 vs the blind scan offline. **BLOCKER: GPU training route** (as R9).
R12. **Tufa Labs LADDER + TTRL — recursive variant self-improvement** (arXiv:2503.00735). Generate
    progressively SIMPLER variants of a hard level, solve those first, RL up to the hard one on a
    verifiable reward; TTRL adds test-time RL. This is the named/cited method behind item #4's ls20
    "variant re-root / Go-Explore", and it strengthens `evolve`/consolidation (B7). *Action:* fold a
    LADDER-style variant-decomposition seed into `blitz.blitz_macros` / the evolve loop for ls20
    L5–L6 — solve a shortened/rooted sub-problem, replay its winning prefix as a Go-Explore seed.
    Pure/offline-testable on a mock variant ladder. *Done when:* a variant-seed prefix registers
    progress on ls20 L5 that cold BFS does not.
R13. **Opus-4.8 as a TEACHER/coder on the network-enabled cadence box** (user obs: Opus-4.8 solves
    ls20 levels as-is). The Mac cadence already runs `--allow-network`; add an env-gated
    `V21_LLM_BACKEND=claude` coder behind the existing `llm_backend` interface (external network +
    key ONLY on the Mac) so `runtime_coder`/`evolve` can use a frontier model to actually crack a
    wall, then DISTILL the `verify_solution`-passed plan into the offline corpus. INVARIANT: the
    teacher runs only during Mac *solving*; the committed corpus is a model-free verified action list,
    so the OFFLINE Kaggle guard is untouched (Kaggle never calls the teacher). Also capture Opus's
    ls20 reasoning traces as macros / to distil the intuition prior (feeds R9). *Done when:* an
    Opus-authored verified plan solves ls20 L5 or L6 and lands shortest-gated in the corpus.
    **BLOCKER: Claude/Opus API key + external network on the Mac** (localhost-only today) — user action.
R14. **(Long-horizon integration target) The ensemble the user is imagining: open world model +
    massively-parallel multi-agent + VLM + neuro-symbolic brain-mimicking nets, combined to crack
    ls20 L5/L6.** This is the north star, and much of it already has homes — don't build it as one
    monolith; assemble it from the env-gated pieces: (i) **open world model** = R2/R10/B2 executable
    WM; (ii) **"100 agents thinking in parallel"** = best-of-N / competing-hypothesis WM synthesis
    (R1 belief-entropy gate + R2 best-of-N + B4 hypotheses) — on the Mac this is sequential-batched
    N, not literally 100 processes (memory-bound alongside BFS), but the DISCRIMINATION logic is the
    same: spawn N candidate models, spend scored actions on the most-discriminating move, keep the
    survivor; (iii) **VLM perception** — we deliberately use exact connected-component perception (B1)
    INSTEAD of a VLM because R8 shows ~80% of VLM ARC failures are perception errors; keep symbolic
    perception as the front-end and reserve a VLM only for a fallback describe-the-scene path; (iv)
    **neuro-symbolic / brain-mimicking nets** = the TRM/HRM/StochasticGoose learned priors (R9/R11)
    feeding the symbolic BFS/WM cascade — that IS the neuro-symbolic combo (tiny net for intuition,
    program-synthesis WM for verified planning). *Action:* treat R14 as the assembly spec, not a
    ticket — each cycle ship ONE of R9–R13 env-gated, and the "ensemble" emerges as the cascade
    `blitz → BFS → [TRM/framechange prior orders actions] → best-of-N executable WM (R1 gate) →
    Opus-teacher fallback`. *Done when:* the combined env-gated stack solves ls20 L5 or L6 end-to-end.
    **BLOCKER: aggregate — GPU training route (R9/R11) + Mac API access (R13) + more RAM for real
    best-of-N.** Sequenced behind R9–R13; do not attempt as a single big-bang change.

# =====================================================================
# EPIC C — Teacher/Student ensemble (shared scratchpad).  Design: TEACHER_STUDENT.md
# =====================================================================
# The unifying frame (user steer 2026-07-08): every solver is a TEACHER; the
# intuitive prior is a TODDLER; they teach each other + the toddler via one shared
# scratchpad `brain/blackboard.py` (built + offline-tested, per-game JSON, provenance
# on every lesson). Lets T1/T2/T3 be built IN PARALLEL — they only meet at the
# blackboard. Same invariants: additive, env-gated OFF, verify+shortest-gated,
# offline-covered, corpus never at risk.
C0. **Blackboard substrate.** [DONE this session] `brain/blackboard.py`: action_effects,
    fragments (Go-Explore seeds), dead_ends (negative constraints), cells (novelty archive),
    world_facts; `hints(level)` for students; `consolidate()` sleep-teacher; persisted at
    `brain/blackboard/<gid>.json`. 8 offline checks in `test_blackboard.py`.
    [WIRED 2026-07-08] `cadence_runner.solve_game` now WRITEs every verified win as a fragment +
    per-action effects (`_bb_record_solution`) and READs the blackboard's verified fragments to
    replay-then-`_verify` on still-UNSOLVED walls (`_bb_seed_candidates` — the C0→C1 Go-Explore
    bridge, a sibling/prior lesson cracking a wall), then consolidate+save each pass. Pure helpers
    `_bb_enabled/_bb_open/_bb_record_solution/_bb_seed_candidates`; env `V21_BLACKBOARD` (default
    OFF); verify+shortest-gated; corpus untouched. Also fixed a latent `consolidate()` unhashable-
    dict crash (json-key dedup) so it survives ACTION6 click-plans. +11 offline checks. *Remaining:*
    a Mac cadence with `V21_BLACKBOARD=1` where a fragment cracks a wall ("BLACKBOARD seed solved"),
    then C1 upgrades `ladder.macro_bfs` to a cell-archive Go-Explore reading these seeds.
C1 (=T1). **Go-Explore intuitive search** — upgrade `ladder.macro_bfs` to a cell-archive
    Go-Explore (`blackboard.cell_key` cells + return-to-promising-cell + macro explore), guided
    by the toddler's `action_order`. The ls20 L5 lever (tames the 19k-state blowup). Env `V21_GOEXPLORE`.
    *Done when:* ls20 L5 registers `levels_completed>=6`.
    [CODED + offline-verified 2026-07-08] `ladder.go_explore`: cell archive keyed on a COARSE
    `cell_fn(state)` (real wiring = `blackboard.cell_key` on the status-bar-masked frame), keeps the
    SHORTEST path per cell, returns-to-promising-cell (fewest visits then shortest path, over-visit
    cap), single-step + corridor-sweep macro edges that drop a breadcrumb in every NEW cell (patience
    stagnation stop — coarse cells don't change every step, so it can't use macro_bfs's stop-on-no-
    change). Wrapper `planner.plan_in_model_goexplore`; runner Stage-3.45 `_goexplore_for_solver`
    (steered by the blackboard toddler `action_order`, primed by its verified fragments) wired into
    `solve_game` for UNSOLVED walls, env `V21_GOEXPLORE` (now exported =1 in run_cadence.sh alongside
    the newly-exported `V21_BLACKBOARD=1`, per the R13 "coded-but-never-exported" lesson).
    verify+shortest-gated; corpus untouched. +5 offline checks (solve/action_order/seed/None-unreach).
    *Remaining:* a Mac cadence where GOEXPLORE registers `levels_completed>=6` on ls20 L5; if flat,
    tune `V21_GOEXPLORE_BINS` (cell coarseness) + `V21_PLANNER_STATES`, or seed from the L4 end-state.
C2 (=T2, subsumes B2). **Persistent executable world model** — `brain/wm/<gid>/` verified vs
    recorded transitions, MDL-refactored, reused across runs; teaches world_facts. Env `V21_WORLD_MODEL`.
    *Done when:* a persisted model reproduces a solved level's transitions and seeds a solve next run.
    [SUBSTRATE CODED + offline-verified 2026-07-08, commit 5924dad] `brain/world_model.py` now ships the
    persistence substrate atop the existing verifier: `build_tabular_model(records)` (trusted-by-
    construction executable WM — reproduces every recorded (state,action)->next), `mdl_refactor` (collapse
    to the SHORTEST equivalent rule: identity/constant, Rodionov 2026), pure `predict_from_model`, and
    on-disk `wm_dir`/`save_model`/`load_model` at `brain/wm/<game>/model.json` (atomic, canonical JSON).
    Pure/dependency-free; +5 offline checks (reproduce/None-unseen/MDL-identity/persist-reload-reuse/absent).
    [WIRED 2026-07-08] `cadence_runner.solve_game` now CAPTUREs live one-step transitions on still-UNSOLVED
    walls (`_wm_step_records`: status-bar-masked frames as state), PERSISTs a per-game model at game-end
    (`_wm_persist`: build_tabular_model->mdl_refactor->save_model->`brain/wm/<gid>/model.json`), and each
    pass READs the prior-run model via `_wm_reuse`->`verify_model`, logging the cross-run `is_trusted` REUSE
    signal. Env `V21_WORLD_MODEL` (default OFF; exported =1 in run_cadence.sh per the R13 coded-but-never-
    exported lesson). New `v21/.gitignore` keeps `brain/wm/`+`brain/blackboard/` runtime state out of the
    `git add -A v21/` sweep. +7 offline wiring checks; corpus + offline guard untouched. *Remaining:* a Mac
    cadence to confirm the `WORLD_MODEL saved`/`WORLD_MODEL reuse: trusted=True` lines, then upgrade the
    tabular model to SEED the Go-Explore planner (predict-through-model rollout) so a persisted model
    actively cracks a wall — the tabular substrate only reproduces SEEN states, so the seed step needs the
    MDL/coder RULE-model, not just the table.
C3 (=T3, subsumes B8/R9/R11). **Intuitive brain / toddler** — frame-change/action scorer
    (self-supervised online from action_effects first; StochasticGoose-lite CNN / TRM later) behind
    the fixed `IntuitionPrior.order_actions` interface; guides C1. Env `V21_TODDLER`.
    *Done when:* toddler-guided search solves a wall in fewer states than blind.
    [CODED + offline-verified 2026-07-08] `brain/toddler.py::Toddler` — behind the FIXED
    `order_actions(game, frame)` interface, BLENDS the corpus `IntuitionPrior` with the blackboard's
    ONLINE `action_effects` (win-weighted change-rate, alpha=0.7): unseen → corpus prior, seen → shift
    to observed effectiveness. Frame-AWARE first form: per-coarse-frame effect memory (`cell_key`) so a
    StochasticGoose/TRM net (R9/R11) drops in behind `_effect_score` with NO interface change. Wired
    env-gated `V21_TODDLER` into `_goexplore_for_solver` (Stage-3.45) via pure `_toddler_enabled` /
    `_toddler_order` (degrades to `bb.action_order`/canonical on off/None/failure). Exported
    `V21_TODDLER=1` in run_cadence.sh. +10 offline checks (no-op empty, prior-lead, learned override,
    frame-conditioning, gate, avail restriction). *Remaining:* a Mac cadence with `V21_TODDLER=1` +
    `V21_GOEXPLORE=1` where the toddler's effect-ranked order reaches ls20 L5's frontier in fewer states
    than blind; then C2 persistent world model is the last Epic-C track.
# Build order (lowest-risk first): C0 wiring (teachers write + students read) → C1 Go-Explore
# → C3 toddler distilled from action_effects → C2 persistent WM. C0's read/write makes every
# later track better, so it is the next cycle's priority.

## Stop condition
All 20 levels across the 3 games solved + verified at RHAE 1.0 (or the highest reachable),
and the offline Kaggle notebook reproduces them. Then freeze and submit. Epic B/C have their OWN
success metric — held-out generalisation (a concept from one game solving another) — pursued
in parallel without ever risking the verified corpus.
