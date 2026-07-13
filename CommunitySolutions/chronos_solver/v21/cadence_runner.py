#!/usr/bin/env python3
# =====================================================================
# Chronos v21 — 4-hour cadence runner (the self-improving flywheel)
#
# Every 4 hours this re-attacks the 3 generalization games (ls20, ft09,
# vc33), tightens each level toward the SHORTEST verified plan (RHAE-optimal),
# grows the flywheel corpus + macro bank the Kaggle notebook ships with, and
# writes an append-only scorecard. Implements REQUIREMENTS.md R1/R4/R5/R6.
#
# It references v19/v20 code READ-ONLY (BFSSolver + version-exact resolver);
# all new state lives under v21/. No network calls in this default path.
#
# Usage:
#   python cadence_runner.py                      # one pass, default 3 games
#   python cadence_runner.py --bfs-timeout 600    # deeper pass
#   python cadence_runner.py --games ls20,ft09    # subset
#   python cadence_runner.py --pass-ladder 180,600,1800   # escalate per level
# Schedule: cron "0 */4 * * *"  (see REQUIREMENTS.md R6.2)
# =====================================================================
import argparse, glob, json, logging, os, re, socket, sys, time, fcntl
try:
    import resource as _resource  # stdlib on macOS/Linux; used only for RSS read
except Exception:  # pragma: no cover — non-POSIX; memory guard degrades to inert
    _resource = None
from datetime import datetime, timezone
import blitz  # Stage-0 cheap-win pre-pass (BACKLOG #2); pure, no engine deps

HERE = os.path.dirname(os.path.abspath(__file__))
CHRONOS = os.path.abspath(os.path.join(HERE, ".."))          # .../chronos_solver
REPO = os.path.abspath(os.path.join(CHRONOS, "..", ".."))    # repo root
V19_SRC = os.path.join(CHRONOS, "v19", "src")
V20_SRC = os.path.join(CHRONOS, "v20", "src")
LOGDIR = os.path.join(HERE, "logs")
SOLDIR = os.path.join(HERE, "solutions")                     # v21 corpus (shipped)
MACRO_BANK = os.path.join(HERE, "v21_macro_bank.json")
LOCKFILE = os.path.join(LOGDIR, ".cadence.lock")
os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(SOLDIR, exist_ok=True)

# 3-game generalization set — one per capability tier (see REQUIREMENTS.md R4.1).
DEFAULT_GAMES = ["ls20", "ft09", "vc33"]
TIER = {"ls20": "reasoning/keyboard-maze", "ft09": "reflex/ACTION6-blind",
        "vc33": "orchestration/click"}
# Human baselines for RHAE are the OFFICIAL per-level `baseline_actions` shipped in
# each game's environment_files/<gid>/<ver>/metadata.json (R1.4). Loaded per game,
# version-exact, at run time — no proxy. Falls back to a labeled proxy only if the
# metadata is missing.
PROXY_BASELINE = {"ls20": 350, "ft09": 8, "vc33": 20}   # only if metadata absent

GAME_ROOTS = [os.path.join(REPO, "arc-prize-2026-arc-agi-3", "environment_files"),
              os.path.join(REPO, "environment_files")]

logger = logging.getLogger("v21.cadence")


# ---- offline guardrail (R3.1): forbid network in the default path -------------
class _NoNetwork(socket.socket):
    def connect(self, *a, **k):
        raise RuntimeError("v21 offline guardrail: network access is forbidden "
                           "in the cadence/submission path (see REQUIREMENTS R3.1)")


def _install_offline_guard(enabled=True):
    if enabled:
        socket.socket = _NoNetwork  # type: ignore


# ---- version-exact source resolution (R3.4) -----------------------------------
def resolve_source(gid):
    """Return (path, class_name, version_hash) for the NEWEST version dir of gid.
    Stale hashes are different puzzles and poison the verifier."""
    cands = []
    for base in GAME_ROOTS:
        cands += glob.glob(os.path.join(base, gid, "*", f"{gid}.py"))
    if not cands:
        return None
    # newest by mtime of the version dir
    path = max(cands, key=lambda p: os.path.getmtime(os.path.dirname(p)))
    ver = os.path.basename(os.path.dirname(path))
    src = open(path).read()
    m = re.search(r"class\s+(\w+)\s*\(\s*ARCBaseGame", src)
    cls = m.group(1) if m else (gid[0].upper() + gid[1:])
    return path, cls, ver


def load_baselines(source_path, gid):
    """Official per-level human baselines from the version-exact metadata.json
    (the RHAE denominators). Returns (list_or_None, src_label)."""
    meta = os.path.join(os.path.dirname(source_path), "metadata.json")
    if os.path.exists(meta):
        try:
            ba = json.load(open(meta)).get("baseline_actions")
            if isinstance(ba, list) and ba:
                return [int(x) for x in ba], "official"
        except Exception:
            pass
    return None, "proxy"


def baseline_for(baselines, src, gid, lvl):
    if baselines and 0 <= lvl < len(baselines):
        return baselines[lvl], src
    return PROXY_BASELINE.get(gid), "proxy"


# ---- corpus I/O: append/replace-shorter only (R5.1, R1.2) ---------------------
def load_corpus(gid):
    p = os.path.join(SOLDIR, f"{gid}.json")
    if os.path.exists(p):
        try:
            raw = json.load(open(p))
            return {int(k): v for k, v in raw.items() if str(k).lstrip("-").isdigit()}
        except Exception:
            return {}
    return {}


def save_corpus(gid, corpus):
    p = os.path.join(SOLDIR, f"{gid}.json")
    json.dump({str(k): v for k, v in sorted(corpus.items())}, open(p, "w"))


def rhae_level(human, ai_actions):
    if not ai_actions or ai_actions <= 0:
        return 0.0
    return min(1.0, (human / ai_actions)) ** 2


# ---- macro harvest (R5.2): distill transferable macros ------------------------
def harvest_macros(gid, corpus, bank):
    """Extract the action-only skeleton of each solution as a candidate macro,
    keyed by (game, level, length). Retrieval logic lives in the agent (R2.3)."""
    for lvl, sol in corpus.items():
        acts = [a for a, _d in sol]
        key = f"{gid}:L{lvl}:len{len(acts)}"
        bank[key] = {"game": gid, "level": lvl, "actions": acts, "len": len(acts)}
    return bank


# ---- solve one game (uses v19 BFSSolver read-only) ----------------------------
def _should_resolve(already_solved, env=None):
    """Whether to run a fresh (expensive) BFS re-search on this level.

    Unsolved (wall) levels ALWAYS get a BFS pass. Levels already solved+verified
    from the corpus are skipped by default — they're at their measured RHAE and a
    fresh BFS cannot improve them, so re-solving only steals the per-level budget
    from the real walls. Set V21_RESOLVE_SOLVED=1 to re-enable the optimality hunt
    on solved levels (e.g. to shorten a sub-1.0 solve). Pure/offline-testable."""
    env = os.environ if env is None else env
    if not already_solved:
        return True
    return env.get("V21_RESOLVE_SOLVED", "0") in ("1", "true", "True")


def _wall_reachable(level_idx, corpus, solutions=None):
    """Whether this wall's start state can be RE-ROOTED for verify/replay.

    A level is re-rooted by replaying the verified plans of every prior level; if
    any earlier level is still unsolved, `_make_start_state` returns None and the
    engine reports "could not re-root level N to replay". That happened for every
    wall behind the frontier on run 073852Z (ls20 L6, ft09 L3–L5, vc33 L5–L6) — and
    for each of those levels the OPUS_TEACHER still burned 2 cloud rounds proposing
    a plan that could NEVER be verified. This pure predicate lets the caller skip
    the paid cloud stages (teacher / world-model) on unreachable walls and spend the
    whole Opus budget on the one reachable frontier wall per game.

    Reachable iff every prior level 0..level_idx-1 has a plan in `corpus` (this run's
    accumulating solutions) or the solver's live `solutions` chain. Fail-OPEN: L0 is
    always reachable and a missing/odd container never wrongly gates a wall. Pure —
    no engine import — so test_offline covers it."""
    if level_idx <= 0:
        return True
    corpus = corpus or {}
    solutions = solutions or {}
    for i in range(level_idx):
        if not (corpus.get(i) or solutions.get(i)):
            return False
    return True


def solve_game(gid, bfs_timeout, BFSSolver):
    """Escalating single-pass solve: for each level, keep the SHORTEST verified
    plan (existing corpus vs a fresh BFS). Returns (rows, new_corpus, improved)."""
    info = resolve_source(gid)
    if not info:
        logger.warning("[%s] no source found — skipping (black-box handled by agent)", gid)
        return [], load_corpus(gid), False
    path, cls, ver = info
    baselines, bsrc = load_baselines(path, gid)
    logger.info("[%s] source=%s class=%s ver=%s baselines=%s(%s)",
                gid, os.path.relpath(path, REPO), cls, ver, baselines, bsrc)

    corpus = load_corpus(gid)
    rows, improved = [], False
    try:
        solver = BFSSolver(path, cls, bfs_timeout=bfs_timeout,
                           workers=int(os.environ.get("V21_BFS_WORKERS", "1")))
        if not solver.load():                     # REQUIRED: builds game_cls from source
            logger.error("[%s] BFSSolver.load() failed — cannot solve", gid)
            return [], corpus, False
    except Exception as e:
        logger.error("[%s] BFSSolver init failed: %s", gid, e)
        return [], corpus, False

    # discover level count: prefer official baseline length, then solver, then corpus
    n_levels = (len(baselines) if baselines else None) \
        or getattr(solver, "num_levels", None) or (max(corpus, default=-1) + 1) or 6

    # Epic C0: shared scratchpad for this game (None unless V21_BLACKBOARD=1).
    bb = _bb_open(gid)

    # Epic C2: persistent executable world model (brain/wm/<gid>/). None unless
    # V21_WORLD_MODEL=1. wm_records accumulates this run's captured transitions;
    # a model persisted on a PRIOR run is verified against them for the reuse signal.
    wm_gd = _wm_game_dir(gid) if _wm_enabled() else None
    wm_records = []

    for lvl in range(n_levels):
        prev = corpus.get(lvl)
        prev_len = len(prev) if prev else None
        # verify the existing plan still holds on this engine version (R5.3)
        if prev and _verify(solver, lvl, prev):
            best, best_len = prev, prev_len
            already_solved = True
        else:
            best, best_len = None, None
            already_solved = False
        # Epic C2 CAPTURE+REUSE: on still-UNSOLVED walls, record live one-step
        # transitions and (if a model was persisted on a prior run) verify it still
        # reproduces them — the cross-run reuse signal. Only for walls so solved
        # corpus levels add ZERO cost; fully guarded; env V21_WORLD_MODEL (OFF).
        if wm_gd is not None and not already_solved:
            try:
                recs = _wm_step_records(solver, lvl)
            except Exception as e:
                recs = []
                logger.debug("[%s L%d] world_model capture error: %s", gid, lvl, e)
            if recs:
                wm_records.extend(recs)
                rep = _wm_reuse(wm_gd, recs)
                if rep is not None:
                    from brain.world_model import is_trusted
                    logger.info("[%s L%d] WORLD_MODEL reuse: trusted=%s acc=%.2f (n=%d)",
                                gid, lvl, is_trusted(rep),
                                rep.get("accuracy", 0.0), rep.get("n_total", 0))
        # Stage-0 blitz pre-pass (BACKLOG #2): only for UNSOLVED (wall) levels —
        # solved levels already have a verified corpus plan, so this adds ZERO
        # cost there. Cheap depth-1 / repeat-K wins crack reflex/orchestration
        # walls that BFS times out on. Fully guarded; any error falls through to
        # BFS. The candidate is still verified + shortest-gated below.
        if best is None and os.environ.get("V21_BLITZ", "1") not in ("0", "false", "False"):
            _bstats = {}
            try:
                bsol = blitz.blitz_for_solver(
                    solver, lvl, repeat_K=int(os.environ.get("V21_BLITZ_K", "200")),
                    stats=_bstats)
            except Exception as e:
                bsol = None
                logger.debug("[%s L%d] blitz error: %s", gid, lvl, e)
            if bsol and _verify(solver, lvl, bsol):
                best, best_len, improved = bsol, len(bsol), True
                corpus[lvl] = bsol
                logger.info("[%s L%d] BLITZ solved in %d actions", gid, lvl, len(bsol))
            else:
                logger.info(_local_stage_note(
                    "BLITZ", gid, lvl, bsol, extra=blitz.blitz_breadth_note(_bstats)))
        # Epic C0 READ (Go-Explore seed replay): for still-UNSOLVED walls, replay
        # the blackboard's verified fragments (from sibling levels / prior runs)
        # and keep the first that VERIFIES on this wall. Cheap (replay, no search),
        # verify + shortest-gated, env V21_BLACKBOARD (default OFF). This is the
        # C0->C1 bridge: a lesson taught on one level cracks another.
        if best is None and bb is not None:
            for seed in _bb_seed_candidates(bb, lvl):
                try:
                    ok_seed = _verify(solver, lvl, seed)
                except Exception:
                    ok_seed = False
                if ok_seed:
                    best, best_len, improved = seed, len(seed), True
                    corpus[lvl] = seed
                    logger.info("[%s L%d] BLACKBOARD seed solved in %d actions",
                                gid, lvl, len(seed))
                    break
        # attempt a fresh (optimal-preferring) solve — BUT skip the expensive
        # re-search on levels already solved+verified from the corpus. Those are
        # already at their measured RHAE (every current ls20/ft09/vc33 solved level
        # is at 1.0, which BFS cannot beat) so re-deriving them each run just burns
        # the per-level BFS wall-clock — e.g. ls20 L0–L4 consumed ~1686s of the 1200s
        # budget in run 20260707T220311Z, leaving ~0 for the real wall L5. Reserving
        # the budget lets the FIRST unsolved wall get a full BFS pass. Re-enable the
        # optimality hunt on solved levels with V21_RESOLVE_SOLVED=1 (e.g. to shorten
        # a sub-1.0 solve for the R4 quadratic-RHAE gain). Corpus is untouched: the
        # verified plan is still `best`, and unsolved levels always run BFS.
        if not _should_resolve(already_solved):
            sol = None
            logger.info("[%s L%d] corpus-verified (%s actions) — skip re-BFS, "
                        "budget reserved for walls", gid, lvl, best_len)
        else:
            try:
                sol = solver.solve_level(lvl)  # v19 'auto' ladder, shortest-first
            except Exception as e:
                sol = None
                logger.debug("[%s L%d] solve error: %s", gid, lvl, e)
        if sol and _verify(solver, lvl, sol) and (best_len is None or len(sol) < best_len):
            best, best_len, improved = sol, len(sol), True
            corpus[lvl] = sol
        # Stage-3.4 BRAIN PLANNER (Epic B3): Go-Explore/macro-BFS over the engine
        # (the trusted white-box model) from this level's re-rooted start — collapses
        # ls20's long corridors that plain BFS can't reach in budget. Only for still-
        # UNSOLVED walls; env-gated V21_BRAIN_PLANNER (default OFF); verified+shortest-
        # gated below. This is the ls20 L5–L6 frontier.
        if best is None and os.environ.get("V21_BRAIN_PLANNER", "0") in ("1", "true", "True"):
            try:
                psol = _brain_planner_for_solver(solver, lvl, gid)
            except Exception as e:
                psol = None
                logger.debug("[%s L%d] brain planner error: %s", gid, lvl, e)
            if psol and _verify(solver, lvl, psol):
                best, best_len, improved = psol, len(psol), True
                corpus[lvl] = psol
                logger.info("[%s L%d] BRAIN_PLANNER solved in %d actions", gid, lvl, len(psol))
            else:
                logger.info(_local_stage_note("BRAIN_PLANNER", gid, lvl, psol))
        # Stage-3.45 GO-EXPLORE (Epic C1): cell-archive Go-Explore over the engine —
        # dedups on a COARSE downsampled-frame cell instead of the exact frame hash,
        # so ls20 L5's corridors merge into a small return-to archive rather than a
        # 19k-state frontier. Steered by the blackboard's toddler action_order and
        # primed by its verified fragments when V21_BLACKBOARD is on. Only for still-
        # UNSOLVED walls; env-gated V21_GOEXPLORE (default OFF); verified+shortest-
        # gated below. The ls20 L5–L6 lever, complementary to the Stage-3.4 planner.
        if best is None and os.environ.get("V21_GOEXPLORE", "0") in ("1", "true", "True"):
            try:
                gsol = _goexplore_for_solver(solver, lvl, bb, gid)
            except Exception as e:
                gsol = None
                logger.debug("[%s L%d] go-explore error: %s", gid, lvl, e)
            if gsol and _verify(solver, lvl, gsol):
                best, best_len, improved = gsol, len(gsol), True
                corpus[lvl] = gsol
                logger.info("[%s L%d] GOEXPLORE solved in %d actions", gid, lvl, len(gsol))
            else:
                logger.info(_local_stage_note("GOEXPLORE", gid, lvl, gsol))
        # Neural toddler harvest (Epic C3 / R11): on an UNSOLVED wall, probe each
        # action ONCE from the re-rooted start and log (frame, action -> changed/won)
        # samples for the StochasticGoose-style frame-change CNN. Trained later in
        # consolidation on the Mac GPU (MPS). Env-gated V21_TODDLER_NET; fully guarded.
        if best is None and os.environ.get("V21_TODDLER_NET", "0") in ("1", "true", "True"):
            try:
                _harvest_toddler_samples(solver, lvl, gid)
            except Exception as e:
                logger.debug("[%s L%d] toddler harvest error: %s", gid, lvl, e)
        # Stage-3.5 runtime code-writer (BACKLOG #3): last resort — only for still
        # UNSOLVED wall levels after blitz + BFS both fail. The local Qwen writes a
        # WorldModel from the observed transitions and proposes shortest-first plans;
        # any winner is still verified + shortest-gated below. OFF by default (loads
        # a local model / adds wall-clock); opt-in via env V21_RUNTIME_CODER=1.
        if best is None and os.environ.get("V21_RUNTIME_CODER", "0") in ("1", "true", "True"):
            # C1+++++ memory guard: skip the memory-heavy ollama coder when the
            # process already holds a large resident set (post-big-BFS), so we
            # don't repeat run 164123Z's OOM `Killed: 9` that ended the whole
            # pass mid-sweep and stranded the lock. Off unless the ceiling env is
            # set. Wall stays UNSOLVED either way; the sweep survives to exit=0.
            _rc_rss = _process_rss_mb()
            if _coder_mem_skip(_rc_rss, os.environ.get("V21_RUNTIME_CODER_MAX_RSS_MB", "0")):
                logger.info("[%s L%d] RUNTIME_CODER skipped: RSS %.0fMB >= ceiling "
                            "%sMB (OOM guard) — leaving wall UNSOLVED, sweep continues",
                            gid, lvl, _rc_rss or 0.0,
                            os.environ.get("V21_RUNTIME_CODER_MAX_RSS_MB"))
                llm = None
            else:
                llm = _get_runtime_llm()
            if llm is not None:
                try:
                    _rc_budget = float(os.environ.get("V21_RUNTIME_CODER_BUDGET", "300"))
                    csol = _call_with_deadline(
                        lambda: _runtime_coder_for_solver(
                            solver, lvl, llm,
                            max_len=int(os.environ.get("V21_RUNTIME_MAXLEN", "200"))),
                        _rc_budget)
                except TimeoutError as e:
                    csol = None
                    logger.info("[%s L%d] RUNTIME_CODER abandoned (%s) — moving to next wall",
                                gid, lvl, e)
                except Exception as e:
                    csol = None
                    logger.debug("[%s L%d] runtime_coder error: %s", gid, lvl, e)
                if csol and _verify(solver, lvl, csol):
                    best, best_len, improved = csol, len(csol), True
                    corpus[lvl] = csol
                    logger.info("[%s L%d] RUNTIME_CODER solved in %d actions",
                                gid, lvl, len(csol))
                else:
                    logger.info(_local_stage_note("RUNTIME_CODER", gid, lvl, csol))
        # Stage-3.6 OPUS TEACHER (R13): the final teacher — when everything local
        # fails a wall, ask cloud Opus to read the WHITE-BOX source and construct the
        # winning sequence. Its plan is UNVERIFIED → still verify + shortest-gate +
        # exploit-refusal below. Env-gated V21_OPUS_TEACHER (needs ANTHROPIC_API_KEY).
        # Frontier gate: a wall behind an unsolved earlier wall CANNOT be re-rooted,
        # so the engine can never verify a plan for it — every paid Opus round on such
        # a level is wasted ("could not re-root level N to replay"). Only spend the
        # cloud teacher / world-model budget on re-rootable (frontier) walls; deeper
        # walls unlock automatically once the frontier one is solved this run.
        _reroot_ok = _wall_reachable(lvl, corpus, getattr(solver, "solutions", None))
        if best is None and not _reroot_ok and (
                os.environ.get("V21_OPUS_TEACHER", "0") in ("1", "true", "True")
                or os.environ.get("V21_OPUS_WM", "0") in ("1", "true", "True")):
            logger.info("[%s L%d] wall gated behind an unsolved earlier level — "
                        "skipping cloud teacher/WM (cannot re-root to verify)", gid, lvl)
        if best is None and _reroot_ok and os.environ.get("V21_OPUS_TEACHER", "0") in ("1", "true", "True"):
            try:
                tsol = _opus_teacher_for_solver(solver, lvl, gid)
            except Exception as e:
                tsol = None
                logger.debug("[%s L%d] opus teacher error: %s", gid, lvl, e)
            if tsol and _verify(solver, lvl, tsol):
                best, best_len, improved = tsol, len(tsol), True
                corpus[lvl] = tsol
                logger.info("[%s L%d] OPUS_TEACHER solved in %d actions", gid, lvl, len(tsol))
        # Stage-3.7 OPUS WORLD MODEL (B2): Opus WRITES an executable WorldModel .py
        # from the white-box source; we exec it, plan in it, verify on the engine, and
        # persist it to brain/wm/<gid>/model.py. The generalization spine — more general
        # than a one-off plan. Env-gated V21_OPUS_WM (needs ANTHROPIC_API_KEY).
        if best is None and _reroot_ok and os.environ.get("V21_OPUS_WM", "0") in ("1", "true", "True"):
            try:
                wsol = _opus_world_model_for_solver(solver, lvl, gid)
            except Exception as e:
                wsol = None
                logger.debug("[%s L%d] opus world-model error: %s", gid, lvl, e)
            if wsol and _verify(solver, lvl, wsol):
                best, best_len, improved = wsol, len(wsol), True
                corpus[lvl] = wsol
                logger.info("[%s L%d] OPUS_WM solved in %d actions", gid, lvl, len(wsol))
        if best is None:
            logger.info("[%s L%d] UNSOLVED at budget %ss", gid, lvl, bfs_timeout)
            continue
        # Epic C0 WRITE: teach this verified win to the shared scratchpad so later
        # levels / next runs can replay it as a Go-Explore seed (see READ above).
        _bb_record_solution(bb, lvl, best, source="cadence")
        solver.solutions[lvl] = best        # chain: later levels verify from here (R5.3)
        hb, hbsrc = baseline_for(baselines, bsrc, gid, lvl)
        r = rhae_level(hb if hb else best_len, best_len)
        rows.append({"game": gid, "tier": TIER.get(gid, "?"), "level": lvl,
                     "actions": best_len, "rhae": round(r, 4),
                     "baseline": hb, "baseline_src": hbsrc})
        flag = " <-- OVER BASELINE" if hb and best_len > hb else ""
        logger.info("[%s L%d] actions=%s baseline=%s rhae=%.3f%s",
                    gid, lvl, best_len, hb, r, flag)
    # Epic C0: persist the lessons taught this pass (bounded via consolidate).
    if bb is not None:
        try:
            bb.consolidate().save()
        except Exception as e:
            logger.debug("[%s] blackboard save failed: %s", gid, e)
    # Epic C2: persist this run's captured transitions as a per-game executable
    # world model (build -> MDL-refactor -> save) so the NEXT run can load+verify
    # (reuse) it. Guarded; only writes brain/wm/<gid>/ runtime state.
    if wm_gd is not None and wm_records:
        model = _wm_persist(wm_gd, wm_records)
        if model is not None:
            logger.info("[%s] WORLD_MODEL saved kind=%s n=%s -> %s",
                        gid, model.get("kind"), model.get("n"),
                        os.path.relpath(wm_gd, HERE))
    return rows, corpus, improved


def _verify(solver, lvl, sol):
    try:
        return bool(solver.verify_solution(lvl, sol))
    except Exception:
        return False


# ---- Epic C0: shared blackboard read/write wiring ----------------------------
# The proven cascade (blitz -> BFS -> planner -> coder) is the TEACHER; the
# blackboard (brain/blackboard.py) is the shared scratchpad the teachers write
# lessons to and students read seeds from. All of this is env-gated V21_BLACKBOARD
# (default OFF) so the verified corpus and offline guard are never at risk: reads
# only ADD verify+shortest-gated candidates, writes only append lessons to a
# per-game JSON that no committed solution depends on. The helpers below are PURE
# (no engine import) so test_offline can exercise them; the engine replay of a
# seed lives in solve_game behind _verify.
def _bb_enabled(env=None):
    env = os.environ if env is None else env
    return env.get("V21_BLACKBOARD", "0") in ("1", "true", "True")


def _bb_open(gid, env=None):
    """Return a Blackboard for this game if V21_BLACKBOARD is on, else None.
    Guarded: any import/load failure degrades to None (cascade unaffected)."""
    if not _bb_enabled(env):
        return None
    try:
        from brain.blackboard import Blackboard
        return Blackboard(gid)
    except Exception as e:
        logger.debug("[%s] blackboard open failed: %s", gid, e)
        return None


def _bb_record_solution(bb, level, plan, source="cadence"):
    """WRITE side: a verified winning plan becomes a Go-Explore fragment (seed for
    similar/sibling levels) plus per-action effects (every action on a winning path
    counts as `changed`; the terminal action counts as `won`). Pure — no engine."""
    if bb is None or not plan:
        return bb
    try:
        bb.teach_fragment(plan, level=level, reached=level + 1, source=source)
        last = len(plan) - 1
        for i, step in enumerate(plan):
            action = step[0] if isinstance(step, (list, tuple)) else step
            bb.teach_action_effect(action, changed=True, won=(i == last), source=source)
    except Exception as e:
        logger.debug("blackboard write failed (L%s): %s", level, e)
    return bb


def _bb_seed_candidates(bb, level):
    """READ side (candidate ordering): verified fragments to REPLAY-then-verify on
    the wall first, shortest-first, same-level preferred. Pure — the actual replay
    is `_verify(solver, level, seed)` in solve_game. Empty list if bb is None."""
    if bb is None:
        return []
    try:
        return bb.seed_plans(level)
    except Exception:
        return []


# --- R7(a) workspace counterexamples (DREAMTEAM arXiv:2605.09650) -------------
# "counterexamples == losses": a wall plan that FAILED verify is negative-constraint
# evidence for the NEXT run. Today the Opus teacher's R7 teach-with-feedback loop
# is WITHIN a single run (acc_notes is in-memory) — every fresh Mac cadence starts
# blank and can re-propose the exact plan that already reached levels_completed=5
# last run (observed run 073852Z: ls20 L5 teacher rounds 1&2 both stalled at 5/6).
# These pure helpers PERSIST failed teacher plans to the per-game blackboard's
# dead_ends and feed them back as a "do NOT repeat" note on the next run. Env-gated
# V21_WORKSPACE_COUNTEREX (default OFF), independent of the full V21_BLACKBOARD
# seeding path; every op degrades to a no-op so the teacher path is never broken.
def _counterex_enabled(env=None):
    env = os.environ if env is None else env
    return env.get("V21_WORKSPACE_COUNTEREX", "0") in ("1", "true", "True")


def _counterex_open(gid, env=None):
    """Open the per-game blackboard purely to persist/read teacher counterexamples,
    independent of V21_BLACKBOARD. None when gated off or on any failure."""
    if not _counterex_enabled(env):
        return None
    try:
        from brain.blackboard import Blackboard
        return Blackboard(gid)
    except Exception as e:
        logger.debug("[%s] counterex open failed: %s", gid, e)
        return None


def _counterex_notes(bb, level, max_shown=6, max_len=900):
    """READ side: a compact 'do NOT repeat these action sequences' note built from
    the blackboard's persisted dead_ends, so a next-run teacher avoids a plan that
    already failed verify. Pure — no engine. Empty string when bb is None / empty."""
    if bb is None:
        return ""
    try:
        prefixes = bb.avoid_prefixes()
    except Exception:
        return ""
    lines = []
    for p in prefixes[-max_shown:]:
        acts = ",".join(str(s[0]) for s in p if s)
        if acts:
            lines.append("[%d actions: %s]" % (len(p), acts))
    if not lines:
        return ""
    note = ("Previously-FAILED plans on this wall (do NOT repeat these action "
            "sequences): " + "; ".join(lines))
    return note[:max_len]


def _counterex_record(bb, level, plan, source="opus"):
    """WRITE side: persist a failed teacher plan as a dead_end and save to disk so
    the lesson survives to the next cadence. Pure — no engine. No-op on None/empty."""
    if bb is None or not plan:
        return bb
    try:
        bb.teach_dead_end(plan, source=source)
        bb.consolidate()
        bb.save()
    except Exception as e:
        logger.debug("counterex record failed (L%s): %s", level, e)
    return bb


def _toddler_enabled(env=None):
    env = os.environ if env is None else env
    return env.get("V21_TODDLER", "0") in ("1", "true", "True")


def _toddler_order(bb, gid, level, avail, frame=None, env=None):
    """C3 toddler action ordering for the search callers (Go-Explore Stage-3.45).
    When V21_TODDLER is on, blend the blackboard's ONLINE action_effects with the
    shipped corpus `IntuitionPrior` behind the fixed `order_actions` interface and
    return the candidate actions in `avail` best-first. Degrades to None (caller
    keeps its existing `bb.action_order`/canonical order) when the flag is off,
    the blackboard is absent, or anything fails — never invents actions, never
    raises. Pure enough to unit-test without the engine (pass `frame=None`)."""
    if not _toddler_enabled(env) or bb is None:
        return None
    # Neural toddler (R11) first, when V21_TODDLER_NET is on AND a trained net
    # exists for this game — a frame-change CNN gives frame-aware ordering the
    # symbolic prior can't. Falls through to the symbolic Toddler on any miss.
    _envv = os.environ if env is None else env
    if _envv.get("V21_TODDLER_NET", "0") in ("1", "true", "True"):
        try:
            from brain.toddler_net import ToddlerNet
            net = ToddlerNet(gid)
            if net.load():
                order = net.order_actions(frame=frame, game=gid)
                order = [a for a in order if a in avail]
                if order:
                    return order
        except Exception as e:
            logger.debug("[%s] neural toddler order failed: %s", gid, e)
    try:
        from brain.toddler import Toddler
        prior = None
        try:
            import intuition
            prior = intuition.IntuitionPrior(os.path.join(HERE, "intuition_prior.json"))
        except Exception:
            prior = None
        order = Toddler(blackboard=bb, prior=prior).order_actions(
            game=gid, frame=frame, actions=list(avail))
        return [a for a in order if a in avail] or None
    except Exception as e:
        logger.debug("[%s] toddler order failed: %s", gid, e)
        return None


# ---- Epic C2 (=T2): persistent executable world model wiring -----------------
# The `brain/world_model.py` substrate (build_tabular_model -> mdl_refactor ->
# save/load, verify_model/is_trusted) is proven offline; this wires it live.
# WRITE: on still-UNSOLVED walls, capture live one-step transitions and persist a
# per-game executable world model at brain/wm/<gid>/model.json. READ (next run):
# load that model and verify it still reproduces freshly-captured transitions —
# is_trusted(report) is the C2 cross-run REUSE signal ("the model transferred").
# All env-gated V21_WORLD_MODEL (default OFF); the verified corpus and offline
# guard are untouched — this only reads/writes brain/wm state no committed
# solution depends on. The pure helpers are engine-free (test_offline exercises
# them); only _wm_step_records touches the engine and is fully guarded.
def _wm_enabled(env=None):
    env = os.environ if env is None else env
    return env.get("V21_WORLD_MODEL", "0") in ("1", "true", "True")


def _wm_game_dir(gid):
    """On-disk dir for this game's persisted world model: brain/wm/<gid>/."""
    from brain.world_model import wm_dir
    return wm_dir(HERE, gid)


def _wm_persist(game_dir, records):
    """PURE: build a trusted-by-construction tabular model from recorded
    (prev, action, next) transitions, MDL-refactor it toward a shorter equivalent
    rule, and save it to <game_dir>/model.json. Returns the saved model dict, or
    None on empty records / failure. Engine-free (json/os only)."""
    if not records:
        return None
    try:
        from brain.world_model import build_tabular_model, mdl_refactor, save_model
        model = mdl_refactor(build_tabular_model(records))
        save_model(game_dir, model)
        return model
    except Exception as e:
        logger.debug("world_model persist failed: %s", e)
        return None


def _wm_reuse(game_dir, records):
    """PURE: load a previously-persisted model and verify it still reproduces the
    freshly-captured `records` (the cross-run reuse check). Returns a verify report
    dict, or None if no model is saved yet / on failure. is_trusted(report) is the
    'model transferred across runs' signal the loop logs. Engine-free."""
    if not records:
        return None
    try:
        from brain.world_model import load_model, verify_model, predict_from_model
        model = load_model(game_dir)
        if model is None:
            return None
        return verify_model(lambda p, a: predict_from_model(model, p, a), records)
    except Exception as e:
        logger.debug("world_model reuse failed: %s", e)
        return None


def _wm_step_records(solver, level_idx):
    """Capture LIVE one-step transitions from this level's TRUE start: replay each
    simple action once on a fresh fork and record (masked-frame, action_id,
    masked-next-frame). The status-bar-masked frame (top/bottom 2 rows zeroed,
    canonical nested lists) is the model's state, so records are deterministic
    across runs (same engine + same start) — build+save this run, load+verify next
    run. Cheap (<=5 forked actions). Engine imports are lazy (Mac-only). Returns a
    list of (prev, action, next) triples, or [] on any failure."""
    import numpy as np
    from combined_agent import ActionInput, GameAction

    res = None
    try:
        res = solver._make_start_state(level_idx)
    except Exception:
        res = None
    if res is None:
        return []
    game, f0 = res

    def _mask(f):
        fm = np.asarray(f).copy()
        if fm.ndim == 2 and fm.shape[0] > 4:            # mask status bars (top/bottom 2 rows)
            fm[:2] = 0; fm[-2:] = 0
        return fm.tolist()

    prev = _mask(f0)
    avail = [a for a in (getattr(game, "_available_actions", []) or []) if a <= 5] \
        or [1, 2, 3, 4, 5]
    recs = []
    for aid in avail:
        try:
            g = solver._restore(solver._snap(game))
            r = g.perform_action(ActionInput(id=GameAction.from_id(aid)), raw=True)
            nf = np.array(r.frame[-1]) if getattr(r, "frame", None) else None
            if nf is not None:
                recs.append((prev, aid, _mask(nf)))
        except Exception:
            continue
    return recs


# ---- LOCAL-lever observability (BACKLOG P1 R16 next-cycle note) ---------------
# The local wall-cracking stages (blitz / brain_planner / go-explore /
# runtime_coder) log ONLY on success today, so a wall that every local lever
# silently failed shows nothing between the BFS timeout and the "UNSOLVED" line —
# we cannot tell which levers actually FIRED or which produced a near-miss to
# deepen next cycle. This pure helper builds the one-line INFO note each stage
# emits when it ran but did NOT commit a verified win. Verified wins keep their
# own "STAGE solved in N actions" log; this covers the (ran, no win) case.
def _local_stage_note(stage, gid, lvl, candidate, extra=None):
    """Return a one-line observability string for a LOCAL wall stage that fired
    but did not commit a verified win. Pure (no I/O); caller logs it.
    - candidate is None            -> stage ran, produced no plan
    - candidate present (unverified/failed the verify+shortest gate) -> report its length
    - extra (optional): a compact ' | key=val ...' breadth suffix (e.g. blitz's
      macros/simple/clicks/tier) so a miss reveals WHY, not just that it happened.
    """
    stage = str(stage)
    if not candidate:
        base = "[%s L%d] %s fired: no candidate" % (gid, lvl, stage)
    else:
        base = ("[%s L%d] %s fired: candidate len=%d failed verify/shortest gate"
                % (gid, lvl, stage, len(candidate)))
    if extra:
        base += " | " + str(extra)
    return base


# ---- Stage-3.5 runtime code-writer (BACKLOG #3) -------------------------------
_RUNTIME_LLM = None  # cached across levels/games so we load the local model once


def _call_with_deadline(fn, deadline):
    """Run fn() under a HARD wall-clock deadline (seconds) in a daemon watchdog
    thread; return its result, or raise TimeoutError if it doesn't finish in time.
    The abandoned call keeps running in its detached daemon thread (doing I/O or
    engine C-work with the GIL mostly released) but can never block the cadence
    sweep. Mirrors OllamaBackend.complete's watchdog, one level up: a single hung
    wall STAGE (swapping local model, wedged fork replay) must not consume the
    whole ~2h cadence and starve the other games' walls (run 164123Z: ls20 L5's
    RUNTIME_CODER went silent 66 min — normally ~1 min — so ft09 L2-L5 and vc33
    L4-L6 were never attempted that sweep). deadline<=0 runs inline (legacy)."""
    if not deadline or deadline <= 0:
        return fn()
    import threading
    box = {}

    def _work():
        try:
            box["r"] = fn()
        except BaseException as e:  # noqa: BLE001 — surfaced to the caller thread
            box["e"] = e

    t = threading.Thread(target=_work, name="v21-stage-deadline", daemon=True)
    t.start()
    t.join(deadline)
    if t.is_alive():
        raise TimeoutError("stage exceeded hard deadline %.0fs" % float(deadline))
    if "e" in box:
        raise box["e"]
    return box.get("r")


def _process_rss_mb():
    """Current process resident-set size in MB, or None if unavailable.
    `ru_maxrss` is BYTES on macOS/BSD but KILOBYTES on Linux — normalize by
    platform so the same ceiling means the same thing on the Mac (where the
    cadence runs) and in the Linux offline sandbox."""
    if _resource is None:
        return None
    try:
        rss = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None
    if rss <= 0:
        return None
    # macOS/BSD report bytes; Linux reports KiB.
    return float(rss) / (1024.0 * 1024.0) if sys.platform == "darwin" else float(rss) / 1024.0


def _coder_mem_skip(rss_mb, ceiling_mb):
    """Pure predicate: should the memory-heavy RUNTIME_CODER (ollama) stage be
    SKIPPED because the process already holds a large resident set?

    Motivated by run 164123Z: ls20 L5's coder stage was `Killed: 9` (SIGKILL /
    OOM) right after a 59k-state BFS left ~20k unique frames resident — loading
    the local code model on top tipped the Mac into an OOM kill that ended the
    WHOLE pass mid-sweep, stranding `.cadence.lock` and stalling the runner ~13h.
    C1++'s `_call_with_deadline` guards WALL-CLOCK, not MEMORY, so it can't catch
    this. Skipping the coder on a wall it already couldn't solve keeps that wall
    UNSOLVED either way (no regression) but lets the pass finish cleanly (exit=0,
    lock released) and lets the remaining games' walls get their turn.

    OFF by default: ceiling_mb<=0 => never skip (preserves current behavior).
    Opt-in via env V21_RUNTIME_CODER_MAX_RSS_MB=<n> (suggest ~40% of the Mac's
    RAM, e.g. 6500 on a 16GB M1 Pro)."""
    try:
        ceiling = float(ceiling_mb)
    except (TypeError, ValueError):
        return False
    if ceiling <= 0 or rss_mb is None:
        return False
    return float(rss_mb) >= ceiling


def _get_runtime_llm():
    """Lazily build the local (offline) LLM backend for the runtime code-writer.
    Cached process-wide. Returns None if a backend can't be built."""
    global _RUNTIME_LLM
    if _RUNTIME_LLM is not None:
        return _RUNTIME_LLM
    try:
        from llm_backend import get_backend
        _RUNTIME_LLM = get_backend(os.environ.get("V21_RUNTIME_LLM"))
        logger.info("[coder] runtime backend=%s", _RUNTIME_LLM.name)
    except Exception as e:
        logger.warning("[coder] backend unavailable: %s", e)
        _RUNTIME_LLM = None
    return _RUNTIME_LLM


def _planner_click_cap(gid, tier=None, env=None):
    """Max ACTION6 click targets to feed the white-box planners (Go-Explore /
    macro-BFS), or None for unlimited; 0 => suppress clicks entirely.

    Branching-factor fix (C1++++): C1+ ADDED `_scan_click_targets` on the ASSUMPTION
    it returns None on keyboard walls (ls20/ft09) — but run 164123Z showed ls20 L5
    scanning `clicks=32`. Those 32 frame-changing-but-off-solution ACTION6 targets are
    NOT the ls20 solution basis (BFS solves it with 4 simple actions), yet feeding them
    to the planners inflates their branching factor from 4 -> 36, so Go-Explore /
    macro-BFS reach ~9x fewer states in the same budget on exactly the ls20 L5-L6 walls
    they were built to crack. Default: keyboard-tier games (ls20) get 0 clicks; the
    click/reflex tiers (vc33 "orchestration/click", ft09 "reflex/ACTION6-blind") keep
    the full set unchanged, so C1+'s vc33 fix is preserved. Override with an explicit
    integer `V21_PLANNER_CLICK_CAP` (applies to every tier; <=0 disables clicks)."""
    env = os.environ if env is None else env
    tier = (TIER.get(gid, "") if tier is None else tier) or ""
    raw = env.get("V21_PLANNER_CLICK_CAP")
    if raw not in (None, ""):
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    return 0 if "keyboard" in tier.lower() else None


def _scan_click_targets(solver, game, f0, gid=None):
    """Effective ACTION6 click-`data` targets for the white-box planners (C1+).

    Mirrors `blitz.blitz_for_solver`'s enumeration so Go-Explore / macro-BFS search
    the SAME clicks blitz does: the solver's own `_scan_actions` (dedup'd by frame
    effect), optionally augmented with B1 perception connected-component centroids
    under `V21_BRAIN_PERCEPTION`. Without this the planners only try simple actions
    1-5 — inert on click-driven games (vc33 L4-L6), which is why run 144827Z showed
    GOEXPLORE/BRAIN_PLANNER 'no candidate' in <1s there while BLITZ had 30 targets.
    Click breadth is capped per `_planner_click_cap(gid)` so keyboard walls (ls20)
    don't inflate the planner branching factor with off-solution clicks (C1++++).
    Returns a list of `data` dicts, or None (no clicks / not a click game / capped
    to 0 / error). Pure w.r.t. persistent state — probes only forks / the passed
    start game."""
    import numpy as np
    cap = _planner_click_cap(gid)
    if cap == 0:                          # keyboard wall (or explicit override) — no clicks
        return None
    raw_avail = list(getattr(game, "_available_actions", []) or [])
    if 6 not in raw_avail:
        return None
    cts = []
    try:
        bg = int(np.bincount(np.asarray(f0).flatten(), minlength=16).argmax())
        for a, d in solver._scan_actions(game, f0, bg):
            if a == 6 and d is not None:
                cts.append(d)
    except Exception:
        cts = []
    if os.environ.get("V21_BRAIN_PERCEPTION", "0") \
            not in ("", "0", "false", "False", "no", "off"):
        try:
            from blitz import merge_click_targets
            cts = merge_click_targets(cts, f0, True)
        except Exception:
            pass  # scan-only clicks on any error
    if cap is not None and cts and len(cts) > cap:
        cts = cts[:cap]
    return cts or None


def _brain_planner_for_solver(solver, level_idx, gid=None):
    """Stage-3.4: Go-Explore/macro-BFS (brain B3) over the engine as the trusted
    white-box model, from this level's re-rooted start. State = {'g':game,'f':frame}
    so we can hash the frame for dedup; macro edges collapse ls20's corridors.
    Returns a plan (UNVERIFIED — caller verifies + shortest-gates) or None.
    Engine imports are lazy (Mac-only)."""
    import hashlib
    import numpy as np
    from combined_agent import ActionInput, GameAction
    from brain import planner

    res = None
    try:
        res = solver._make_start_state(level_idx)
    except Exception:
        res = None
    if res is None:
        return None
    game, f0 = res
    avail = [a for a in (getattr(game, "_available_actions", []) or []) if a <= 5]
    if not avail:
        avail = [1, 2, 3, 4, 5]

    def _clone(s):
        return {"g": solver._restore(solver._snap(s["g"])), "f": s["f"]}

    def _play(s, step):
        aid, data = step
        ai = (ActionInput(id=GameAction.from_id(aid), data=data) if data
              else ActionInput(id=GameAction.from_id(aid)))
        r = s["g"].perform_action(ai, raw=True)
        if getattr(r, "frame", None):
            s["f"] = np.array(r.frame[-1])
        return int(getattr(r, "levels_completed", 0) or 0)

    def _hash(s):
        f = s["f"]
        fm = np.asarray(f).copy()
        if fm.ndim == 2 and fm.shape[0] > 4:          # mask status bars (top/bottom 2 rows)
            fm[:2] = 0; fm[-2:] = 0
        return hashlib.md5(fm.tobytes()).hexdigest()[:16]

    # C1+: feed click-driven walls (vc33) the same effective ACTION6 targets blitz
    # uses; macro-BFS with only simple actions 1-5 is inert on click games. C1++++:
    # gid lets _scan_click_targets suppress ls20's off-solution clicks (branching fix).
    click_targets = _scan_click_targets(solver, game, f0, gid)
    start = {"g": game, "f": np.asarray(f0)}
    budget = int(os.environ.get("V21_PLANNER_STATES", "200000"))
    macro = int(os.environ.get("V21_PLANNER_MACRO", "64"))
    return planner.plan_in_model_macro(start, avail, _clone, _play, _hash,
                                       goal=level_idx + 1, max_states=budget,
                                       max_macro=macro, click_targets=click_targets)


def _goexplore_for_solver(solver, level_idx, bb=None, gid=None):
    """Stage-3.45 (Epic C1): cell-archive Go-Explore over the engine as the trusted
    white-box model, from this level's re-rooted start. Same {'g','f'} state as the
    brain planner, but dedup is on a COARSE downsampled-frame cell (blackboard
    `cell_key`) instead of the exact frame hash — so ls20 L5's corridors merge into
    a small archive we can return-to instead of a 19k-state frontier. When a
    blackboard is open its toddler `action_order` steers and its verified fragments
    prime the archive. Returns an UNVERIFIED plan (caller verifies + shortest-gates)
    or None. Engine imports are lazy (Mac-only)."""
    import numpy as np
    from combined_agent import ActionInput, GameAction
    from brain import planner
    from brain import blackboard as _BB

    res = None
    try:
        res = solver._make_start_state(level_idx)
    except Exception:
        res = None
    if res is None:
        return None
    game, f0 = res
    avail = [a for a in (getattr(game, "_available_actions", []) or []) if a <= 5]
    if not avail:
        avail = [1, 2, 3, 4, 5]

    def _clone(s):
        return {"g": solver._restore(solver._snap(s["g"])), "f": s["f"]}

    def _play(s, step):
        aid, data = step
        ai = (ActionInput(id=GameAction.from_id(aid), data=data) if data
              else ActionInput(id=GameAction.from_id(aid)))
        r = s["g"].perform_action(ai, raw=True)
        if getattr(r, "frame", None):
            s["f"] = np.array(r.frame[-1])
        return int(getattr(r, "levels_completed", 0) or 0)

    def _cell(s):
        f = np.asarray(s["f"]).copy()
        if f.ndim == 2 and f.shape[0] > 4:            # mask status bars before coarsening
            f[:2] = 0; f[-2:] = 0
        return _BB.cell_key(f, bins=int(os.environ.get("V21_GOEXPLORE_BINS", "8")))

    # toddler order + seeds from the shared scratchpad, when the blackboard is on
    a_order = None
    seeds = None
    if bb is not None:
        # C3 toddler (V21_TODDLER): corpus-prior + online action_effects, frame-aware.
        # Falls back to the raw blackboard action_order when the toddler is off/None.
        a_order = _toddler_order(bb, getattr(bb, "game", None), level_idx, avail, frame=f0)
        if a_order is None:
            try:
                a_order = [a for a in bb.action_order(level_idx) if a in avail]
            except Exception:
                a_order = None
        try:
            seeds = bb.seed_plans(level_idx)
        except Exception:
            seeds = None

    # C1+: on CLICK-driven walls (vc33 L4-L6) simple actions 1-5 are inert, so
    # Go-Explore with only those explores an empty set and returns instantly with no
    # candidate (run 144827Z: vc33 L4 GOEXPLORE no-candidate 0.2s while BLITZ had 30
    # click targets). Hand it the same effective ACTION6 targets blitz enumerates so
    # it can actually search click dynamics; None on keyboard walls (ls20/ft09).
    click_targets = _scan_click_targets(solver, game, f0, gid)
    start = {"g": game, "f": np.asarray(f0)}
    budget = int(os.environ.get("V21_PLANNER_STATES", "200000"))
    macro = int(os.environ.get("V21_PLANNER_MACRO", "64"))
    return planner.plan_in_model_goexplore(start, avail, _clone, _play, _cell,
                                           goal=level_idx + 1, max_states=budget,
                                           max_macro=macro, click_targets=click_targets,
                                           action_order=a_order, seed_plans=seeds)


def _wm_candidate_plans_with_safety(wm, obs, maxlen, gid="?", level_idx=-1):
    """Enumerate a world-model's candidate plans, but NEVER let a crashing or empty
    LLM-authored `candidate_plans()` discard the whole (expensive) Opus-WM call.

    History: on every frontier wall the model's own candidate_plans() kept raising a
    *new* runtime bug — "name 'str' is not defined" (ls20 L5, cron 152556Z),
    "list indices must be integers or slices, not tuple" (ls20 L5, cron 192513Z),
    U+2014 exec-parse (ft09 L2). Patching the sandbox builtins one crash at a time is
    whack-a-mole. The structural fix (mirrors RuntimeCoder.solve_level's safety net):
    always merge the LLM-independent trivial-win plans so the stage still replays real
    candidates against the fork even when the generated enumerator is broken."""
    import runtime_coder as rc
    try:
        plans = list(wm.candidate_plans(maxlen) or [])
    except Exception as e:
        logger.info("[%s L%d] opus WM candidate_plans crashed: %s — using safety-net plans",
                    gid, level_idx, e)
        plans = []
    try:
        plans = plans + rc._safety_net_plans(obs, maxlen)
    except Exception:
        pass
    return plans


def _opus_world_model_for_solver(solver, level_idx, gid):
    """Stage-3.7: ask Opus to WRITE an executable WorldModel .py from the white-box
    source (B2 — the world model, not just a plan). Sandbox-exec it, enumerate its
    candidate plans, replay each on a fork, return the shortest winner (UNVERIFIED —
    caller verifies). Persists the model to brain/wm/<gid>/model.py for reuse."""
    import numpy as np
    from combined_agent import ActionInput, GameAction
    from brain.teacher import OpusTeacher
    import runtime_coder as rc

    teacher = OpusTeacher()
    if not teacher.available():
        return None
    src = ""
    try:
        src = open(getattr(solver, "game_path", "")).read()
    except Exception:
        pass
    res = None
    try:
        res = solver._make_start_state(level_idx)
    except Exception:
        res = None
    if res is None:
        return None
    game, f0 = res
    f0 = np.asarray(f0)
    avail = [a for a in (getattr(game, "_available_actions", []) or []) if 1 <= a <= 5] or [1, 2, 3, 4, 5]

    code = teacher.write_world_model(gid, src, level_idx, avail)
    if not code:
        return None
    # persist the world model .py (the B2 artifact — reusable next run)
    try:
        wmdir = os.path.join(HERE, "brain", "wm", gid)
        os.makedirs(wmdir, exist_ok=True)
        open(os.path.join(wmdir, "model.py"), "w").write(code)
    except Exception:
        pass

    obs = {"level": level_idx, "available_actions": avail,
           "frame": f0.tolist() if hasattr(f0, "tolist") else f0}
    wm, err = rc._exec_world_model(code, obs)
    maxlen = int(os.environ.get("V21_OPUS_WM_MAXLEN", "400"))
    if wm is None:
        # The LLM-authored WM module failed to EXEC at all — a syntax/parse crash
        # BEFORE candidate_plans() can even run (ft09 L2 'unterminated string literal',
        # cron 212257Z; ls20 L5 U+2014 exec-parse earlier). The candidate_plans safety
        # net (below) only protects a *built* model, so previously the whole fork
        # opportunity was discarded on an exec crash. Structural fix: still replay the
        # LLM-independent safety-net plans on the fork so a trivial win can crack the wall.
        logger.info("[%s L%d] opus WM exec failed: %s — using safety-net plans",
                    gid, level_idx, err)
        try:
            plans = rc._safety_net_plans(obs, maxlen)
        except Exception:
            plans = []
    else:
        plans = _wm_candidate_plans_with_safety(wm, obs, maxlen, gid, level_idx)
    if not plans:
        return None

    def _clone(g):
        return solver._restore(solver._snap(g))

    def _play(g, step):
        aid, data = step
        ai = (ActionInput(id=GameAction.from_id(aid), data=data) if data
              else ActionInput(id=GameAction.from_id(aid)))
        r = g.perform_action(ai, raw=True)
        return int(getattr(r, "levels_completed", 0) or 0)

    goal = level_idx + 1
    for plan in sorted([p for p in plans if p], key=len)[:64]:
        if any(a == 6 and isinstance(d, dict) and (d.get("x") is None or d.get("y") is None)
               for a, d in plan):
            continue                         # R2.7 exploit refusal
        try:
            if rc.replay_wins(game, plan, _clone, _play, goal):
                logger.info("[%s L%d] OPUS_WM model-plan wins (%d actions)", gid, level_idx, len(plan))
                return plan
        except Exception:
            continue
    logger.info("[%s L%d] opus WM: no candidate plan won", gid, level_idx)
    return None


def _teacher_ground_enabled(env=None):
    env = os.environ if env is None else env
    return env.get("V21_TEACHER_GROUND", "0") in ("1", "true", "True")


def _teacher_ground2_enabled(env=None):
    """R14 full grounding: give the Opus teacher the REAL level-start frame (symbolic
    scene digest) + a per-action effect table. Independent of the older click-only
    grounding (V21_TEACHER_GROUND). Default OFF in code; run_cadence.sh turns it on."""
    env = os.environ if env is None else env
    return env.get("V21_TEACHER_GROUND2", "0") in ("1", "true", "True")


def _teacher_action_effects(solver, level_idx):
    """Probe each available discrete action ONCE from the re-rooted level start and
    return (start_frame, transitions) where transitions is a list of
    {action, changed, levels_completed} — the per-action effect table that tells the
    teacher which actions actually DO something before it commits to a plan (this run:
    vc33 L4 round-1's first action was a no-op — Opus couldn't know that up front).
    Pure fork (never mutates the real run); engine imports lazy (Mac-only). Returns
    (None, []) on any failure so the teacher path is never broken by grounding."""
    try:
        import numpy as np
        from combined_agent import ActionInput, GameAction
        res = solver._make_start_state(level_idx)
        if res is None:
            return None, []
        game, f0 = res
        f0 = np.asarray(f0)
        avail = [a for a in (getattr(game, "_available_actions", []) or [])
                 if 1 <= a <= 5] or [1, 2, 3, 4, 5]
        trans = []
        for a in avail:
            try:
                g = solver._restore(solver._snap(game))
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                nf = np.array(r.frame[-1]) if getattr(r, "frame", None) else None
                changed = bool(nf is not None and not np.array_equal(nf, f0))
                lc = int(getattr(r, "levels_completed", 0) or 0)
                trans.append({"action": int(a), "changed": changed,
                              "levels_completed": lc})
            except Exception:
                continue
        return f0, trans
    except Exception:
        return None, []


def _teacher_state_digest(solver, level_idx):
    """Build the GROUNDED current-state block for the Opus teacher: the real level-start
    frame turned into a symbolic scene (objects/centroids/click targets via
    brain.summarize.digest) PLUS the per-action effect table. This is the fix for the
    teacher 'reading the rulebook but playing blindfolded' — it now sees the actual
    board and what each action does from here, not just the source. Pure; returns "" on
    any failure so the teacher path degrades to the ungrounded prompt."""
    try:
        f0, trans = _teacher_action_effects(solver, level_idx)
        if f0 is None:
            return "", 0
        from brain.summarize import digest
        obs = {"level": level_idx,
               "available_actions": [t["action"] for t in trans] or [1, 2, 3, 4, 5],
               "frame": f0.tolist() if hasattr(f0, "tolist") else f0,
               "transitions": trans}
        return (digest(obs) or ""), len(trans)
    except Exception:
        return "", 0


def _teacher_click_note(frame, limit=24, max_chars=500):
    """R8/B1 grounding: format the level-start frame's valid ACTION6 click targets
    (B1 perception component centroids) as a bounded note so the Opus teacher clicks
    REAL objects instead of dead coordinates (run 152556Z: vc33 L4 round 1 first
    action was a no-op — clicked empty space). Pure — imports only brain.perception;
    returns "" on any failure so the teacher path is never broken by grounding."""
    try:
        from brain.perception import to_grid, click_targets
        grid = to_grid(frame)
        if not grid or not grid[0]:
            return ""
        tgts = click_targets(grid, limit=limit)
        if not tgts:
            return ""
        pairs = ", ".join("(%d,%d)" % (t["x"], t["y"]) for t in tgts)
        note = ("Valid ACTION6 click targets (col,row) from perception of the "
                "level-start frame — prefer clicking these object centroids over "
                "guessed coordinates: [" + pairs + "].")
        if len(note) > max_chars:
            note = note[:max_chars - 2].rstrip().rstrip(",") + "]."
        return note
    except Exception:
        return ""


def _format_click_note(probed, max_chars=500):
    """Pure formatter for the ENGINE-PROBED ACTION6 grounding note. `probed` is a list
    of {'x','y','changed','lc'} from _probe_click_targets (each perception centroid
    actually clicked once on a fork of the level-start state). VERIFIED-effective
    targets (those that change the board) are recommended first; VERIFIED no-ops are
    listed as 'never lead a plan with these' so the teacher stops opening a plan with a
    dead click (ft09 L2 / vc33 L4 round-1 first-action no-ops, cron 030701Z). Pure;
    returns '' on empty input."""
    if not probed:
        return ""
    try:
        eff = [t for t in probed if t.get("changed")]
        dead = [t for t in probed if not t.get("changed")]
        _fmt = lambda ts: ", ".join("(%d,%d)" % (int(t["x"]), int(t["y"])) for t in ts)
        if eff:
            note = ("Probed ACTION6 click targets (col,row) — VERIFIED to change the "
                    "board from the level-start frame, prefer these: [" + _fmt(eff) + "].")
            if dead:
                note += (" These are VERIFIED no-ops from the start (never lead a plan "
                         "with them): [" + _fmt(dead) + "].")
        else:
            note = ("Probed ACTION6 click targets (col,row) — none changed the "
                    "level-start frame alone (may still matter after other actions): ["
                    + _fmt(probed) + "].")
        if len(note) > max_chars:
            note = note[:max_chars - 2].rstrip().rstrip(",") + "]."
        return note
    except Exception:
        return ""


def _probe_click_targets(solver, level_idx, max_targets=8):
    """Fork the re-rooted level-start state and actually PERFORM ACTION6 at each
    perception click target once, recording whether it changes the frame + reached
    levels_completed. Upgrades the R8/B1 STATIC-centroid grounding (unverified guesses)
    to engine-verified effective targets, killing the leading-no-op-click failure
    (ft09 L2 / vc33 L4). Pure fork (never mutates the real run); engine imports are
    Mac-only/lazy. Returns None on any failure so the teacher path degrades to the
    static note. `max_targets` bounds the click probes (each is an engine step)."""
    try:
        import numpy as np
        from combined_agent import ActionInput, GameAction
        from brain.perception import click_targets
        res = solver._make_start_state(level_idx)
        if res is None:
            return None
        game, f0 = res
        f0 = np.asarray(f0)
        tgts = click_targets(f0.tolist(), limit=max_targets) or []
        if not tgts:
            return None
        probed = []
        for t in tgts:
            try:
                g = solver._restore(solver._snap(game))
                r = g.perform_action(
                    ActionInput(id=GameAction.from_id(6),
                                data={"x": t["x"], "y": t["y"]}), raw=True)
                nf = np.array(r.frame[-1]) if getattr(r, "frame", None) else None
                changed = bool(nf is not None and not np.array_equal(nf, f0))
                lc = int(getattr(r, "levels_completed", 0) or 0)
            except Exception:
                changed, lc = False, 0
            probed.append({"x": int(t["x"]), "y": int(t["y"]),
                           "changed": changed, "lc": lc})
        return probed
    except Exception:
        return None


def _teacher_effective_click_note(solver, level_idx, frame):
    """R8/B1+ grounding: probe each perception click target on a fork and hand the Opus
    teacher the ENGINE-VERIFIED effective ACTION6 targets. Falls back to the static
    unverified centroid note when the engine probe is unavailable (e.g. offline sandbox
    or no click targets). Pure; returns '' only if both paths yield nothing."""
    try:
        note = _format_click_note(_probe_click_targets(solver, level_idx))
        if note:
            return note
    except Exception:
        pass
    return _teacher_click_note(frame)


def _opus_teacher_for_solver(solver, level_idx, gid):
    """Stage-3.6: hand the WHITE-BOX game source + stuck level to cloud Opus and get
    a candidate plan. UNVERIFIED (caller verifies + shortest-gates). Refuses the
    null-coord ACTION6 exploit. Needs ANTHROPIC_API_KEY in the environment."""
    from brain.teacher import OpusTeacher
    teacher = OpusTeacher()
    if not teacher.available():
        logger.info("[%s L%d] opus teacher: no API key — skipping", gid, level_idx)
        return None
    # read the version-exact source the solver loaded (the white-box advantage)
    src = ""
    try:
        src = open(getattr(solver, "game_path", "")).read()
    except Exception:
        pass
    avail = [1, 2, 3, 4, 5]
    _f0 = None
    try:
        res = solver._make_start_state(level_idx)
        if res is not None:
            g, _f0 = res
            avail = [a for a in (getattr(g, "_available_actions", []) or []) if 1 <= a <= 5] or avail
    except Exception:
        pass
    notes = f"local blitz/BFS/Go-Explore/Qwen all failed level {level_idx}."

    # R8/B1 click-target GROUNDING: on ACTION6 games (vc33/ft09) the teacher must
    # guess x,y from reading the source; run 152556Z showed vc33 L4 round 1 clicked
    # empty space (first no-op at action index 0, delta 0 cells changed). Hand Opus
    # the level-START frame's valid click targets (B1 perception component centroids)
    # up front so its FIRST-round clicks land on real objects, not dead coordinates.
    # This complements R6/R8's failure-scene feedback (which only fires round 2+).
    # Env V21_TEACHER_GROUND (default OFF); pure + degrades to no-op; the plan is
    # still verify + shortest + exploit-gated so a bad ground note can't corrupt the
    # corpus. _f0 is the start frame captured from _make_start_state above.
    if _teacher_ground_enabled() and _f0 is not None:
        _gnote = _teacher_effective_click_note(solver, level_idx, _f0)
        if _gnote:
            notes = notes + " " + _gnote

    # R7(a) workspace counterexamples: read the dead-ends this wall accumulated on
    # PRIOR runs and tell Opus not to re-propose them (a fresh cadence otherwise
    # starts blank and re-tries the same near-miss). Env V21_WORKSPACE_COUNTEREX;
    # degrades to no-op when off. The bb handle is reused to RECORD each round's
    # failure below so the constraint grows across runs.
    _cex_bb = _counterex_open(gid)
    _cex = _counterex_notes(_cex_bb, level_idx)
    if _cex:
        notes = notes + " " + _cex

    # R7 teach-with-feedback: EXECUTE each proposed plan on a fresh fork and feed the
    # engine's failure report (how far it got + where it stalled) back to Opus for the
    # next round, instead of discarding a near-miss (this run: ls20 L5 got a 19-action
    # plan that failed verify and was thrown away). Env V21_OPUS_ROUNDS controls rounds
    # (default 2); 1 preserves the old single-shot behavior.
    # OBSERVABILITY (this cycle): the iterative loop RETURNS None when every round
    # fails verify (the common case on an uncracked wall), so the caller's single
    # INFO line below ("opus teacher proposed ...") is never reached and the whole
    # teacher effort goes invisible in the cron log — exactly what happened on run
    # 213152Z (ls20 L5/L6 showed no teacher line, unlike single-shot run 194224Z).
    # Log EACH round's proposed-plan length + how far it reached, so PART-B health
    # reporting can always see the teacher fired and whether it is getting closer.
    _round = {"n": 0}

    def _try_plan(p):
        _round["n"] += 1
        try:
            solved = _verify(solver, level_idx, p)
        except Exception:
            solved = False
        if solved:
            logger.info("[%s L%d] OPUS_TEACHER round %d: %d-action plan SOLVED",
                        gid, level_idx, _round["n"], len(p or []))
            return True, "solved"
        fb = _replay_feedback(solver, level_idx, p)
        logger.info("[%s L%d] OPUS_TEACHER round %d: %d-action plan failed verify — %s",
                    gid, level_idx, _round["n"], len(p or []), fb)
        # R7(a): persist this dead-end so the NEXT run's teacher won't re-propose it.
        _counterex_record(_cex_bb, level_idx, p)
        return False, fb

    # R14 GROUNDING: hand Opus the REAL level-start frame (symbolic scene digest) + a
    # per-action effect table, so it plans over the actual board instead of blind over
    # the source. This is the fix for the core teacher failure mode (this run: ls20 L5
    # plans changed 86-90 cells but never crossed the goal; vc33 L4 first action a
    # no-op). Env V21_TEACHER_GROUND2 (default OFF); pure + degrades to the old prompt
    # on any failure; the plan is still verify + shortest + exploit-gated so a bad
    # digest can never corrupt the corpus.
    state = ""
    if _teacher_ground2_enabled():
        state, n_probed = _teacher_state_digest(solver, level_idx)
        if state:
            logger.info("[%s L%d] OPUS_TEACHER grounded: %d-char state digest, %d actions probed",
                        gid, level_idx, len(state), n_probed)

    try:
        rounds = int(os.environ.get("V21_OPUS_ROUNDS", "2"))
    except Exception:
        rounds = 2
    if rounds > 1 and hasattr(teacher, "solve_wall_iterative"):
        plan = teacher.solve_wall_iterative(
            gid, src, level_idx, avail, _try_plan, max_rounds=rounds, notes=notes, state=state)
    else:
        plan = teacher.solve_wall(gid, src, level_idx, avail, notes=notes, state=state)
    if not plan:
        return None
    # R2.7: never accept the null-coordinate ACTION6 exploit
    for a, d in plan:
        if a == 6 and isinstance(d, dict) and (d.get("x") is None or d.get("y") is None):
            logger.info("[%s L%d] opus plan used null-coord ACTION6 — refused", gid, level_idx)
            return None
    logger.info("[%s L%d] opus teacher proposed a %d-action plan", gid, level_idx, len(plan))
    return plan


def _replay_feedback(solver, level_idx, plan):
    """Replay a proposed plan on a FRESH fork and return a compact textual report of
    how far it got — the R7 feedback the teacher's next round reads. Never mutates the
    real run (fork only); engine imports are lazy (Mac-only). Degrades to a generic
    note on any error so the teach loop is never broken by feedback generation."""
    try:
        import numpy as np
        from combined_agent import ActionInput, GameAction
        res = solver._make_start_state(level_idx)
        if res is None:
            return "could not re-root level %d to replay" % level_idx
        game, f0 = res
        g = solver._restore(solver._snap(game))
        f0 = np.asarray(f0)
        goal = level_idx + 1
        reached = 0
        prev = f0
        stalled_at = None
        for i, (a, d) in enumerate(plan or []):
            try:
                if int(a) == 6 and isinstance(d, dict):
                    ai = ActionInput(id=GameAction.from_id(6),
                                     data={"x": d.get("x"), "y": d.get("y")})
                else:
                    ai = ActionInput(id=GameAction.from_id(int(a)))
                r = g.perform_action(ai, raw=True)
            except Exception:
                stalled_at = i
                break
            reached = max(reached, int(getattr(r, "levels_completed", 0) or 0))
            nf = np.array(r.frame[-1]) if getattr(r, "frame", None) else None
            if nf is not None:
                if stalled_at is None and np.array_equal(nf, prev):
                    stalled_at = i  # first action that changed nothing
                prev = nf
            if reached >= goal:
                break
        parts = ["reached levels_completed=%d of goal %d after %d/%d actions"
                 % (reached, goal, min(len(plan or []), (stalled_at + 1) if stalled_at is not None else len(plan or [])), len(plan or []))]
        if stalled_at is not None:
            parts.append("first no-op/failure at action index %d (%s)"
                         % (stalled_at, (plan[stalled_at][0] if stalled_at < len(plan) else "?")))
        # R6/R8: hand the teacher a perception-first view of the STUCK end state
        # (objects + delta-from-start), not just a level count, so the next R13
        # round reasons over what the wall looks like. Pure + bounded; degrades
        # to no extra note on any error.
        try:
            from brain.summarize import plan_failure_scene
            note = plan_failure_scene(f0, prev)
            if note:
                parts.append(note)
        except Exception:
            pass
        return "; ".join(parts)
    except Exception as e:
        return "replay feedback unavailable: %s" % e


def _harvest_toddler_samples(solver, level_idx, gid):
    """Probe each action once from the level's re-rooted start and append labeled
    (frame, action -> changed/won) samples for the neural toddler CNN (R11). Both
    positives (actions that change/win) and negatives (no-ops) are recorded — the
    exact supervised signal StochasticGoose uses. Engine imports lazy (Mac-only)."""
    import numpy as np
    from combined_agent import ActionInput, GameAction
    from brain import toddler_net as TN

    res = None
    try:
        res = solver._make_start_state(level_idx)
    except Exception:
        res = None
    if res is None:
        return
    import random
    game, f0 = res
    avail = [a for a in (getattr(game, "_available_actions", []) or []) if 1 <= a <= 5] or [1, 2, 3, 4, 5]
    steps = int(os.environ.get("V21_TODDLER_HARVEST_STEPS", "24"))   # rollout length
    samples = []
    cur = game
    curf = np.asarray(f0)
    for _ in range(steps):
        fl = curf.tolist() if hasattr(curf, "tolist") else curf
        # probe EVERY action once from the current state (frame,action->changed/won)
        for a in avail:
            try:
                g = solver._restore(solver._snap(cur))
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                nf = np.array(r.frame[-1]) if getattr(r, "frame", None) else None
                changed = bool(nf is not None and not np.array_equal(nf, curf))
                won = int(getattr(r, "levels_completed", 0) or 0) >= level_idx + 1
                samples.append({"frame": fl, "action": a, "changed": changed, "won": won})
            except Exception:
                continue
        # advance one random step to visit a NEW state (breadth of training data)
        try:
            nxt = solver._restore(solver._snap(cur))
            r = nxt.perform_action(ActionInput(id=GameAction.from_id(random.choice(avail))), raw=True)
            if getattr(r, "frame", None):
                curf = np.array(r.frame[-1]); cur = nxt
            else:
                break
        except Exception:
            break
    if samples:
        TN.append_samples(gid, samples)
        logger.info("[%s L%d] toddler harvest: +%d samples (rollout=%d)",
                    gid, level_idx, len(samples), steps)


def _runtime_coder_for_solver(solver, level_idx, llm, max_len):
    """Stage-3.5: when blitz + BFS both fail a wall level, hand the observed
    transitions to the RuntimeCoder (local Qwen world-model writer). It writes a
    WorldModel, sandbox-execs it, enumerates SHORTEST-first candidate plans and
    replays each on a fresh fork; returns the first winning plan (UNVERIFIED — the
    caller still runs verify_solution + the shortest-plan gate) or None.

    Engine imports are lazy (Mac-only) so `import cadence_runner` stays light.
    """
    import runtime_coder as rc
    from combined_agent import ActionInput, GameAction  # lazy: Mac-only deps
    import numpy as np

    # --- build the level's TRUE chained start state (reuse the solver's own) ---
    game, f0 = None, None
    try:
        res = solver._make_start_state(level_idx)
        if res is not None:
            game, f0 = res
    except Exception:
        game = None
    if game is None:
        return None

    avail = list(getattr(game, "_available_actions", []) or [])
    simple = [a for a in avail if a <= 5]

    def _clone(g):
        return solver._restore(solver._snap(g))

    def _play(g, step):
        aid, data = step
        ai = (ActionInput(id=GameAction.from_id(aid), data=data)
              if data else ActionInput(id=GameAction.from_id(aid)))
        r = g.perform_action(ai, raw=True)
        return int(getattr(r, "levels_completed", 0) or 0)

    # --- observations: initial frame + one-step (action -> resulting frame) ---
    obs = {"level": level_idx, "available_actions": avail,
           "frame": (f0.tolist() if hasattr(f0, "tolist") else f0),
           "transitions": []}
    for aid in simple:
        try:
            g = _clone(game)
            ai = ActionInput(id=GameAction.from_id(aid))
            r = g.perform_action(ai, raw=True)
            nf = np.array(r.frame[-1]) if getattr(r, "frame", None) else None
            obs["transitions"].append({
                "action": aid,
                "levels_completed": int(getattr(r, "levels_completed", 0) or 0),
                "changed": bool(nf is not None and not np.array_equal(nf, f0))})
        except Exception:
            continue

    goal = level_idx + 1
    try_plan = lambda plan: rc.replay_wins(game, plan, _clone, _play, goal)
    coder = rc.RuntimeCoder(llm, max_len=max_len)
    try:
        return coder.solve_level(obs, try_plan)
    except Exception as e:
        logger.debug("[coder] L%d runtime solve error: %s", level_idx, e)
        return None


# ---- regression gate (R1.5) ---------------------------------------------------
def game_rhae(rows, gid):
    rs = [x["rhae"] for x in rows if x["game"] == gid]
    return sum(rs) / len(rs) if rs else 0.0


def last_game_rhae(gid):
    p = os.path.join(LOGDIR, "scorecard.jsonl")
    if not os.path.exists(p):
        return None
    best = None
    for line in open(p):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("game") == gid and "game_rhae" in d:
            best = d["game_rhae"]  # last one wins (chronological)
    return best


# ---- Phase 2: the 274-game generalization corpus (CONTINGENT on 3-game crack) --
def _all_levels_solved(n_solved, n_levels):
    """Pure gate: a game is 'cracked' iff every level has a verified solution."""
    return n_levels > 0 and n_solved >= n_levels


def default_games_cracked(default_games=None):
    """True only when EVERY default game (ls20/ft09/vc33) has all its levels solved
    in the corpus. This is the hard gate that unlocks Phase 2 — the 274-game corpus
    stays dark until the 3-game focus set is fully cracked."""
    for gid in (default_games or DEFAULT_GAMES):
        info = resolve_source(gid)
        if not info:
            return False
        baselines, _ = load_baselines(info[0], gid)
        n_levels = len(baselines) if baselines else 0
        corpus = load_corpus(gid)
        if not _all_levels_solved(len(corpus), n_levels):
            return False
    return True


def discover_all_games():
    """Every game id whose <id>/**/<id>.py engine source is reachable (the 274-game
    testbed lives here already; solve_all.discover_games uses the same roots)."""
    found = set()
    for base in GAME_ROOTS + ([os.environ["V21_EXTRA_GAMES_DIR"]] if os.environ.get("V21_EXTRA_GAMES_DIR") else []):
        if base and os.path.isdir(base):
            for d in os.listdir(base):
                if glob.glob(os.path.join(base, d, "**", f"{d}.py"), recursive=True):
                    found.add(d)
    return sorted(found)


def phase2_harvest(BFSSolver, max_games=None, exclude=None):
    """Phase 2 (generalization): harvest neural-toddler samples across the WIDE game
    corpus — probe L0 of each discovered game so the toddler learns a game-AGNOSTIC
    frame-change prior. Bounded per run (V21_PHASE2_MAX). Harvest-only + guarded; it
    never solves/commits, so it can't touch the verified 3-game corpus."""
    exclude = set(exclude or DEFAULT_GAMES)
    games = [g for g in discover_all_games() if g not in exclude]
    cap = int(max_games if max_games is not None else os.environ.get("V21_PHASE2_MAX", "40"))
    done = 0
    for gid in games:
        if done >= cap:
            break
        info = resolve_source(gid)
        if not info:
            continue
        path, cls, ver = info
        try:
            solver = BFSSolver(path, cls, bfs_timeout=10)
            if not solver.load():
                continue
            _harvest_toddler_samples(solver, 0, gid)   # probe L0 -> (frame,action)->changed
            done += 1
        except Exception as e:
            logger.debug("[phase2 %s] harvest skipped: %s", gid, e)
    logger.info("phase2 harvest: probed %d/%d wide games (cap %d)", done, len(games), cap)
    return done


def walls_for(gid, corpus):
    """Unsolved (wall) levels + official baselines — the evolve code-writer's target."""
    info = resolve_source(gid)
    if not info:
        return []
    baselines, _ = load_baselines(info[0], gid)
    n = len(baselines) if baselines else (max(corpus, default=-1) + 1)
    return [{"game": gid, "level": lvl,
             "baseline": (baselines[lvl] if baselines and lvl < len(baselines) else None)}
            for lvl in range(n) if lvl not in corpus]


def _corpus_eval_fn(train_rhae):
    """Injected eval_fn(config, games)->{game:rhae}. Config-INSENSITIVE floor: scores
    the current verified corpus so evolve never promotes on noise. Kept as the safe
    fallback; the config-aware evaluator (evolve.config_aware_eval_fn) supersedes it
    whenever a live wall probe is available."""
    def _f(config, games):
        return {g: train_rhae.get(g, 0.0) for g in games}
    return _f


def _make_evolve_probe(BFSSolver, bfs_timeout):
    """Config-aware WALL probe for evolve.config_aware_eval_fn (BACKLOG #1).
    probe(config, gid, level) -> shortest verified action count | None, by APPLYING
    the challenger's blitz_K to the real engine budget (bigger blitz_K -> more BFS
    states, so a budget-gated wall the champion missed can be cracked and PROMOTED).

    OFF by default (returns None) — the live rollout multiplies cadence wall-clock, so
    it is opt-in via env V21_EVOLVE_PROBE=1 once the box's time budget allows it."""
    if os.environ.get("V21_EVOLVE_PROBE", "0") not in ("1", "true", "True"):
        return None
    cache = {}  # gid -> loaded BFSSolver (reused across challengers)

    def _solver_for(gid):
        if gid in cache:
            return cache[gid]
        info = resolve_source(gid)
        if not info:
            cache[gid] = None
            return None
        path, cls, _ver = info
        try:
            s = BFSSolver(path, cls, bfs_timeout=bfs_timeout,
                          workers=int(os.environ.get("V21_BFS_WORKERS", "1")))
            cache[gid] = s if s.load() else None
        except Exception as e:
            logger.warning("[evolve-probe] %s solver load failed: %s", gid, e)
            cache[gid] = None
        return cache[gid]

    def probe(config, gid, level):
        solver = _solver_for(gid)
        if solver is None:
            return None
        blitz_K = int(config.get("blitz_K", 200) or 200)
        max_states = max(50_000, min(2_000_000, blitz_K * 5_000))
        try:
            sol = solver.solve_level(level, max_states=max_states)
        except Exception as e:
            logger.debug("[evolve-probe] %s L%s solve error: %s", gid, level, e)
            return None
        if sol and _verify(solver, level, sol):
            return len(sol)
        return None

    return probe


# ---- main ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(DEFAULT_GAMES))
    ap.add_argument("--bfs-timeout", type=int, default=180, help="seconds per level")
    ap.add_argument("--allow-network", action="store_true",
                    help="disable the offline guardrail (cadence world-model box only)")
    ap.add_argument("--world-model", action="store_true",
                    help="run the optional offline WM generator (needs --allow-network)")
    ap.add_argument("--evolve", action="store_true",
                    help="run the EVOLVE half: distill intuition + champion/challenger code-writer")
    ap.add_argument("--llm-backend", default=None, help="auto|hf|openai|mock (default: env/auto)")
    ap.add_argument("--heldout", default="cn04,sk48,tu93",
                    help="held-out games for the generalization gate in evolve")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.FileHandler(os.path.join(LOGDIR, "cadence.log")),
                                  logging.StreamHandler(sys.stdout)])

    # lock guard (R6.5)
    lock = open(LOCKFILE, "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("another cadence run is active — exiting"); return 0

    _install_offline_guard(enabled=not args.allow_network)
    for p in (V19_SRC, V20_SRC):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # CPU BFS; safe fork pools
    from combined_agent import BFSSolver                # v19, read-only

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    all_rows, bank = [], (json.load(open(MACRO_BANK)) if os.path.exists(MACRO_BANK) else {})
    summary_lines = [f"# v21 cadence {run_id} (bfs_timeout={args.bfs_timeout}s)"]
    walls, cur_rhae = [], {}                       # for the EVOLVE half

    for gid in games:
        rows, corpus, improved = solve_game(gid, args.bfs_timeout, BFSSolver)
        gr = game_rhae(rows, gid)
        cur_rhae[gid] = round(gr, 4); walls += walls_for(gid, corpus)
        prev = last_game_rhae(gid)
        # regression gate: only replace the shipped corpus if not worse (R1.5)
        if prev is not None and gr < prev - 1e-6:
            logger.warning("[%s] REGRESSION %.4f < %.4f — keeping previous corpus", gid, gr, prev)
            summary_lines.append(f"- {gid} ({TIER[gid]}): REGRESSION {gr:.3f}<{prev:.3f} — kept prior")
        else:
            save_corpus(gid, corpus)
            bank = harvest_macros(gid, corpus, bank)
            delta = "" if prev is None else f" (Δ{gr-prev:+.3f})"
            summary_lines.append(f"- {gid} ({TIER[gid]}): RHAE {gr:.3f}{delta}"
                                 f"{' [improved]' if improved else ''}")
        # append scorecard rows (R6.3)
        with open(os.path.join(LOGDIR, "scorecard.jsonl"), "a") as f:
            for r in rows:
                r["run_id"] = run_id; f.write(json.dumps(r) + "\n")
            f.write(json.dumps({"run_id": run_id, "game": gid, "game_rhae": round(gr, 4),
                                "improved": improved}) + "\n")
        # csv history (R7.4)
        with open(os.path.join(LOGDIR, "rhae_history.csv"), "a") as f:
            f.write(f"{run_id},{gid},{TIER[gid]},{gr:.4f},{int(improved)}\n")
        all_rows += rows

    json.dump(bank, open(MACRO_BANK, "w"))

    # ---- EVOLVE half: distill intuition + champion/challenger code-writer --------
    # (1) Intuition prior is distilled EVERY run — it's cheap and pure-local.
    try:
        import intuition
        intuition.distill(SOLDIR, os.path.join(HERE, "intuition_prior.json"))
        summary_lines.append("- intuition: prior re-distilled from corpus")
    except Exception as e:
        logger.warning("intuition distill skipped: %s", e)

    # (1b) Neural toddler TRAIN step (Epic C3 / R11) — wake-sleep consolidation.
    # Trains the StochasticGoose-style frame-change CNN on the harvested samples,
    # on the Mac GPU (MPS auto-detected). Env-gated V21_TODDLER_NET; degrades to a
    # logged reason if torch/data are missing (never blocks the run).
    if os.environ.get("V21_TODDLER_NET", "0") in ("1", "true", "True"):
        try:
            from brain.toddler_net import ToddlerNet, last_champion_acc, adaptive_epochs
            for gid in games:
                # Train the WEAKEST world models harder: read the game's last held-out
                # champion_acc (opus_arch audit, persists on disk with no cloud call) and
                # scale epochs up when it's below floor. ft09 (0.8526) trains deeper than
                # ls20/vc33 (1.0). Unknown acc -> base epochs = today's behavior exactly.
                _acc = last_champion_acc(gid)
                _ep = adaptive_epochs(_acc)
                status = ToddlerNet(gid).train(epochs=_ep)
                logger.info("[%s] toddler_net train: %s (adaptive epochs=%s, last_acc=%s)",
                            gid, status, _ep, _acc)
                summary_lines.append(f"- toddler_net[{gid}]: {status} (epochs={_ep})")
        except Exception as e:
            logger.warning("toddler_net train skipped: %s", e)

    # (1b2) OPUS-AS-ML-ENGINEER (R13 x R11): Opus DESIGNS the toddler's PyTorch net.
    # Champion/challenger on held-out accuracy — Opus writes an improved build_net, we
    # train+score it on the Mac GPU, and ADOPT it (brain/toddler/<gid>_arch.py) only if
    # it beats the current net. Then the next train uses the adopted architecture.
    # Env-gated V21_OPUS_ARCH (needs ANTHROPIC_API_KEY + torch + enough samples).
    if os.environ.get("V21_OPUS_ARCH", "0") in ("1", "true", "True"):
        try:
            from brain.toddler_net import opus_arch_step
            from brain.teacher import OpusTeacher
            teacher = OpusTeacher()
            if teacher.available():
                for gid in games:
                    st = opus_arch_step(gid, teacher)
                    logger.info("[%s] opus_arch: %s", gid, st)
                    summary_lines.append(f"- opus_arch[{gid}]: {st}")
        except Exception as e:
            logger.warning("opus_arch step skipped: %s", e)

    # (1c) PHASE 2 (274-game generalization) — HARD-GATED on the 3 default games
    # being fully cracked. Until ls20+ft09+vc33 are 100% solved this is a no-op that
    # just reports it's still gated; once cracked, it harvests toddler samples across
    # the wide corpus so the intuitive prior + world models generalize to unseen games.
    if os.environ.get("V21_PHASE2", "0") in ("1", "true", "True"):
        try:
            if default_games_cracked():
                n = phase2_harvest(BFSSolver)
                summary_lines.append(f"- PHASE 2 UNLOCKED (3 games cracked): harvested {n} wide games")
            else:
                summary_lines.append("- phase 2: GATED — 3 default games not yet fully solved")
        except Exception as e:
            logger.warning("phase 2 skipped: %s", e)

    # (2) The code-writer evolution runs only with --evolve (needs an LLM backend).
    if args.evolve:
        try:
            import evolve
            from llm_backend import get_backend
            llm = get_backend(args.llm_backend)
            logger.info("[evolve] backend=%s", llm.name)
            heldout = [g.strip() for g in args.heldout.split(",") if g.strip()]
            # BACKLOG #1: config-aware evaluator. Bucket walls per game, then build a
            # config-sensitive eval_fn. The live wall probe is opt-in (V21_EVOLVE_PROBE);
            # without it the evaluator safely degrades to the corpus floor.
            walls_by_game = {}
            for w in walls:
                walls_by_game.setdefault(w["game"], []).append(w)
            probe_fn = _make_evolve_probe(BFSSolver, args.bfs_timeout)
            eval_fn = evolve.config_aware_eval_fn(cur_rhae, walls_by_game, probe_fn)
            # R7(b) action-frugality tie-break (DREAMTEAM 2605.09650, 31% fewer env-actions):
            # feed the SAME live wall probe as a cost_fn so a challenger that TIES held-out
            # RHAE but solves the walls in STRICTLY fewer env-actions still PROMOTES(frugal).
            # Degrades inert offline — with probe_fn None (V21_EVOLVE_PROBE off) config_aware_cost_fn
            # returns 0.0 for every config, so best_cost==cc and the frugal branch can never fire;
            # nothing promotes on noise. The generalization + strict-RHAE gates are untouched.
            cost_fn = evolve.config_aware_cost_fn(walls_by_game, probe_fn)
            # C1+++ END-OF-SWEEP STALL GUARD: evolve_step drives the SAME local ollama
            # code-writer that wedged run 164123Z's RUNTIME_CODER (fixed by C1++). But
            # evolve is the LAST stage, and C1++ only wrapped RUNTIME_CODER — so if the
            # evolve code-writer hangs, the run never logs a clean `cadence exit=` line
            # and the stale .cadence.lock can block the next launchd cadence. Bound it
            # with the same _call_with_deadline watchdog. Default 5400s = 90 min ≈ 2x
            # the observed legit run (144827Z evolve ran 11:58→12:41 = 43 min on n=4
            # challengers), so a real evolve never aborts while a truly wedged model
            # can't strand the sweep. V21_EVOLVE_BUDGET<=0 restores the legacy inline
            # call. On abandon champion.json is left untouched (no promotion this run),
            # and evolve_step only ever writes on a gen+strict-RHAE-gated promotion, so
            # even a late-finishing daemon thread cannot worsen the corpus.
            _evo_budget = float(os.environ.get("V21_EVOLVE_BUDGET", "5400"))
            champ, promoted = _call_with_deadline(
                lambda: evolve.evolve_step(
                    os.path.join(HERE, "champion.json"),
                    os.path.join(LOGDIR, "evolution_history.jsonl"),
                    walls, cur_rhae, llm, eval_fn, games, heldout, n=4, cost_fn=cost_fn),
                _evo_budget)
            probe_note = "live-probe" if probe_fn else "corpus-floor"
            summary_lines.append(
                f"- evolve: {'PROMOTED champion v'+str(champ.get('version')) if promoted else 'no promotion'} "
                f"(backend={llm.name}, {len(walls)} wall levels targeted, eval={probe_note})")
        except TimeoutError as e:
            logger.warning("[evolve] abandoned (%s) — champion unchanged this run", e)
            summary_lines.append(f"- evolve: abandoned (hard deadline {int(_evo_budget)}s)")
        except Exception as e:
            logger.warning("evolve step skipped: %s", e)
            summary_lines.append(f"- evolve: skipped ({e})")

    open(os.path.join(LOGDIR, "last_summary.md"), "w").write("\n".join(summary_lines) + "\n")
    print("\n".join(summary_lines))

    if args.world_model:
        if not args.allow_network:
            logger.error("--world-model requires --allow-network (offline generator, cadence box only)")
        else:
            logger.info("world-model generator hook — implement offline distillation here (R6.6)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
