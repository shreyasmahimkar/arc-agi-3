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

    for lvl in range(n_levels):
        prev = corpus.get(lvl)
        prev_len = len(prev) if prev else None
        # verify the existing plan still holds on this engine version (R5.3)
        if prev and _verify(solver, lvl, prev):
            best, best_len = prev, prev_len
        else:
            best, best_len = None, None
        # Stage-0 blitz pre-pass (BACKLOG #2): only for UNSOLVED (wall) levels —
        # solved levels already have a verified corpus plan, so this adds ZERO
        # cost there. Cheap depth-1 / repeat-K wins crack reflex/orchestration
        # walls that BFS times out on. Fully guarded; any error falls through to
        # BFS. The candidate is still verified + shortest-gated below.
        if best is None and os.environ.get("V21_BLITZ", "1") not in ("0", "false", "False"):
            try:
                bsol = blitz.blitz_for_solver(
                    solver, lvl, repeat_K=int(os.environ.get("V21_BLITZ_K", "200")))
            except Exception as e:
                bsol = None
                logger.debug("[%s L%d] blitz error: %s", gid, lvl, e)
            if bsol and _verify(solver, lvl, bsol):
                best, best_len, improved = bsol, len(bsol), True
                corpus[lvl] = bsol
                logger.info("[%s L%d] BLITZ solved in %d actions", gid, lvl, len(bsol))
        # attempt a fresh (optimal-preferring) solve
        try:
            sol = solver.solve_level(lvl)  # v19 'auto' ladder, shortest-first
        except Exception as e:
            sol = None
            logger.debug("[%s L%d] solve error: %s", gid, lvl, e)
        if sol and _verify(solver, lvl, sol) and (best_len is None or len(sol) < best_len):
            best, best_len, improved = sol, len(sol), True
            corpus[lvl] = sol
        if best is None:
            logger.info("[%s L%d] UNSOLVED at budget %ss", gid, lvl, bfs_timeout)
            continue
        solver.solutions[lvl] = best        # chain: later levels verify from here (R5.3)
        hb, hbsrc = baseline_for(baselines, bsrc, gid, lvl)
        r = rhae_level(hb if hb else best_len, best_len)
        rows.append({"game": gid, "tier": TIER.get(gid, "?"), "level": lvl,
                     "actions": best_len, "rhae": round(r, 4),
                     "baseline": hb, "baseline_src": hbsrc})
        flag = " <-- OVER BASELINE" if hb and best_len > hb else ""
        logger.info("[%s L%d] actions=%s baseline=%s rhae=%.3f%s",
                    gid, lvl, best_len, hb, r, flag)
    return rows, corpus, improved


def _verify(solver, lvl, sol):
    try:
        return bool(solver.verify_solution(lvl, sol))
    except Exception:
        return False


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
            champ, promoted = evolve.evolve_step(
                os.path.join(HERE, "champion.json"),
                os.path.join(LOGDIR, "evolution_history.jsonl"),
                walls, cur_rhae, llm, eval_fn, games, heldout, n=4)
            probe_note = "live-probe" if probe_fn else "corpus-floor"
            summary_lines.append(
                f"- evolve: {'PROMOTED champion v'+str(champ.get('version')) if promoted else 'no promotion'} "
                f"(backend={llm.name}, {len(walls)} wall levels targeted, eval={probe_note})")
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
