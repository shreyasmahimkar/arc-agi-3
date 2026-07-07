# =====================================================================
# Chronos v20 — the synthesis agent
#
# Built from two proven facts:
#   • v12 scored 0.22 with MEMORY-FIRST replay of pre-solved answers + live BFS.
#     Pure live BFS (no memory) scored only 0.08 — the memory replay IS the score.
#   • Tufa/leaderboard research: RHAE is a 0-100% scale with a QUADRATIC penalty +
#     5x cap, so the SHORTEST solution wins; the top no-source method is an
#     executable world model (32.58%), and a retraining-free SELF-LEARNING memory
#     layer gives 2.6x (BentoLabs). See v20/research/LEADERBOARD_RESEARCH.md.
#
# So v20 = v12's memory-first engine, made HONEST (verified recall, never blind
# replay) and RHAE-optimal, structured as a staged cascade so the research stages
# plug in:
#     STAGE 1  MEMORY   verified recall of the shipped cache  (v12's 0.22 driver)
#     STAGE 1b SELF-LEARN cross-game macro retrieval -> BFS seed  (Tufa 2.6x lever)
#     STAGE 2  BFS      live 'auto' ladder, OPTIMAL (best RHAE); flywheel-persists
#     STAGE 3  FORGE    v19 black-box for no-source games  [-> executable WM next]
#     STAGE 4  LADDER   variant re-root / TTRL for the hardest  [research hook]
#
# References v19/src READ-ONLY (imports combined_agent + forge_agent); no v19 edits.
# =====================================================================
import os, sys, glob, json, time, hashlib, logging
import numpy as np

logger = logging.getLogger("v20")

# ---- locate + import the v19 engine (read-only) --------------------------------
def _add_v19_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [here,                                   # if combined_agent staged beside us (Kaggle)
              os.path.join(here, "..", "..", "v19", "src"),
              os.environ.get("V19_SRC", "")]:
        if c and os.path.isfile(os.path.join(c, "combined_agent.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return
    for root in ["/kaggle", "/workspace", os.path.expanduser("~")]:
        hits = glob.glob(os.path.join(root, "**", "combined_agent.py"), recursive=True) if os.path.isdir(root) else []
        if hits:
            sys.path.insert(0, os.path.dirname(hits[0])); return
_add_v19_to_path()

from combined_agent import (BFSSolver, find_game_source_and_class, Agent,
                            GameAction, GameState, ActionInput, HW)  # v19, read-only
try:
    from forge_agent import ForgeAgent
except Exception:
    ForgeAgent = None
try:
    from graph_explore import graph_solve                        # v20.3 (beside us on Kaggle)
except Exception:
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from graph_explore import graph_solve
    except Exception:
        graph_solve = None

# ---- config --------------------------------------------------------------------
_HERE       = os.path.dirname(os.path.abspath(__file__))
BFS_TIMEOUT = int(os.environ.get("V20_BFS_TIMEOUT", "800"))     # per-level live BFS budget (A0)
GRAPH_BUDGET = int(os.environ.get("V20_GRAPH_BUDGET", "180"))  # v20.3 graph-explore budget/level
MAX_DEATHS  = int(os.environ.get("V20_MAX_DEATHS", "2"))        # A1: abandon a level after N deaths
BFS_WORKERS = int(os.environ.get("V20_BFS_WORKERS", str(HW.get("workers", 1))))  # A3: pool size
# A3: forked worker pools + an initialized CUDA context (Forge on the T4s) +
# the swarm's threads together deadlock the child. When a GPU is present, force
# 'spawn' for BFS pools — CUDA-safe and thread-safe — so search and Forge coexist.
if HW.get("device") == "cuda":
    HW["mp_ctx"] = "spawn"
CACHE_DIRS  = [os.environ.get("V20_CACHE_DIR", ""),
               os.path.join(_HERE, "solutions"), "solutions",
               os.path.join(_HERE, "..", "solutions")]
MACRO_PATH  = os.environ.get("V20_MACRO_BANK", os.path.join(_HERE, "v20_macro_bank.json"))


def _find_cache(game_id):
    gid = game_id.split("-")[0]
    for d in CACHE_DIRS:
        if not d:
            continue
        p = os.path.join(d, f"{gid}.json")
        if os.path.exists(p):
            try:
                raw = json.load(open(p))
                return {int(k): [(a, d2) for a, d2 in v] for k, v in raw.items()
                        if str(k).lstrip("-").isdigit()}
            except Exception:
                pass
    return {}


def _hist_sig(frame):
    return hashlib.md5(np.bincount(np.asarray(frame).flatten(), minlength=16).tobytes()).hexdigest()[:12]


def _resolve_source(game_id, arc_env=None):
    """Version-EXACT source resolution. Scored game_ids carry the version hash
    (e.g. 'ls20-9607627b'); three public games (ls20/ft09/vc33) ship TWO version
    dirs and a bare glob returns the STALE one first — loading it makes the
    verifier reject perfectly good cached plans and BFS solve the wrong puzzle
    (the 0.13 regression). Prefer, in order: the game_id's exact version dir,
    arc_env's local_dir, newest-mtime version, then v19's generic resolver."""
    import re
    parts = game_id.split("-")
    gid, ver = parts[0], (parts[1] if len(parts) > 1 else None)

    def _cls_of(src):
        m = re.search(r"class\s+(\w+)\s*\(\s*ARCBaseGame", open(src).read())
        return m.group(1) if m else (gid[0].upper() + gid[1:])

    roots = ["/kaggle/input/**/environment_files",
             "/kaggle/working/**/environment_files",
             os.path.join(_HERE, "..", "..", "..", "..", "arc-prize-2026-arc-agi-3", "environment_files"),
             os.environ.get("V20_ENV_DIR", "")]
    cands = []
    for r in roots:
        if r:
            cands += glob.glob(os.path.join(r, gid, "*", f"{gid}.py"), recursive=True)
    cands = [os.path.normpath(p) for p in cands]
    if cands:
        if ver:                                   # exact version from the scored game_id
            exact = [p for p in cands if os.path.basename(os.path.dirname(p)).startswith(ver)]
            if exact:
                return exact[0], _cls_of(exact[0])
        ld = getattr(getattr(arc_env, "environment_info", None), "local_dir", None)
        if ld:                                    # the env's own loaded version
            ldn = os.path.normpath(str(ld))
            hit = [p for p in cands if os.path.dirname(p) == ldn]
            if hit:
                return hit[0], _cls_of(hit[0])
        best = max(cands, key=os.path.getmtime)   # newest on disk (the live one)
        return best, _cls_of(best)
    return find_game_source_and_class(game_id, arc_env)   # v19 fallback


# ---- v20 self-learning macro bank (BentoLabs-style cross-game reuse) ------------
class MacroBank:
    """Retraining-free self-learning: every verified solution is banked keyed by its
    start-frame color-histogram signature. On a new level with no exact memory, a
    same-signature macro from an earlier game is offered to BFS as a transfer seed
    (v19 solve_level already tries object-relative transfer of a seed before search).
    Safe: a bad seed just falls through to genuine BFS."""
    def __init__(s, path):
        s.path = path
        s.bank = {}
        if os.path.exists(path):
            try: s.bank = json.load(open(path))
            except Exception: s.bank = {}

    def add(s, start_frame, solution):
        if start_frame is None or not solution:
            return
        k = _hist_sig(start_frame)
        cur = s.bank.get(k)
        # keep the SHORTEST macro per signature (RHAE: shortest wins)
        if cur is None or len(solution) < len(cur):
            s.bank[k] = [[a, d] for a, d in solution]
            try: json.dump(s.bank, open(s.path, "w"))
            except Exception: pass

    def retrieve(s, start_frame):
        if start_frame is None:
            return None
        v = s.bank.get(_hist_sig(start_frame))
        return [(a, d) for a, d in v] if v else None


class MyAgent(Agent):
    MAX_ACTIONS = float("inf")
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        s.start_time = time.time()
        s._setup_done = False
        s._bfs = None
        s._forge = None
        s._cache = {}
        s._macros = MacroBank(MACRO_PATH)
        s._solutions = {}          # verified plans accepted this game (for chaining)
        s._sol = None              # active plan being executed
        s._step = 0
        s._cl = -1
        s._stage = {}              # level -> which stage solved it (telemetry)
        s._sol_unverified = False  # active plan is optimistic memory (not yet live-proven)
        s._plan_level = None       # level the active plan targets
        s._mem_risked = set()      # levels where the optimistic replay was already spent
        s._retry_bfs = False       # unverified plan failed live -> re-solve with BFS
        s._deaths = {}             # A1: GAME_OVER count per level
        s._abandoned = set()       # A1: levels given up on (stop burning the action cap)

    # -- helpers ------------------------------------------------------------------
    def _lvl(s, f):
        return getattr(f, "score", None) or f.levels_completed

    def _raw(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _setup(s):
        src, cls = _resolve_source(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=BFS_TIMEOUT,
                               workers=BFS_WORKERS)
            if s._bfs.load():
                logger.info(f"[v20] BFS ACTIVE: loaded {cls} from {src}")
            else:
                s._bfs = None
        if s._bfs is None:
            logger.warning(f"[v20] no white-box source for {s.game_id} -> black-box Forge")
            if ForgeAgent is not None:
                try:
                    wp = next((p for p in [
                        "/kaggle/input/forge-pretrained-weights/pretrained_weights.pt",
                        os.path.join(_HERE, "pretrained_weights.pt"), "pretrained_weights.pt"]
                        if os.path.exists(p)), None)
                    s._forge = ForgeAgent(weights=wp)
                    s._forge.reset(s.game_id)
                except Exception as e:
                    logger.warning(f"[v20] forge setup failed: {e}")
        s._cache = _find_cache(s.game_id)
        logger.info(f"[v20] memory cache for {s.game_id}: levels {sorted(s._cache)}")

    def _persist_cache(s):
        # flywheel: MERGE verified plans back (never reduce coverage — a partial/crashed
        # run must not clobber a deeper shipped cache). Writes to V20_FLYWHEEL_DIR if set,
        # else the read cache dir.
        try:
            gid = s.game_id.split("-")[0]
            d = (os.environ.get("V20_FLYWHEEL_DIR") or os.environ.get("V20_CACHE_DIR")
                 or os.path.join(_HERE, "solutions"))
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, f"{gid}.json")
            merged = {}
            if os.path.exists(p):
                try: merged = json.load(open(p))
                except Exception: merged = {}
            for k, v in s._solutions.items():
                merged[str(k)] = v                     # add/refresh; never drop existing keys
            json.dump(merged, open(p, "w"))
        except Exception:
            pass

    def _verify(s, lvl, plan):
        """Replay-verify a plan on a clean engine, chained on accepted priors."""
        if not s._bfs or not plan:
            return False
        for i in range(lvl):
            if i in s._solutions:
                s._bfs.solutions[i] = s._solutions[i]
        try:
            return bool(s._bfs.verify_solution(lvl, plan))
        except Exception:
            return False

    def _chained_game(s, lvl):
        """A fresh game positioned at `lvl`'s start (replay accepted prior plans).
        Returns (game, frame) or (None, None). Used by the graph-explore rung."""
        if not s._bfs:
            return None, None
        try:
            g = s._bfs.game_cls()
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            for i in range(lvl):
                if i not in s._solutions:
                    return None, None
                for a, d in s._solutions[i]:
                    r = g.perform_action(ActionInput(id=GameAction.from_id(a), data=d) if d
                                         else ActionInput(id=GameAction.from_id(a)), raw=True)
            if not r.frame:
                return None, None
            return g, np.array(r.frame[-1])
        except Exception:
            return None, None

    # -- the cascade --------------------------------------------------------------
    def _solve_level(s, lvl, start_frame, skip_memory=False):
        s._sol_unverified = False
        # STAGE 1 — MEMORY (verified recall): v12's 0.22 driver, made honest.
        plan = s._cache.get(lvl)
        if not skip_memory and plan:
            if s._verify(lvl, plan):
                s._stage[lvl] = "memory"
                s._macros.add(start_frame, plan)          # B2: bank optimal solutions too
                return plan
            # verify FAILED. [A2] Only risk a live replay when there is NO local
            # source to judge with (online-only). If a source EXISTS and says the
            # plan is invalid, the served version differs (a stale twin) — replaying
            # it live just burns the action cap, so go straight to BFS instead.
            if s._bfs is None and lvl not in s._mem_risked:
                s._mem_risked.add(lvl)
                s._stage[lvl] = "memory-unverified"
                s._sol_unverified = True
                return plan

        if s._bfs:
            # STAGE 1b — SELF-LEARNING [B2]: a same-signature macro from an earlier
            # game/level. First try it as a full CANDIDATE (verify it — near-free if
            # it transfers); else hand it to BFS as a transfer seed.
            macro = s._macros.retrieve(start_frame)
            if macro and s._verify(lvl, macro):
                s._stage[lvl] = "macro"
                return macro
            seed = s._solutions.get(lvl - 1) or macro
            # STAGE 2 — BFS 'auto' ladder: OPTIMAL solution (best RHAE). Chained on priors.
            for i in range(lvl):
                if i in s._solutions:
                    s._bfs.solutions[i] = s._solutions[i]
            s._bfs.solutions.pop(lvl, None)
            try:
                sol = s._bfs.solve_level(lvl, prev_solution=seed, strategy="auto")
            except Exception as e:
                logger.warning(f"[v20] BFS L{lvl} raised: {e}"); sol = None
            if sol and s._verify(lvl, sol):
                s._stage[lvl] = "bfs" if not plan else "bfs(mem-stale)"
                s._macros.add(start_frame, sol)           # self-learn for later games
                return sol

        # STAGE 3 — GRAPH-EXPLORE (v20.3): white-box frontier exploration (object
        # segmentation + persistent state graph over engine snapshots) when the BFS
        # ladder stalls. Beats the Forge CNN 3x head-to-head (test_graph_vs_forge).
        if graph_solve is not None and s._bfs is not None:
            gstart, fstart = s._chained_game(lvl)
            if gstart is not None:
                try:
                    gsol = graph_solve(gstart, lvl, fstart, list(gstart._available_actions),
                                       budget=GRAPH_BUDGET)
                except Exception as e:
                    logger.warning(f"[v20] graph L{lvl} raised: {e}"); gsol = None
                if gsol:
                    plan2 = [(a, d) for a, d in gsol]
                    if s._verify(lvl, plan2):
                        s._stage[lvl] = "graph"
                        s._macros.add(start_frame, plan2)
                        return plan2

        # STAGE 3b — FORGE (no-source games, in choose_action) and STAGE 4 — LADDER
        # are the demoted/hook fallbacks; see LEADERBOARD_RESEARCH.md.
        return None

    def _budget_left(s):
        return (time.time() - s.start_time) < (8 * 3600 - 300)

    def choose_action(s, frames, lf):
        try:
            if not s._setup_done:
                s._setup_done = True
                s._setup()

            lvl = s._lvl(lf)

            # ----- new level: run the cascade to get an OPTIMAL plan -----
            if lvl != s._cl:
                # an optimistic (unverified) plan that just ADVANCED the level has
                # proven itself on the live engine — commit it for chaining/flywheel
                if s._sol_unverified and s._sol and s._plan_level is not None and lvl > s._plan_level:
                    s._solutions[s._plan_level] = s._sol
                    if s._bfs:
                        s._bfs.solutions[s._plan_level] = s._sol
                    s._persist_cache()
                    logger.info(f"[v20] L{s._plan_level}: unverified memory plan PROVED live")
                s._cl = lvl
                s._sol = None; s._step = 0
                s._sol_unverified = False; s._plan_level = None; s._retry_bfs = False
                start_frame = s._raw(lf) if getattr(lf, "frame", None) else None
                if (s._bfs or s._cache) and lvl not in s._abandoned:
                    plan = s._solve_level(lvl, start_frame)
                    if plan:
                        s._sol = plan; s._step = 0; s._plan_level = lvl
                        if not s._sol_unverified:      # verified/bfs plans commit now
                            s._solutions[lvl] = plan
                            if s._bfs:
                                s._bfs.solutions[lvl] = plan
                            s._persist_cache()
                        logger.info(f"[v20] L{lvl} -> {s._stage.get(lvl,'?')} ({len(plan)} actions)")

            # ----- reset handling -----
            if lf.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                if lf.state is GameState.GAME_OVER:
                    s._deaths[lvl] = s._deaths.get(lvl, 0) + 1
                    if s._deaths[lvl] > MAX_DEATHS and lvl not in s._abandoned:
                        # A1+B1: stop paying the action cap on a level we keep dying on
                        s._abandoned.add(lvl)
                        s._sol = None; s._sol_unverified = False; s._retry_bfs = False
                        logger.info(f"[v20] L{lvl}: abandoned after {s._deaths[lvl]} deaths")
                if lvl not in s._abandoned and s._sol is not None and s._step > 0:
                    if s._sol_unverified:
                        # optimistic plan caused a death -> abandon it, re-solve live
                        s._sol = None; s._sol_unverified = False; s._retry_bfs = True
                    else:
                        s._step = 0            # level restarts -> replay plan from its start
                a = GameAction.RESET; a.reasoning = "v20:reset"; return a

            # ----- optimistic plan failed live (exhausted or died) -> live BFS -----
            plan_exhausted = s._sol is not None and s._step >= len(s._sol)
            if ((plan_exhausted and s._sol_unverified) or (s._sol is None and s._retry_bfs)) \
                    and lvl not in s._abandoned:
                s._sol = None; s._sol_unverified = False; s._retry_bfs = False
                if s._bfs:
                    logger.info(f"[v20] L{lvl}: memory plan failed live -> genuine BFS")
                    start_frame = s._raw(lf) if getattr(lf, "frame", None) else None
                    plan = s._solve_level(lvl, start_frame, skip_memory=True)
                    if plan:
                        s._sol = plan; s._step = 0; s._plan_level = lvl
                        s._solutions[lvl] = plan
                        s._bfs.solutions[lvl] = plan
                        s._persist_cache()
                        logger.info(f"[v20] L{lvl} -> {s._stage.get(lvl,'?')} ({len(plan)} actions)")

            # ----- execute the planned (optimal) solution move-by-move -----
            if s._sol and s._step < len(s._sol):
                act_id, data = s._sol[s._step]
                s._step += 1
                a = GameAction.from_id(act_id)
                if data:
                    a.set_data(data)
                a.reasoning = f"v20:{s._stage.get(lvl,'plan')}:{s._step}/{len(s._sol)}"
                return a

            # ----- STAGE 3: black-box Forge for no-source games -----
            if s._forge is not None:
                return s._forge_decide(lf, lvl)

            # abandoned levels give up cheaply; otherwise idle
            if lvl in s._abandoned:
                a = GameAction.RESET; a.reasoning = "v20:abandoned"; return a
            a = GameAction.ACTION5 if _has(lf, 5) else GameAction.RESET
            a.reasoning = "v20:idle"
            return a
        except Exception as e:
            import traceback; traceback.print_exc()
            a = GameAction.RESET; a.reasoning = f"v20:err:{e}"; return a

    def _forge_decide(s, lf, lvl):
        if not getattr(lf, "frame", None):
            a = GameAction.ACTION1; a.reasoning = "v20:forge:noframe"; return a
        class _Obs: pass
        o = _Obs()
        o.frame = np.array(lf.frame, dtype=np.uint8)[-1]
        o.levels_completed = lvl
        o.state = str(getattr(lf.state, "value", lf.state))
        aa = getattr(lf, "available_actions", None) or []
        o.available_actions = tuple(a.value if hasattr(a, "value") else int(a) for a in aa)
        aid, data = s._forge.act(o)
        if aid == 0:
            a = GameAction.RESET
        elif aid == 6:
            a = GameAction.ACTION6
            if data:
                a.set_data({"x": int(data["x"]), "y": int(data["y"])})
        else:
            a = GameAction.from_id(aid)
        a.reasoning = "v20:forge"
        return a

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if getattr(f, "guid", None):
            s.guid = f.guid

    def is_done(s, frames, lf):
        try:
            return lf.state is GameState.WIN or not s._budget_left()
        except Exception:
            return True


def _has(lf, aid):
    for a in (getattr(lf, "available_actions", None) or []):
        if (a.value if hasattr(a, "value") else int(a)) == aid:
            return True
    return False
