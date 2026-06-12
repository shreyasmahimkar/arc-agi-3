# =====================================================================
# CHRONOS SOLVER V15 — PLM (Puzzle Language Model) agent
#
# = v14 harness wrapper over the v15 PLM (token-conditioned simulator;
# see plm/world_model.py for the measured v14 memorization failure the
# new architecture fixes). Same two tiers:
#
#   tier 1: PLM  — tokenize -> recursive belief -> goose explore /
#                  latent-BFS plan inside the learned world model
#   tier 2: numpy experience-bandit (v13's torch-free fallback) — used
#           when torch or PLM weights are unavailable, or the PLM errors
#
# INTEGRITY (decided in v13, kept here): no solution caches, no engine
# source loading at eval time. The local engines and v13 caches are used
# ONLY OFFLINE to train the PLM weights. At eval the agent learns from
# its own interactions — which is the point of the benchmark.
# =====================================================================
import hashlib
import logging
import os
import random
import sys
import time

import numpy as np

# make `plm` importable both locally (this dir) and on Kaggle, where the
# rerun cell copies my_agent.py AND the plm/ package side by side
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---- torch is OPTIONAL (same contract as v13): without it the PLM tier
# is disabled and the numpy bandit carries the run -------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

from arcengine import FrameData, GameAction, GameState, ActionInput  # noqa: E402

# [v13 carry-over] Python < 3.11: tuple-valued enum members register tuple
# keys in _value2member_map_, breaking GameAction(<int>) and deepcopy.
for _m in GameAction:
    GameAction._value2member_map_.setdefault(_m.value, _m)

# [v13 carry-over] the ARC-AGI-3-Agents package drags langgraph/langsmith
# in at import time; fall back to a minimal Agent base when unavailable.
try:
    from agents.agent import Agent
except Exception:
    class Agent:
        MAX_ACTIONS = 80
        action_counter = 0

        def __init__(s, card_id="", game_id="", agent_name="", ROOT_URL="",
                     record=False, arc_env=None, tags=None, **kw):
            s.card_id = card_id
            s.game_id = game_id
            s.agent_name = agent_name
            s.ROOT_URL = ROOT_URL
            s.record = record
            s.arc_env = arc_env
            s.tags = tags or []
            s.guid = ""
            s.frames = [FrameData(levels_completed=0)]
            s.timer = time.time()

logger = logging.getLogger(__name__)

# ---- PLM import is also survivable: a broken/missing package must never
# take the whole agent down on the eval server ----------------------------
PLM_AVAILABLE = False
if TORCH_AVAILABLE:
    try:
        from plm.agent_plm import PLMAgent, candidate_actions
        from plm.config import PLMConfig
        from plm import ttt
        PLM_AVAILABLE = True
    except Exception as e:
        logger.warning(f"PLM package unavailable ({e}) — bandit fallback only")

# ---- the THREE-PASS runtime (all inside this agent; the game is unknown) --
# pass 1 SCOUT: bandit explores for real, recording every transition
# pass 2 TRAIN: finetune the PLM on this game's own transitions (TTT)
# pass 3 PLAY:  PLM + deep-think planner; re-scout + retrain when stuck
SCOUT_ACTIONS = int(os.environ.get("V15_SCOUT_ACTIONS", 80))
RESCOUT_ACTIONS = int(os.environ.get("V15_RESCOUT_ACTIONS", 40))
TTT_SECONDS = float(os.environ.get("V15_TTT_SECONDS", 240))
STUCK_WINDOW = int(os.environ.get("V15_STUCK_WINDOW", 60))


class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time() * 1e6) + hash(s.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        s.start_time = time.time()
        s.cl = -1                       # current level
        s.la = 0                        # actions taken on this level
        s._eps = 0.15
        s._eps_min, s._eps_decay = 0.03, 0.9997
        # bandit fallback state (v13's torch-free tier)
        s._act_stats = {}               # action key -> (count, productive)
        s._noop_memory = set()          # (state_hash, action) known no-ops
        s.pr = None                     # previous raw frame
        s.pai = None                    # previous action key
        s.ph = None                     # previous state hash
        # PLM tier
        s._plm = None
        if PLM_AVAILABLE:
            try:
                s._plm = PLMAgent()
                if not s._plm.loaded and os.environ.get("V15_REQUIRE_WEIGHTS"):
                    s._plm = None       # untrained PLM is worse than bandit
            except Exception as e:
                logger.warning(f"PLM init failed: {e} — bandit fallback")
                s._plm = None
        # three-pass runtime state (game is UNKNOWN — everything below is
        # learned from this run's own interactions)
        s._phase = 'scout' if s._plm else 'bandit'
        s._scout_left = SCOUT_ACTIONS
        s._episodes = []                # closed episodes: (frames, acts, rews)
        s._ep_frames, s._ep_acts, s._ep_rews = [], [], []
        s._last_raw = None
        s._last_act3 = None             # (id,x,y) actually sent
        s._last_lvl = None
        s._since_progress = 0
        s._ttt_runs = 0
        logger.info(f"V15 agent ready: plm={'on' if s._plm else 'OFF'} "
                    f"torch={'yes' if TORCH_AVAILABLE else 'no'} "
                    f"phase={s._phase} (scout {SCOUT_ACTIONS} acts, "
                    f"ttt {TTT_SECONDS:.0f}s)")

    # ---------------- three-pass plumbing ----------------
    def _record(s, raw, lvl):
        """Bank the (prev_frame, action -> this_frame, reward) transition.
        Runs on EVERY step regardless of phase — the buffer is the game's
        only textbook."""
        if s._last_raw is not None and s._last_act3 is not None:
            rew = 1 if (s._last_lvl is not None and lvl > s._last_lvl) else 0
            if not s._ep_frames:
                s._ep_frames.append(s._last_raw.astype(np.uint8))
            s._ep_frames.append(raw.astype(np.uint8))
            s._ep_acts.append(s._last_act3)
            s._ep_rews.append(rew)
        s._last_raw = raw.copy()
        s._last_lvl = lvl

    def _close_episode(s):
        if len(s._ep_acts) > 0:
            s._episodes.append((np.stack(s._ep_frames),
                                np.asarray(s._ep_acts, np.int16),
                                np.asarray(s._ep_rews, np.uint8)))
        s._ep_frames, s._ep_acts, s._ep_rews = [], [], []
        s._last_raw = None
        s._last_act3 = None

    def _train_now(s, seconds):
        """Pass 2: finetune the PLM on everything recorded so far."""
        s._close_episode()
        eps = list(s._episodes)
        if not eps or s._plm is None:
            return
        logger.info(f"V15 TTT #{s._ttt_runs + 1}: training on "
                    f"{len(eps)} episodes "
                    f"({sum(len(e[1]) for e in eps)} transitions)...")
        try:
            stats = ttt.finetune(s._plm.tok, s._plm.belief_core, s._plm.sim,
                                 eps, s._plm.device, s._plm.cfg,
                                 seconds=seconds)
            s._ttt_runs += 1
            logger.info(f"V15 TTT done: {stats}")
            # model changed — let the goose re-measure, drop stale plans
            s._plm.goose.err = max(s._plm.goose.err, 0.5)
            s._plm.goose.hist = []
            s._plm.plan_queue = []
            s._plm.plan_misses = 0
        except Exception as e:
            logger.warning(f"V15 TTT failed ({e}) — continuing untrained")

    # ---------------- v13-pattern plumbing ----------------
    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid:
            s.guid = f.guid

    def _lvl(s, f):
        return getattr(f, 'score', None) or f.levels_completed

    def _raw(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def is_done(s, frames, lf):
        try:
            return lf.state is GameState.WIN or \
                (time.time() - s.start_time) >= 8 * 3600 - 300
        except Exception:
            return True

    # ---------------- main decision loop ----------------
    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                if s.cl >= 0:
                    logger.info(f"V15: level {s.cl} done in {s.la} actions")
                s.cl = lvl
                s.la = 0
                s._since_progress = 0
                s._act_stats = {}
                s._noop_memory = set()
                s.pr = None; s.pai = None; s.ph = None
                if s._plm:
                    # rules persist across levels in a game; layouts don't —
                    # the PLM keeps its belief, the goose re-verifies
                    s._plm.on_level_change()

            # ===== RESET states =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s._close_episode()      # bank what this life taught us
                if s._plm:
                    s._plm.reset_episode()
                a = GameAction.RESET
                a.reasoning = "reset"
                s._last_act3 = None
                return a

            raw = s._raw(lf)
            avail = getattr(lf, 'available_actions', None) or []
            avail_ids = {int(x.value) if hasattr(x, 'value') else int(x)
                         for x in avail}

            # ===== THREE-PASS RUNTIME =====
            s._record(raw, lvl)                       # every step feeds pass 2
            s._since_progress += 1

            # pass 1 -> pass 2 transition: scout budget exhausted
            if s._phase == 'scout' and s._scout_left <= 0:
                s._train_now(TTT_SECONDS)
                s._phase = 'plm'
            # pass 3 stuck -> brief re-scout, then retrain on the new data
            elif s._phase == 'plm' and s._since_progress > STUCK_WINDOW \
                    and s._plm is not None:
                logger.info(f"V15: no progress in {STUCK_WINDOW} actions — "
                            f"re-scouting {RESCOUT_ACTIONS}")
                s._phase = 'scout'
                s._scout_left = RESCOUT_ACTIONS
                s._since_progress = 0

            # ===== PASS 1: SCOUT (bandit explores, buffer records) =====
            if s._phase == 'scout' and s._plm is not None:
                s._scout_left -= 1
                sel = s._bandit_choose(raw, avail_ids)
                sel.reasoning = f"scout({s._scout_left} left):" \
                                + getattr(sel, 'reasoning', '')
                d = sel.get_data() if hasattr(sel, 'get_data') else None
                d = d if isinstance(d, dict) else {}
                s._last_act3 = (int(sel.value), int(d.get('x', 0)),
                                int(d.get('y', 0)))
                return sel

            # ===== PASS 3: PLM + deep think =====
            if s._plm is not None:
                try:
                    (aid, ax, ay), info = s._plm.step(raw, avail_ids)
                    sel = GameAction.from_id(aid)
                    if aid == 6:
                        sel.set_data({"x": int(ax), "y": int(ay)})
                    sel.reasoning = info
                    s.la += 1
                    s._last_act3 = (int(aid), int(ax), int(ay))
                    return sel
                except Exception as e:
                    logger.warning(f"PLM step failed ({e}) — bandit takes over")
                    s._plm = None       # don't retry a broken model every step

            # ===== FALLBACK: numpy experience-bandit (no torch / PLM dead) ==
            sel = s._bandit_choose(raw, avail_ids)
            d = sel.get_data() if hasattr(sel, 'get_data') else None
            d = d if isinstance(d, dict) else {}
            s._last_act3 = (int(sel.value), int(d.get('x', 0)),
                            int(d.get('y', 0)))
            return sel

        except Exception as e:
            import traceback
            traceback.print_exc()
            a = GameAction.from_id(random.choice([x for x in (1, 2, 3, 4, 5)]))
            a.reasoning = f"err:{e}"
            return a

    # ---------------- tier 2: the v13 bandit, compacted ----------------
    def _bandit_choose(s, raw, avail_ids):
        """UCB over available actions; never repeats a known no-op from the
        same state; clicks target small-object centroids."""
        ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]

        # learn from the previous transition (the only feedback we get)
        if s.pr is not None and s.pai is not None:
            changed = bool(np.any(s.pr != raw))
            n, w = s._act_stats.get(s.pai, (0, 0.0))
            s._act_stats[s.pai] = (n + 1, w + (1.0 if changed else 0.0))
            if not changed and s.ph is not None:
                s._noop_memory.add((s.ph, s.pai))

        cands = [(a, None) for a in sorted(avail_ids) if 1 <= a <= 5]
        if 6 in avail_ids:
            cnt = np.bincount(raw.flatten(), minlength=16)
            bg = int(cnt.argmax())
            targets = []
            for c in range(16):
                if c == bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys, xs = np.where(raw == c)
                if len(ys) >= 2:
                    targets.append((int(cnt[c]), int(np.median(ys)),
                                    int(np.median(xs))))
            targets.sort()
            for _, y, x in targets[:8]:
                cands.append((6, (y, x)))
        if not cands:
            a = GameAction.ACTION5 if 5 in avail_ids else GameAction.RESET
            a.reasoning = "np:none-avail"
            return a

        total_n = sum(n for n, _ in s._act_stats.values()) + 1
        scored = []
        for aid, coords in cands:
            key = aid if aid != 6 else (6, coords)
            n, w = s._act_stats.get(key, (0, 0.0))
            prod = (w / n) if n else 0.6          # optimistic prior
            ucb = prod + 0.8 * np.sqrt(np.log(total_n + 1) / (n + 1))
            if (ch, key) in s._noop_memory:
                ucb -= 10.0                        # hard-avoid known no-ops
            scored.append((ucb, aid, coords, key))
        scored.sort(reverse=True, key=lambda t: t[0])
        if random.random() < s._eps:
            _, aid, coords, key = random.choice(scored)
        else:
            _, aid, coords, key = scored[0]
        s._eps = max(s._eps_min, s._eps * s._eps_decay)

        if aid == 6:
            sel = GameAction.ACTION6
            y, x = coords
            sel.set_data({"x": int(x), "y": int(y)})
            sel.reasoning = f"np:c({x},{y})"
        else:
            sel = GameAction.from_id(aid)
            sel.reasoning = f"np:a{aid}"
        s.pai = key
        s.pr = raw.copy()
        s.ph = ch
        s.la += 1
        return sel
