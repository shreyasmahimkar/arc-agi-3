# =====================================================================
# CHRONOS SOLVER V14 — PLM (Puzzle Language Model) agent
#
# Follows the v13 harness pattern (Agent base, choose_action(frames, lf),
# GameAction compat patch, hardware auto-profile) but replaces the entire
# decision stack with the PLM world model:
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
#
# UNTESTED SKELETON — run `python -m plm.smoke` then play_game.py locally
# before trusting it.
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
        PLM_AVAILABLE = True
    except Exception as e:
        logger.warning(f"PLM package unavailable ({e}) — bandit fallback only")


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
                if not s._plm.loaded and os.environ.get("V14_REQUIRE_WEIGHTS"):
                    s._plm = None       # untrained PLM is worse than bandit
            except Exception as e:
                logger.warning(f"PLM init failed: {e} — bandit fallback")
                s._plm = None
        logger.info(f"V14 agent ready: plm={'on' if s._plm else 'OFF'} "
                    f"torch={'yes' if TORCH_AVAILABLE else 'no'}")

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
                    logger.info(f"V14: level {s.cl} done in {s.la} actions")
                s.cl = lvl
                s.la = 0
                s._act_stats = {}
                s._noop_memory = set()
                s.pr = None; s.pai = None; s.ph = None
                if s._plm:
                    # rules persist across levels in a game; layouts don't —
                    # the PLM keeps its belief, the goose re-verifies
                    s._plm.on_level_change()

            # ===== RESET states =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                if s._plm:
                    s._plm.reset_episode()
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

            raw = s._raw(lf)
            avail = getattr(lf, 'available_actions', None) or []
            avail_ids = {int(x.value) if hasattr(x, 'value') else int(x)
                         for x in avail}

            # ===== TIER 1: PLM =====
            if s._plm is not None:
                try:
                    (aid, ax, ay), info = s._plm.step(raw, avail_ids)
                    sel = GameAction.from_id(aid)
                    if aid == 6:
                        sel.set_data({"x": int(ax), "y": int(ay)})
                    sel.reasoning = info
                    s.la += 1
                    return sel
                except Exception as e:
                    logger.warning(f"PLM step failed ({e}) — bandit takes over")
                    s._plm = None       # don't retry a broken model every step

            # ===== TIER 2: numpy experience-bandit (v13 fallback) =====
            return s._bandit_choose(raw, avail_ids)

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
