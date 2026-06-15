# =====================================================================
# chronos_solver v13_3 — v13_2 base + partial-observability fixes (see RESEARCH.md)
#
# New in v13_3:
#   1. REVEAL-NOVELTY ATOMS in IW: pixels that transition from background
#      to non-background are grouped into 4×4 grid-cell regions and added
#      as explicit novelty atoms ('rev', region_id). IW now KEEPS exploratory
#      moves that uncover hidden areas instead of pruning them.
#   2. REACTIVE-BLOCK ATOMS: non-player pixels that change between parent
#      and child frames are tracked as coarse ('rb', grid_y, grid_x, color)
#      atoms (one per 8×8 cell+color) instead of 16 per-pixel atoms per
#      rotation step. Keeps IW width tight for ls20-style rotating blocks.
#   3. SENSING PREPASS rung: short BFS that maximises newly-revealed pixels,
#      then immediately retries IW(1) from each revelation checkpoint.
#      Finds solutions in partially-observable levels where the clean-start
#      reachable space exhausts without a win.
#
# ===================== inherited v13_2 header ========================
# New in v13_2:
#   1. ANYTIME INCUMBENT TIGHTENING (ARA*-flavored): the first verified
#      solution from any non-optimal rung becomes an upper bound L; with
#      leftover budget, exact masked BFS reruns with max_depth = L-1 —
#      finds a strictly shorter solution or proves L optimal-in-model.
#      (v13's 600s ls20 L4 banked greedy's 78 actions; 44 existed.)
#   2. ENFORCED HILL CLIMBING rung (FF-style): plateau BFS until the first
#      state with a never-seen progress signature, COMMIT, restart from it.
#      Waypoint decomposition with subgoals discovered by search instead of
#      guessed from centroids. Incomplete -> replay-verified + ladder
#      fallback (exactly FF's EHC -> best-first fallback).
#   3. `max_depth` plumbed through exact BFS for the bounded passes.
#
# ===================== inherited v13_1 header ========================
# chronos_solver v13_1 — v13 base + space-shrinking search rungs
#
# New in v13_1 (see RESEARCH.md):
#   1. WAYPOINT/TSP decomposition: macro-search over object-centroid
#      orderings (tiny TSP), A* legs between waypoints; click games get a
#      centroid-click-only macro BFS (branching = #objects, not #scanned px).
#   2. IW(1)/IW(2) novelty pruning (Lipovetzky & Geffner): a child is kept
#      only if it makes some atom (cell=color, or scalar attr=value) true
#      for the FIRST time in the whole search. Linear in #atoms.
#   3. A* rung: priority depth + w*manhattan(player, goal)/step — player
#      auto-detected (centroid moves under directional actions), goal =
#      rarest non-player object. Falls back to h=0 (=BFS) when undetectable.
#   4. Dominance pruning: prune states whose (player-cell, histogram,
#      scalar attrs) were already seen at <= depth (aggressive rungs only).
#   5. strategy='auto' retry ladder: bfs -> waypoint -> astar -> iw1 ->
#      iw2 -> greedy (budget fractions), then v13's unmasked/hidden retry.
#      All non-exact rung solutions are VERIFIED by replay before caching.
#   6. Workers additionally return the child's public scalar state, so the
#      parent can compute novelty/dominance/h without restoring snapshots.
#
# ====================== inherited v13 header =========================
# FORGE v19 — v18 base + 4 targeted bug fixes   (chronos_solver v13 base)
#
# Fixes applied on top of v18:
#
# FIX 1: _visited_hashes was never initialized in __init__ — reward
#         signal was broken: always gave +1.5 for ANY hash change,
#         never penalizing loops. Now properly tracks and deduplicates.
#
# FIX 2: CLTI frame extraction used get_pixels() which is inconsistent
#         with _raw() (which reads frame[-1] from perform_action).
#         Now uses perform_action result frames throughout, so injected
#         expert demos have correct state representations.
#
# FIX 3: BFS hidden retry used 3 RESET calls instead of 2, landing
#         in a different initial state than the first pass scan,
#         causing the retry to search from a mismatched baseline.
#
# FIX 4: Epsilon always reset to 0.15 on level change even when BFS
#         already solved the level. Now only resets if BFS failed,
#         preserving learned exploration for CNN fallback.
#
# [v13-compat] additions (sandbox/local portability, no behavior change
# on Kaggle where torch + agents package exist):
#   - torch import is optional; numpy bandit fallback when missing
#   - GameAction int-key registration for Python < 3.11
#   - minimal Agent base fallback when `agents` package can't import
# =====================================================================
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import pickle
import random
import sys
import time
import traceback
import zlib
from collections import deque
import numpy as np

# [v13-compat] torch is OPTIONAL — CNN fallback activates only when available
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    class _NNStub:
        class Module:
            def __init__(self, *a, **kw): raise RuntimeError("torch unavailable")
    nn = _NNStub()

from arcengine import FrameData, GameAction, GameState, ActionInput

# [v13-compat] Python < 3.11: GameAction members are declared with tuple values
# and reassign _value_ in __init__; older Pythons keep tuple keys in
# _value2member_map_, breaking GameAction(<int>) and copy.deepcopy of game
# states (which BFSSolver depends on). Harmless on 3.12+.
for _m in GameAction:
    GameAction._value2member_map_.setdefault(_m.value, _m)

# [v13-compat] the ARC-AGI-3-Agents `agents` package drags in langgraph/langsmith
# at import time. Fall back to a minimal compatible Agent base when unavailable.
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

# ==================== [v13] HARDWARE AUTO-PROFILE ====================
# Build locally (M1 Pro / mps), deploy on RTX 6000 (cuda) or Kaggle T4s —
# one codebase, no edits. BFS workers scale with CPU cores everywhere.
import platform
import multiprocessing as _mp

def get_hw_profile():
    p = {'mode': 'CPU', 'device': 'cpu',
         'workers': max(1, _mp.cpu_count() - 1),
         'bsz': 64, 'buf_size': 50_000, 'tfreq': 10, 'net_mult': 1,
         'compile': False, 'amp': False,
         'mp_ctx': 'spawn' if platform.system() == 'Darwin' else 'fork'}
    if not TORCH_AVAILABLE:
        return p
    if torch.cuda.is_available():
        p['device'] = 'cuda'
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except Exception:
            vram_gb = 0
        p['vram_gb'] = round(vram_gb, 1)
        if vram_gb > 30:
            # --- BEAST MODE: RTX PRO 6000 Blackwell (96GB) or
            #     RTX 6000 Ada (48GB) — detection is vram > 30GB either way.
            #     Batch scales with VRAM; the _train OOM backoff adapts
            #     further at runtime if needed.
            p.update(mode='RTX_6000',
                     workers=max(1, _mp.cpu_count() - 1),
                     bsz=2048 if vram_gb > 80 else 1024,
                     buf_size=2_000_000, tfreq=2, net_mult=4,
                     compile=hasattr(torch, 'compile'), amp=True)
        else:
            # --- CLOUD: STANDARD GPU (T4 16GB / Kaggle 2xT4) ---
            p.update(mode='CUDA_STD',
                     workers=max(1, _mp.cpu_count() - 2),
                     bsz=128, buf_size=100_000, tfreq=5, net_mult=1, amp=True)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # --- LOCAL: APPLE SILICON (M1 Pro, 32GB unified) ---
        p.update(mode='M1_PRO', device='mps',
                 workers=8,          # 8 performance cores drive the BFS
                 bsz=128, buf_size=100_000, tfreq=5, net_mult=1,
                 amp=False,          # MPS prefers native FP32 for stability
                 mp_ctx='spawn')     # CRITICAL: fork deadlocks torch on macOS
    return p

HW = get_hw_profile()
logger.info(f"[v13] hardware profile: {HW['mode']} device={HW['device']} "
            f"vram={HW.get('vram_gb', 'n/a')}GB workers={HW['workers']} "
            f"bsz={HW['bsz']} mp_ctx={HW['mp_ctx']}")

# ==================== [v13] PARALLEL BFS WORKERS ====================
_BFS_W = {}

def _scalar_state(g):
    """[v13] Public scalar attrs of the game object (key state, countdown
    timers, player coords). Folding these into the state hash is essential:
    e.g. ls20's lock-opening countdown produces pixel-identical frames that
    differ only in a hidden counter — without this, BFS prunes the countdown
    chain as 'visited' and the win is unreachable."""
    out = []
    for k, v in g.__dict__.items():
        if k.startswith('_'):
            continue
        if isinstance(v, (bool, int)):
            out.append((k, int(v)))
    return tuple(sorted(out))

def _bfs_worker_init(game_path, hash_mask, sig_mask, hidden_fields):
    """Pool initializer: load the game module so snapshots unpickle."""
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location('game_mod', game_path)
    mod = _ilu.module_from_spec(spec)
    sys.modules['game_mod'] = mod
    spec.loader.exec_module(mod)
    _BFS_W['mask'] = hash_mask      # excluded from the dedup hash
    _BFS_W['sig_mask'] = sig_mask   # excluded from the progress histogram
    _BFS_W['hidden'] = hidden_fields

def _bfs_expand_task(args):
    """Worker: restore snapshot, apply one action, hash result."""
    snap, act_id, data, level_idx = args
    try:
        g = BFSSolver._restore(snap)
        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
        r = g.perform_action(ai, raw=True)
        if not r.frame:
            return None
        f = np.array(r.frame[-1])
        mask = _BFS_W.get('mask')
        fm = f
        if mask is not None:
            fm = f.copy(); fm[mask] = 0
        h = hashlib.md5(fm.tobytes() + repr(_scalar_state(g)).encode()).hexdigest()[:16]
        hidden = _BFS_W.get('hidden')
        if hidden:
            extras = []
            for k in hidden:
                v = getattr(g, k, None)
                if v is not None:
                    extras.append(f"{k}={v}")
            if extras:
                h = h + "|" + "|".join(extras)
        win = bool(r.levels_completed > level_idx or g._current_level_index > level_idx)
        sig_mask = _BFS_W.get('sig_mask')
        fs = f
        if sig_mask is not None:
            fs = f.copy(); fs[sig_mask] = 0
        sig = tuple(np.bincount(fs.flatten(), minlength=16).tolist())
        # [v13_1] also ship the child's public scalar state so the parent
        # can compute novelty atoms / dominance keys / heuristics without
        # restoring the snapshot.
        return (h, win, BFSSolver._snap(g), sig,
                f.astype(np.uint8).tobytes(), f.shape, _scalar_state(g))
    except Exception:
        return None

def _dyn_clicks(frame, limit=12):
    """[v13] dynamic click targets from the CURRENT frame: centroids of
    non-background color regions. Selection/toggle games create new
    clickable objects as the board changes — a click list scanned once at
    the root cannot reach them."""
    try:
        cnt = np.bincount(frame.flatten(), minlength=16)
        bg = int(cnt.argmax())
        out = []
        for c in range(16):
            if c == bg or cnt[c] == 0 or cnt[c] > frame.size // 2:
                continue
            ys, xs = np.where(frame == c)
            out.append((int(cnt[c]), int(np.median(xs)), int(np.median(ys))))
        out.sort()
        return [(6, {'x': x, 'y': y, 'game_id': 'bfs'}) for _, x, y in out[:limit]]
    except Exception:
        return []

def _frame_objs(frame, bg=None, min_px=2, max_frac=0.5):
    """[v13_1] color-blob objects of a frame: [(color, cx, cy, npix), ...].
    Same notion of 'object' as _dyn_clicks (per-color median centroid)."""
    try:
        cnt = np.bincount(frame.flatten(), minlength=16)
        if bg is None:
            bg = int(cnt.argmax())
        out = []
        for c in range(16):
            if c == bg or cnt[c] < min_px or cnt[c] > frame.size * max_frac:
                continue
            ys, xs = np.where(frame == c)
            out.append((int(c), float(np.median(xs)), float(np.median(ys)),
                        int(cnt[c])))
        return out
    except Exception:
        return []


def _bfs_expand_node(args):
    """[v13-speed] Worker: restore ONE snapshot, apply ALL actions.
    Ships each snapshot to the pool once instead of once per action.
    When `dyn` is set and the node frame is provided, click targets are
    augmented from the current frame's object centroids."""
    snap, actions, level_idx, fbytes, fshape, dyn = args
    acts = list(actions)
    if dyn and fbytes is not None:
        frame = np.frombuffer(fbytes, dtype=np.uint8).reshape(fshape)
        seen = {(d.get('x'), d.get('y')) for a, d in acts if a == 6 and d}
        for a, d in _dyn_clicks(frame):
            if (d['x'], d['y']) not in seen:
                acts.append((a, d))
    out = []
    for act_id, data in acts:
        res = _bfs_expand_task((snap, act_id, data, level_idx))
        out.append((act_id, data, res))
    return out

# ==================== BFS SOLVER ====================
class BFSSolver:
    """Offline BFS solver using direct game class instantiation."""
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120,
                 workers=1):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.workers = workers  # [v13] >1 enables multiprocess node expansion
        self.game_cls = None
        self.solutions = {}  # level_idx → action list
        self.level_stats = {}  # [v13_1] level_idx → per-rung search stats
    def load(self):
        """Load the game class from source."""
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            # [v13-speed] register module so game objects are picklable —
            # pickle snapshots are ~2x faster than copy.deepcopy
            sys.modules['game_mod'] = mod
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    # [v13] transient pixel detection — pixels that change for EVERY action
    # (timer bars, step counters) cause exponential state aliasing in BFS.
    def _detect_transient(self, game, f0, actions):
        try:
            if len(actions) < 2:
                return None
            base = self._snap(game)
            # (a) single-step intersection across DIFFERENT actions:
            # pixels that change no matter what you do = pure time artifacts
            inter = None
            for act_id, data in actions[:6]:
                g = self._restore(base)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                except:
                    continue
                if not r.frame:
                    continue
                d = (np.array(r.frame[-1]) != f0)
                if d.any():
                    inter = d if inter is None else (inter & d)
            # (b) per-action rollouts: a row only counts as transient (timer
            # bar / HUD animation) if it stays hot under EVERY single-action
            # rollout. The player's activity band can't stay hot in all
            # directions, so this won't swallow real game state.
            hot_sets = []
            H = f0.shape[0]
            for act_id, data in actions[:4]:
                g = self._restore(base)
                prev = f0
                row_hits = np.zeros(H, dtype=np.int32)
                steps = 0
                for _ in range(8):
                    try:
                        ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                        r = g.perform_action(ai, raw=True)
                    except:
                        break
                    if not r.frame:
                        break
                    f = np.array(r.frame[-1])
                    d = (f != prev)
                    if d.any():
                        rows = np.unique(np.where(d)[0])
                        row_hits[rows] += 1
                        steps += 1
                    prev = f
                if steps >= 4:
                    hot_sets.append(set(np.where(row_hits >= int(steps * 0.75))[0].tolist()))
            mask = np.zeros(f0.shape, dtype=bool)
            if inter is not None:
                mask |= inter
            if len(hot_sets) >= 2:
                hot_rows = set.intersection(*hot_sets)
                if 0 < len(hot_rows) <= 8:
                    mask[sorted(hot_rows), :] = True
            n = int(mask.sum())
            if 0 < n <= max(2, f0.size // 5):  # sanity cap: don't mask real puzzle content
                logger.info(f"BFS: transient mask covers {n} px / rows {sorted(set(np.where(mask.any(axis=1))[0].tolist()))}")
                return mask
            return None
        except Exception:
            return None

    # [v13-speed] engine state snapshots: pickle (fast) with deepcopy fallback
    @staticmethod
    def _snap(g):
        try:
            return ('p', zlib.compress(pickle.dumps(g, -1), 1))
        except Exception:
            return ('d', copy.deepcopy(g))

    @staticmethod
    def _restore(snap):
        kind, payload = snap
        if kind == 'p':
            return pickle.loads(zlib.decompress(payload))
        return copy.deepcopy(payload)
    def _state_hash(self, g, frame, hidden_fields=None, mask=None):
        """Hash frame + discovered hidden scalar fields (fast).
        [v13] `mask` marks transient pixels (timers/HUD) excluded from the
        hash so they don't explode the BFS state space."""
        if mask is not None:
            frame = frame.copy()
            frame[mask] = 0
        fh = hashlib.md5(frame.tobytes() + repr(_scalar_state(g)).encode()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(g, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh
    def _probe_hidden_fields(self, game, actions):
        """Dynamic state probing — discover which scalar fields change per action.
        Returns list of field names that are hidden state (change without pixel change)."""
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v
        changing_fields = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            f = g.get_pixels(0, 0, 64, 64)
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)
        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)
    def _scan_actions(self, game, f0, bg):
        """Scan for effective actions. Returns list of (action_id, data)."""
        avail = game._available_actions
        actions = []
        base = self._snap(game)  # [v13-speed] snapshot once, restore per probe
        # Directional/interact actions
        deferred = []
        for a in [a for a in avail if a <= 5]:
            g = self._restore(base)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
                else:
                    # [v13] do NOT prune state-dependent actions: an interact
                    # button can be a no-op at spawn yet required later
                    # (stand on switch → press). Keep it, ordered last.
                    deferred.append((a, None))
            except:
                pass
        actions.extend(deferred)
        # Click actions ([v13] shape-aware: grids are not always 64x64).
        # Two passes: non-background pixels first, then background pixels —
        # selection-style games (e.g. vc33) take clicks on empty cells.
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
            H, W = f0.shape[:2]
            step = 2 if max(H, W) > 32 else 1
            for probe_bg in (False, True):
                if time.time() - t0 > self.scan_timeout:
                    break
                for y in range(0, H, step):
                    if time.time() - t0 > self.scan_timeout:
                        break
                    for x in range(0, W, step):
                        if (f0[y, x] == bg) != probe_bg:
                            continue
                        g = self._restore(base)
                        try:
                            r = g.perform_action(
                                ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                                raw=True
                            )
                            if not r.frame:
                                continue
                            f = np.array(r.frame[-1])
                            diff = np.sum(f0 != f)
                            if diff > 0:
                                effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if effect_hash not in seen_effects:
                                    seen_effects.add(effect_hash)
                                    actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                        except:
                            pass
        # [v13] if NOTHING was effective, report empty so warmup-unlock runs
        if len(actions) == len(deferred):
            return []
        return actions
    def _make_start_state(self, level_idx):
        """[v13] Build level N's TRUE start state by chaining the cached
        solutions for levels 0..N-1 from a fresh game — `set_level(N)` + RESET
        produces a DIFFERENT state than naturally advancing (player position,
        carried key rotation, etc.), so solutions found from the synthetic
        baseline can fail when replayed in the real environment."""
        try:
            g = self.game_cls()
            g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            for li in range(level_idx):
                sol = self.solutions.get(li)
                if sol is None:
                    return None
                for a, d in sol:
                    ai = ActionInput(id=GameAction.from_id(a), data=d) if d else ActionInput(id=GameAction.from_id(a))
                    r = g.perform_action(ai, raw=True)
                if g._current_level_index != li + 1:
                    logger.warning(f"BFS: chained replay desynced at L{li} "
                                   f"(at level {g._current_level_index})")
                    return None
            if not r.frame:
                return None
            # NOTE: frames must come from perform_action (camera-rendered,
            # 64x64) — get_pixels() returns the raw grid in native size and
            # the two are NOT comparable.
            return g, np.array(r.frame[-1])
        except Exception as e:
            logger.warning(f"BFS: chained baseline failed: {e}")
            return None

    def solve_level(self, level_idx, max_states=500000, prev_solution=None, frontier_path=None, strategy='bfs'):
        """Find optimal solution for a level via BFS (Memory Optimised via Action Replay)."""
        if not self.game_cls:
            return None
        # [v13] solution cache hit (pre-solved offline or in an earlier session)
        if level_idx in self.solutions:
            logger.info(f"BFS L{level_idx}: cache hit ({len(self.solutions[level_idx])} actions)")
            return self.solutions[level_idx]
        game = None
        f0 = None
        if level_idx > 0 and all(i in self.solutions for i in range(level_idx)):
            res = self._make_start_state(level_idx)
            if res is not None:
                game, f0 = res
                logger.info(f"BFS L{level_idx}: using TRUE chained baseline (replayed L0..L{level_idx-1})")
        if game is None:
            game = self.game_cls()
            game.set_level(level_idx)
            game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r0.frame:
                return None
            f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
        # Try solution transfer from previous level first
        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result
        # Phase 1: Scan for effective actions
        actions = self._scan_actions(game, f0, bg)
        # Warm-up unlock for locked initial states (sc25-type).
        # [v13 FIX] the warmup action MUST be prepended to any solution found
        # from the post-warmup state, or env replay desyncs by one action.
        warmup_prefix = []
        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    r_w = g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    if not r_w.frame:
                        continue
                    f_after = np.array(r_w.frame[-1])
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}! {len(warmup_actions)} actions")
                        game = g_warmup; f0 = f_after; actions = warmup_actions
                        warmup_prefix = [(warmup_id, None)]
                        break
                except:
                    pass
        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

        def _with_prefix(res):
            if res is not None and warmup_prefix:
                res = warmup_prefix + res
                self.solutions[level_idx] = res
            return res
        # ==========================================
        # Phase 2: BFS — [v13-speed] snapshot frontier (no history replay)
        # ==========================================
        transient = self._detect_transient(game, f0, actions)
        # ---------- [v13_1] space-shrinking strategy ladder ----------
        if strategy in ('auto', 'astar', 'iw', 'waypoint', 'ehc'):
            return self._solve_ladder(game, f0, actions, level_idx, strategy,
                                      max_states, frontier_path, transient,
                                      _with_prefix)
        # ---------- v13 original path (strategy: bfs / greedy) ----------
        # [v13] if a previous pass already proved the masked space is a dead
        # end (an unmasked frontier exists), skip the masked phase entirely
        skip_masked = bool(frontier_path and os.path.exists(frontier_path + '.nomask'))
        if not skip_masked:
            res, stats = self._bfs_search(game, f0, actions, level_idx, None,
                                          self.bfs_timeout, max_states,
                                          frontier_path=frontier_path,
                                          mask=transient,
                                          greedy=(strategy == 'greedy'))
            if res is not None:
                return _with_prefix(res)
            explored, n_unique, elapsed_first = stats
        else:
            explored, n_unique, elapsed_first = 0, 0, 0.0
        # [v13] masked search exhausted the (aliased) space without a goal —
        # time/phase matters in this level (waiting for doors/timers), so the
        # dedup hash must keep the transient pixels. Retry with unmasked hash;
        # greedy keeps the masked histogram as its progress prior.
        if skip_masked or (transient is not None and elapsed_first < self.bfs_timeout * 0.5):
            logger.info(f"BFS L{level_idx}: masked space exhausted ({elapsed_first:.1f}s) — continuing with unmasked hash")
            res, stats = self._bfs_search(game, f0, actions, level_idx, None,
                                          self.bfs_timeout - elapsed_first, max_states,
                                          frontier_path=(frontier_path + '.nomask') if frontier_path else None,
                                          mask=None, tag="unmasked",
                                          greedy=(strategy == 'greedy'),
                                          sig_mask=transient)
            if res is not None:
                return _with_prefix(res)
            explored, n_unique, elapsed_first = stats
        logger.info(f"BFS L{level_idx}: first pass timeout ({explored} explored, {n_unique} unique, {elapsed_first:.1f}s)")
        # Smart early exit — game may be too expensive to BFS
        if explored < 20 and elapsed_first > 10.0:
            logger.info(f"BFS L{level_idx}: early exit (only {explored} explored in {elapsed_first:.1f}s) — handing off to CNN")
            return None
        # If too few unique states found → hidden state detected → retry with probed fields
        if n_unique < 50 and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields}")
                # [v13] rebuild the SAME baseline as the first pass (chained
                # when possible — FIX 3 generalized)
                game2 = None
                f0_2 = None
                if level_idx > 0 and all(i in self.solutions for i in range(level_idx)):
                    res2 = self._make_start_state(level_idx)
                    if res2 is not None:
                        game2, f0_2 = res2
                if game2 is None:
                    game2 = self.game_cls()
                    game2.set_level(level_idx)
                    game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    r0_2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                    if not r0_2.frame:
                        return None
                    f0_2 = np.array(r0_2.frame[-1])
                # [v13] replay the warmup prefix so the retry baseline matches
                for wa, wd in warmup_prefix:
                    try:
                        r_w = game2.perform_action(ActionInput(id=GameAction.from_id(wa)), raw=True)
                        if r_w.frame:
                            f0_2 = np.array(r_w.frame[-1])
                    except: pass
                remaining = max(30, self.bfs_timeout - elapsed_first)
                res2, stats2 = self._bfs_search(game2, f0_2, actions, level_idx,
                                                hidden_fields, remaining, max_states,
                                                tag="hidden retry", mask=transient)
                if res2 is not None:
                    return _with_prefix(res2)
                logger.info(f"BFS L{level_idx}: hidden retry also failed ({stats2[0]} explored, {stats2[1]} unique)")
        return None
    def _solve_ladder(self, game, f0, actions, level_idx, strategy,
                      max_states, frontier_path, transient, _with_prefix):
        """[v13_1] retry ladder over the search rungs. Order follows
        RESEARCH.md: exact masked BFS first (keeps optimality on easy
        levels), then the space-shrinking incomplete rungs (waypoint tour,
        A*, IW novelty), then greedy, then v13's unmasked/hidden rescues.
        Non-exact rung solutions are verified by replay before caching."""
        t0 = time.time()
        budget = self.bfs_timeout
        lstats = self.level_stats.setdefault(level_idx, {'rungs': []})
        root_snap = self._snap(game)
        # [v13_1] ladder shape: a short exact-BFS sprint catches trivial
        # levels optimally; the aggressive space-shrinking rungs get small
        # slices; then exact BFS returns with ALL remaining time and RESUMES
        # the sprint's frontier (so no exact work is wasted — the ladder
        # never gives BFS less total time than v13's bfs pass would get).
        # [v13_1-tune] from the 600s M1 benchmark: iw rungs are cheap
        # (2-36s, two outright wins at 10-28x fewer states) -> run them
        # FIRST; waypoint/astar burned their full %-slices (48-72s) on
        # every level they didn't win -> absolute caps so the ladder tax
        # stops scaling with budget. Exact BFS still gets all leftover
        # time and resumes the sprint frontier.
        RUNG_CAPS = {'sense': 20.0, 'iw1': 30.0, 'iw2': 45.0, 'ehc': 30.0,
                     'waypoint': 25.0, 'astar': 20.0}
        if strategy == 'auto':
            # [v13_2] ehc slots after the iw rungs (cheap, search-discovered
            # subgoals) and before the geometry-guessing rungs
            # [v13_3-tune] sense rung sits AFTER iw1/iw2: those are cheap and
            # find solutions from root with optimal action counts. sense only
            # fires when iw1/iw2 failed — i.e. when the root-reachable space
            # is exhausted. Running sense first caused regressions: it found
            # solutions from revelation checkpoints that were longer than what
            # iw1 would find directly from root (ar25 L0: 19 vs 15 actions).
            plan = [('bfs', 'sprint'), ('iw1', 0.08), ('iw2', 0.08),
                    ('sense', 0.05), ('ehc', 0.07), ('waypoint', 0.06),
                    ('astar', 0.05), ('bfs', 'rest'), ('greedy', 'rest')]
        elif strategy == 'iw':
            plan = [('iw1', 0.6), ('iw2', 0.4)]
        else:
            plan = [(strategy, 1.0)]
        # internal resumable frontier for the bfs rungs (benchmark/callers
        # may pass frontier_path=None; sprint->rest resume still needs one)
        fp_bfs = (frontier_path + '.bfs') if frontier_path else os.path.join(
            '/tmp', f"v13_3_ladder_{os.path.basename(self.game_path)}"
                    f"_L{level_idx}_{os.getpid()}.pkl")
        own_fp = frontier_path is None

        def _cleanup():
            if own_fp:
                for p in (fp_bfs, fp_bfs + '.nomask'):
                    try:
                        if os.path.exists(p): os.unlink(p)
                    except OSError:
                        pass
        explored_acc, unique_max = 0, 0
        # [v13_1-tune] aggressive rungs share a POOL of at most half the
        # budget, >= 5s per rung or the rung is skipped — exact BFS always
        # keeps the other half. Fixes both small-budget failure modes:
        # %-slices too thin to win (3s iw1 missing a 3.1s solve) and
        # floors starving the exact rung (5 x 6s floors eating a 28s
        # budget). At 600s nothing changes (all rungs fit in the pool).
        AGG_MIN = 5.0
        agg_pool = budget * 0.5
        for name, frac in plan:
            remaining = budget - (time.time() - t0)
            if remaining < 2:
                break
            if frac == 'sprint':
                rb = min(remaining, max(3.0, min(6.0, budget * 0.12)))
            elif frac == 'rest':
                rb = remaining
            else:
                if agg_pool < AGG_MIN:
                    continue  # pool exhausted — skip, exact rungs still run
                rb = max(AGG_MIN, budget * frac)
                if strategy == 'auto' and name in RUNG_CAPS:
                    rb = min(rb, RUNG_CAPS[name])
                rb = min(rb, remaining, agg_pool)
            if name in ('bfs', 'greedy'):
                res, stats = self._bfs_search(game, f0, actions, level_idx,
                                              None, rb, max_states,
                                              frontier_path=(fp_bfs if name == 'bfs' else None),
                                              mask=transient,
                                              greedy=(name == 'greedy'))
            elif name == 'sense':
                # [v13_3] sensing prepass: find revelation checkpoints then
                # immediately retry IW(1) from each one
                revs, stats = self._sense_search(
                    game, f0, actions, level_idx, rb, max_states,
                    mask=transient)
                res = None
                for n_rev, rev_snap, rfb, rfs, rev_hist in revs:
                    rem2 = budget - (time.time() - t0) - 2
                    rem2 = min(rem2, max(8.0, budget * 0.06))
                    if rem2 < 3:
                        break
                    g_rev = self._restore(rev_snap)
                    f_rev = np.frombuffer(rfb, dtype=np.uint8).reshape(rfs)
                    iw_res, iw_st = self._guided_search(
                        g_rev, f_rev, actions, level_idx, None,
                        rem2, max_states, tag='sense+iw1',
                        mask=transient, mode='iw1')
                    stats = (stats[0]+iw_st[0], max(stats[1], iw_st[1]),
                             time.time()-t0)
                    if iw_res is not None:
                        full = rev_hist + iw_res
                        if self._verify_from_snap(root_snap, full, level_idx):
                            lstats['strategy'] = 'sense+iw1'
                            self.solutions[level_idx] = full
                            _cleanup()
                            return _with_prefix(full)
            elif name == 'ehc':
                res, stats = self._ehc_search(game, f0, actions, level_idx,
                                              None, rb, max_states,
                                              mask=transient)
            elif name in ('astar', 'iw1', 'iw2'):
                res, stats = self._guided_search(game, f0, actions, level_idx,
                                                 None, rb, max_states,
                                                 mask=transient, mode=name)
            else:  # waypoint
                res, stats = self._waypoint_search(game, f0, actions,
                                                   level_idx, None, rb,
                                                   max_states, mask=transient)
            explored_acc += stats[0]; unique_max = max(unique_max, stats[1])
            if frac not in ('sprint', 'rest'):
                agg_pool -= stats[2]
            lstats['rungs'].append({'rung': name, 'explored': stats[0],
                                    'unique': stats[1],
                                    'elapsed': round(stats[2], 2),
                                    'solved': bool(res)})
            if res is not None:
                exact = name in ('bfs', 'greedy')
                if exact or self._verify_from_snap(root_snap, res, level_idx):
                    lstats['strategy'] = name
                    # [v13_2] ANYTIME TIGHTEN: a win from any rung except
                    # exact FIFO BFS (already shortest-in-model) becomes an
                    # incumbent upper bound; spend leftover budget on a
                    # depth-bounded exact pass that either shortens it or
                    # proves it optimal-in-model. (ls20 L4: 78 -> 44.)
                    # only depth-greedy rungs produce long solutions; iw1/
                    # iw2 expand FIFO so their wins are already near-
                    # shortest — tightening them re-proves at full cost
                    remaining = budget - (time.time() - t0)
                    if name in ('greedy', 'ehc', 'waypoint', 'astar') \
                            and len(res) > 12 and remaining > 8:
                        bound = len(res)
                        tb = min(remaining - 2, max(8.0, budget * 0.35))
                        logger.info(f"BFS L{level_idx}: tighten pass — "
                                    f"incumbent {bound} actions, bounded "
                                    f"exact BFS ({tb:.0f}s)")
                        g2 = self._restore(root_snap)
                        res2, st2 = self._bfs_search_fifo(
                            g2, f0, actions, level_idx, None, tb,
                            max_states, tag=f"tighten<{bound}",
                            mask=transient, max_depth=bound - 1)
                        lstats['rungs'].append(
                            {'rung': f'tighten<{bound}', 'explored': st2[0],
                             'unique': st2[1], 'elapsed': round(st2[2], 2),
                             'solved': bool(res2)})
                        if res2 is not None and len(res2) < bound:
                            logger.info(f"BFS L{level_idx}: tightened "
                                        f"{bound} -> {len(res2)} actions")
                            lstats['strategy'] = name + '+tighten'
                            res = res2
                    self.solutions[level_idx] = res
                    _cleanup()
                    return _with_prefix(res)
                logger.warning(f"BFS L{level_idx}: {name} solution FAILED "
                               f"replay verification — discarded")
                lstats['rungs'][-1]['solved'] = 'failed-verify'
        # rescue 1 (v13): masked space may alias real time-dependence —
        # rerun exact BFS with the unmasked hash on leftover time
        remaining = budget - (time.time() - t0)
        if transient is not None and remaining > 5:
            res, stats = self._bfs_search(game, f0, actions, level_idx, None,
                                          remaining, max_states,
                                          tag="unmasked",
                                          frontier_path=fp_bfs + '.nomask',
                                          mask=None, sig_mask=transient)
            explored_acc += stats[0]; unique_max = max(unique_max, stats[1])
            lstats['rungs'].append({'rung': 'bfs-unmasked',
                                    'explored': stats[0], 'unique': stats[1],
                                    'elapsed': round(stats[2], 2),
                                    'solved': bool(res)})
            if res is not None:
                lstats['strategy'] = 'bfs-unmasked'
                _cleanup()
                return _with_prefix(res)
        # rescue 2 (v13): hidden scalar fields (countdowns etc.)
        # [v13_1-tune] also fire when the space EXHAUSTED early with most of
        # the budget unspent (su15 pattern: every rung dries up at ~1.3k
        # unique states in <60s of a 600s budget — that smells like state
        # the frame+public-scalar hash can't see, exactly what probed
        # hidden fields are for)
        remaining = budget - (time.time() - t0)
        early_exhaust = remaining > budget * 0.5
        if (unique_max < 50 or early_exhaust) and remaining > 5:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: ladder RETRY with hidden fields: {hidden_fields}")
                g2 = self._restore(root_snap)
                res, stats = self._bfs_search(g2, f0, actions, level_idx,
                                              hidden_fields, remaining,
                                              max_states, tag="hidden retry",
                                              mask=transient)
                lstats['rungs'].append({'rung': 'bfs-hidden',
                                        'explored': stats[0],
                                        'unique': stats[1],
                                        'elapsed': round(stats[2], 2),
                                        'solved': bool(res)})
                if res is not None:
                    lstats['strategy'] = 'bfs-hidden'
                    _cleanup()
                    return _with_prefix(res)
        _cleanup()
        logger.info(f"BFS L{level_idx}: ladder exhausted "
                    f"({explored_acc} explored across rungs, "
                    f"{time.time() - t0:.1f}s)")
        return None

    def _bfs_search(self, game, f0, actions, level_idx, hidden_fields,
                    time_budget, max_states, tag="", frontier_path=None,
                    mask=None, greedy=False, sig_mask=None):
        # [v13] greedy mode: best-first on "progress events" — a state whose
        # color histogram changed vs its parent has interacted with the map
        # (collected a key, opened a door). Movement alone never changes the
        # histogram (the sprite just translates; timer rows are masked).
        if greedy:
            return self._greedy_search(game, f0, actions, level_idx, hidden_fields,
                                       time_budget, max_states, tag, frontier_path,
                                       mask, sig_mask=sig_mask)
        return self._bfs_search_fifo(game, f0, actions, level_idx, hidden_fields,
                                     time_budget, max_states, tag, frontier_path, mask)

    @staticmethod
    def _hist_sig(frame, mask):
        fm = frame
        if mask is not None:
            fm = frame.copy(); fm[mask] = 0
        return tuple(np.bincount(fm.flatten(), minlength=16).tolist())

    def _greedy_search(self, game, f0, actions, level_idx, hidden_fields,
                       time_budget, max_states, tag="", frontier_path=None,
                       mask=None, sig_mask=None):
        if sig_mask is None:
            sig_mask = mask
        import heapq
        visited = set()
        heap = []
        explored = 0
        counter = 0
        dyn = any(a == 6 for a, _ in actions)
        root_sig = self._state_hash(game, f0, hidden_fields, mask=None)
        if frontier_path and os.path.exists(frontier_path):
            try:
                with open(frontier_path, 'rb') as fh:
                    st = pickle.load(fh)
                if st.get('root') != root_sig or st.get('fmt') != 2:
                    logger.info(f"BFS L{level_idx}: greedy frontier baseline/format changed — discarding")
                else:
                    visited, heap, explored, counter = st['visited'], st['heap'], st['explored'], st['counter']
                    logger.info(f"BFS L{level_idx}: resumed greedy frontier ({len(heap)} nodes, {len(visited)} visited)")
            except Exception as e:
                logger.warning(f"greedy frontier resume failed: {e}")
                visited, heap, explored, counter = set(), [], 0, 0
        if not heap:
            visited = set(); explored = 0; counter = 0
            visited.add(self._state_hash(game, f0, hidden_fields, mask=mask))
            sig0 = self._hist_sig(f0, sig_mask)
            heapq.heappush(heap, (0, 0, 0, self._snap(game), [], sig0,
                                  f0.astype(np.uint8).tobytes(), f0.shape))
        pool = None
        if self.workers > 1 and self._snap(game)[0] == 'p':
            try:
                import multiprocessing as mp
                # macOS requires 'spawn' (fork deadlocks torch); Linux uses 'fork'
                pool = mp.get_context(HW['mp_ctx']).Pool(
                    self.workers, initializer=_bfs_worker_init,
                    initargs=(self.game_path, mask, sig_mask, hidden_fields))
            except Exception:
                pool = None
        t0 = time.time()
        while heap and explored < max_states and (time.time() - t0) < time_budget:
            batch, metas = [], []
            take = max(1, (self.workers * 4) if pool else 1)
            while heap and len(batch) < take:
                negprog, depth, _, snap, hist, sig, fbytes, fshape = heapq.heappop(heap)
                batch.append((snap, actions, level_idx, fbytes, fshape, dyn))
                metas.append((negprog, depth, hist, sig))
            if pool:
                try:
                    node_results = pool.map(_bfs_expand_node, batch,
                                            chunksize=max(1, len(batch) // self.workers))
                except Exception as e:
                    logger.warning(f"greedy pool batch failed ({e})"); break
            else:
                _BFS_W['mask'] = mask; _BFS_W['sig_mask'] = sig_mask
                _BFS_W['hidden'] = hidden_fields
                node_results = [_bfs_expand_node(a) for a in batch]
            for node_out, (negprog, depth, hist, sig) in zip(node_results, metas):
                explored += len(node_out)
                for act_id, data, res in node_out:
                    if res is None:
                        continue
                    h, win, child_snap, child_sig, rfb, rfs, _scal = res
                    if h in visited:
                        continue
                    visited.add(h)
                    new_hist = hist + [(act_id, data)]
                    if win:
                        if pool: pool.terminate()
                        elapsed = time.time() - t0
                        logger.info(f"BFS L{level_idx}: SOLVED (greedy{(' '+tag) if tag else ''}) in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                        self.solutions[level_idx] = new_hist
                        if frontier_path and os.path.exists(frontier_path):
                            try: os.unlink(frontier_path)
                            except: pass
                        return new_hist, (explored, len(visited), elapsed)
                    prog = -negprog + (1 if child_sig != sig else 0)
                    if depth < 400:
                        counter += 1
                        heapq.heappush(heap, (-prog, depth + 1, counter, child_snap,
                                              new_hist, child_sig, rfb, rfs))
        if pool:
            pool.terminate()
        if frontier_path and heap:
            try:
                import shutil as _sh
                free_gb = _sh.disk_usage(os.path.dirname(frontier_path) or '/tmp').free / 1e9
                if free_gb < 1.5:
                    logger.warning(f"BFS L{level_idx}: <1.5GB disk free — skipping greedy frontier persist")
                else:
                    if len(heap) > 25000:
                        logger.warning(f"BFS L{level_idx}: greedy frontier capped 25000/{len(heap)} (keeping best-priority nodes)")
                        heap = sorted(heap)[:25000]
                    with open(frontier_path, 'wb') as fh:
                        pickle.dump({'visited': visited, 'heap': heap,
                                     'explored': explored, 'counter': counter,
                                     'root': root_sig, 'fmt': 2}, fh, -1)
                    logger.info(f"BFS L{level_idx}: greedy frontier persisted ({len(heap)} nodes)")
            except Exception as e:
                logger.warning(f"greedy frontier persist failed: {e}")
        elif frontier_path and not heap and os.path.exists(frontier_path):
            # [v13] search space exhausted — remove stale frontier so future
            # passes don't resume a dead end
            try: os.unlink(frontier_path)
            except: pass
        return None, (explored, len(visited), time.time() - t0)

    def _bfs_search_fifo(self, game, f0, actions, level_idx, hidden_fields,
                         time_budget, max_states, tag="", frontier_path=None,
                         mask=None, force_dyn=False, max_depth=200):
        """[v13-speed] BFS storing compressed pickle snapshots in the frontier.
        The v19 'memory optimised replay' re-simulated the full action history
        for every dequeued node (O(depth) sims + 2 deepcopies per expansion).
        Snapshots make node expansion O(branching) with ~7ms state restore.
        Returns (solution_or_None, (explored, unique, elapsed))."""
        visited = set()
        queue = deque()
        explored = 0
        dyn = force_dyn or any(a == 6 for a, _ in actions)
        root_sig = self._state_hash(game, f0, hidden_fields, mask=None)
        # [v13] resumable search: reload a persisted frontier if present —
        # but only if it was built from the SAME baseline state
        if frontier_path and os.path.exists(frontier_path):
            try:
                with open(frontier_path, 'rb') as fh:
                    st = pickle.load(fh)
                if st.get('root') != root_sig or st.get('fmt') != 2:
                    logger.info(f"BFS L{level_idx}: frontier baseline/format changed — discarding stale frontier")
                else:
                    visited, queue, explored = st['visited'], st['queue'], st['explored']
                    logger.info(f"BFS L{level_idx}: resumed frontier ({len(queue)} nodes, {len(visited)} visited, {explored} explored)")
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: frontier resume failed: {e}")
                visited, queue, explored = set(), deque(), 0
        if not queue:
            visited = set(); explored = 0
            h0 = self._state_hash(game, f0, hidden_fields, mask=mask)
            visited.add(h0)
            queue.append((self._snap(game), [], 0,
                          f0.astype(np.uint8).tobytes(), f0.shape))
        t0 = time.time()
        # [v13] optional multiprocess expansion pool. Some games hold
        # unpicklable members (lambdas) — those must run sequential.
        pool = None
        if self.workers > 1 and self._snap(game)[0] == 'p':
            try:
                import multiprocessing as mp
                pool = mp.get_context(HW['mp_ctx']).Pool(
                    self.workers, initializer=_bfs_worker_init,
                    initargs=(self.game_path, mask, mask, hidden_fields))
            except Exception as e:
                logger.warning(f"BFS: pool unavailable ({e}); sequential")
                pool = None
        elif self.workers > 1:
            logger.info("BFS: game state not picklable — sequential expansion")

        def _finish(sol, elapsed):
            if pool:
                pool.terminate()
            label = f" ({tag})" if tag else ""
            logger.info(f"BFS L{level_idx}: SOLVED{label} in {len(sol)} actions ({explored} explored, {elapsed:.1f}s)")
            self.solutions[level_idx] = sol
            if frontier_path and os.path.exists(frontier_path):
                try: os.unlink(frontier_path)
                except: pass
            return sol, (explored, len(visited), elapsed)

        while queue and explored < max_states and (time.time() - t0) < time_budget:
            # batch a slice of the frontier (one task per node so each
            # snapshot ships to the pool exactly once); the same node-level
            # expansion (incl. dynamic clicks) runs sequentially without pool
            batch, metas = [], []
            take = (self.workers * 4) if pool is not None else 1
            while queue and len(batch) < take:
                snap, hist, depth, fbytes, fshape = queue.popleft()
                batch.append((snap, actions, level_idx, fbytes, fshape, dyn))
                metas.append((hist, depth))
            if pool is not None:
                try:
                    node_results = pool.map(_bfs_expand_node, batch,
                                            chunksize=max(1, len(batch) // self.workers))
                except Exception as e:
                    logger.warning(f"BFS: pool batch failed ({e})")
                    break
            else:
                _BFS_W['mask'] = mask; _BFS_W['sig_mask'] = mask
                _BFS_W['hidden'] = hidden_fields
                node_results = [_bfs_expand_node(a) for a in batch]
            wins = []
            for node_out, (hist, depth) in zip(node_results, metas):
                explored += len(node_out)
                for act_id, data, res in node_out:
                    if res is None:
                        continue
                    h, win, child_snap, _sig, rfb, rfs, _scal = res
                    if h in visited:
                        continue
                    visited.add(h)
                    new_hist = hist + [(act_id, data)]
                    if win:
                        wins.append(new_hist)
                        continue
                    # [v13] depth cap raised 30 → 200: visited-dedup already
                    # bounds the search; 30 silently truncated longer mazes.
                    # [v13_2] max_depth doubles as the incumbent bound for
                    # anytime tighten passes (nothing >= L is expanded).
                    if depth + 1 < max_depth:
                        queue.append((child_snap, new_hist, depth + 1, rfb, rfs))
            if wins:
                best = min(wins, key=len)
                return _finish(best, time.time() - t0)
        if pool:
            pool.terminate()
        # [v13] persist frontier for a future resumed invocation.
        # Guards: cap node count (huge frontiers filled a 40GB disk once) and
        # require free disk headroom before writing.
        if frontier_path and queue:
            try:
                import shutil as _sh
                free_gb = _sh.disk_usage(os.path.dirname(frontier_path) or '/tmp').free / 1e9
                if free_gb < 1.5:
                    logger.warning(f"BFS L{level_idx}: <1.5GB disk free — skipping frontier persist")
                else:
                    if len(queue) > 25000:
                        logger.warning(f"BFS L{level_idx}: frontier capped 25000/{len(queue)} nodes (completeness reduced)")
                        queue = deque(list(queue)[:25000])
                    with open(frontier_path, 'wb') as fh:
                        pickle.dump({'visited': visited, 'queue': queue,
                                     'explored': explored, 'root': root_sig,
                                     'fmt': 2}, fh, -1)
                    logger.info(f"BFS L{level_idx}: frontier persisted ({len(queue)} nodes)")
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: frontier persist failed: {e}")
        elif frontier_path and not queue and os.path.exists(frontier_path):
            try: os.unlink(frontier_path)
            except: pass
        return None, (explored, len(visited), time.time() - t0)
    # ================= [v13_1] SPACE-SHRINKING SEARCH RUNGS =================

    def _detect_player(self, game, f0, actions):
        """Detect the player sprite: the color whose centroid MOVES (same
        pixel count) under directional actions. Returns (color, step_px) or
        (None, 1). Black-box: probed by simulation from the root state."""
        try:
            base = self._snap(game)
            dir_acts = [(a, d) for a, d in actions if 1 <= a <= 4]
            votes, steps = {}, []
            for act_id, data in dir_acts[:4]:
                g = self._restore(base)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                except Exception:
                    continue
                if not r.frame:
                    continue
                f1 = np.array(r.frame[-1])
                d = (f0 != f1)
                if not d.any():
                    continue
                changed = set(np.unique(f0[d]).tolist()) | set(np.unique(f1[d]).tolist())
                for c in changed:
                    n0, n1 = int((f0 == c).sum()), int((f1 == c).sum())
                    if n0 == 0 or n0 != n1 or n0 > f0.size // 4:
                        continue
                    y0, x0 = np.where(f0 == c)
                    y1, x1 = np.where(f1 == c)
                    dx = float(np.median(x1) - np.median(x0))
                    dy = float(np.median(y1) - np.median(y0))
                    disp = abs(dx) + abs(dy)
                    if 0 < disp <= 16:
                        votes[c] = votes.get(c, 0) + 1
                        steps.append(disp)
            if not votes:
                return None, 1.0
            pcolor = max(votes, key=votes.get)
            step = float(np.median(steps)) if steps else 1.0
            return int(pcolor), max(step, 1.0)
        except Exception:
            return None, 1.0

    @staticmethod
    def _player_pos(frame, pcolor):
        if pcolor is None:
            return None
        ys, xs = np.where(frame == pcolor)
        if len(xs) == 0:
            return None
        return (float(np.median(xs)), float(np.median(ys)))

    @staticmethod
    def _detect_goal(f0, pcolor, bg):
        """Goal guess for the A* heuristic: rarest non-player non-bg object.
        Wrong guesses only weaken the heuristic (h stays >= 0 informative-ish
        via weighted A*); h=None disables guidance (falls back to BFS order)."""
        objs = [o for o in _frame_objs(f0, bg) if o[0] != pcolor]
        if not objs:
            return None
        objs.sort(key=lambda o: o[3])  # fewest pixels = most special
        return (objs[0][1], objs[0][2])


    def _bg_mask_for(self, f0):
        """[v13_3] Background mask for the initial frame (cached).
        bg_mask[y,x]=True means that pixel was background at level start —
        any child state where it is non-background is a 'revelation event'."""
        fid = id(f0)
        if getattr(self, '_cached_f0_id', None) != fid:
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            self._cached_bg_mask = (f0 == bg)
            self._cached_bg     = bg
            self._cached_f0_id  = fid
        return self._cached_bg_mask, self._cached_bg

    def _sense_search(self, game, f0, actions, level_idx,
                      time_budget, max_states, tag="", mask=None):
        """[v13_3] Sensing prepass: short BFS maximising newly-revealed pixels.
        Returns (top_revs, stats) where top_revs is a list of up to 3
        (n_revealed, snap, fbytes, fshape, history) tuples (best first).
        The caller restarts IW(1) from each checkpoint — if IW finds a win,
        the full solution is rev_history + iw_solution."""
        import heapq as _hq
        bg_mask, bg = self._bg_mask_for(f0)
        _BFS_W['mask']     = mask
        _BFS_W['sig_mask'] = mask
        _BFS_W['hidden']   = None
        dyn    = any(a == 6 for a, _ in actions)
        root_h = self._state_hash(game, f0, None, mask=mask)
        visited = {root_h}
        ctr  = 0
        heap = [(-0, 0, ctr, self._snap(game), [],
                 f0.astype(np.uint8).tobytes(), f0.shape)]
        revelations = []
        explored = 0
        t0 = time.time()
        MAX_DEPTH = 25

        while heap and explored < max_states and (time.time()-t0) < time_budget:
            neg_rev, depth, _, snap, hist, fbytes, fshape = _hq.heappop(heap)
            if depth >= MAX_DEPTH:
                continue
            out = _bfs_expand_node((snap, actions, level_idx, fbytes, fshape, dyn))
            explored += len(out)
            for act_id, data, res in out:
                if res is None:
                    continue
                h, win, child_snap, child_sig, rfb, rfs, child_scal = res
                if h in visited:
                    continue
                visited.add(h)
                cf       = np.frombuffer(rfb, dtype=np.uint8).reshape(rfs)
                n_rev    = int(((cf != bg) & bg_mask).sum())
                new_hist = hist + [(act_id, data)]
                if n_rev > 0:
                    revelations.append((n_rev, child_snap, rfb, rfs, new_hist))
                ctr += 1
                _hq.heappush(heap, (-n_rev, depth+1, ctr, child_snap,
                                    new_hist, rfb, rfs))

        # deduplicate by n_revealed, keep top-3
        revelations.sort(key=lambda x: -x[0])
        seen_nc, top_revs = set(), []
        for item in revelations:
            nc = item[0]
            if nc not in seen_nc:
                seen_nc.add(nc)
                top_revs.append(item)
            if len(top_revs) >= 3:
                break

        best = top_revs[0][0] if top_revs else 0
        elapsed = time.time() - t0
        logger.info(f"BFS L{level_idx}: sense{(' '+tag) if tag else ''} — "
                    f"{explored} explored, best reveal: {best} px, "
                    f"{len(top_revs)} checkpoints ({elapsed:.1f}s)")
        return top_revs, (explored, len(visited), elapsed)

    def _guided_search(self, game, f0, actions, level_idx, hidden_fields,
                       time_budget, max_states, tag="", mask=None,
                       sig_mask=None, mode='astar', dominance=True,
                       w_h=1.5):
        """[v13_1] one engine, three rungs (all reuse the v13 worker pool):
          mode='astar': best-first on depth + w*manhattan(player, goal)/step
          mode='iw1'  : BFS pruned by IW(1) novelty (cell=color + scalar atoms)
          mode='iw2'  : iw1 + pairs of object-level atoms when iw1 atoms stale
        dominance=True additionally prunes states whose (player-cell,
        histogram, scalars) were seen at <= depth. These rungs are NOT
        complete/optimal — callers must verify returned solutions by replay
        (solve_level does)."""
        import heapq
        if sig_mask is None:
            sig_mask = mask
        H, W = f0.shape[:2]
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
        pcolor, step = self._detect_player(game, f0, actions)
        goal = self._detect_goal(f0, pcolor, bg) if mode == 'astar' else None
        if mode == 'astar' and (pcolor is None or goal is None):
            logger.info(f"BFS L{level_idx}: astar — no player/goal detected, h=0 (BFS order)")

        def _h(frame):
            if pcolor is None or goal is None:
                return 0.0
            pos = self._player_pos(frame, pcolor)
            if pos is None:
                return 0.0
            return (abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])) / step

        # IW novelty stores. Frame atoms: cell*16+color (bool array, O(1)).
        # Scalar atoms: hashed (key, value) pairs. iw2 adds PAIRS of
        # object-level atoms (color, qx, qy) — small, objects are few.
        seen_atoms = np.zeros(H * W * 16, dtype=bool)
        seen_scal = set()
        seen_pairs = set()
        cell_idx = np.arange(H * W, dtype=np.int64) * 16
        # [v13_3] reveal + reactive-block atom stores
        bg_mask_v3, _ = self._bg_mask_for(f0)
        seen_extra = set()

        def _novel(frame, scal, parent_frame=None):
            atoms = cell_idx + frame.reshape(-1).astype(np.int64)
            fresh = ~seen_atoms[atoms]
            novel = bool(fresh.any())
            seen_atoms[atoms] = True
            for kv in scal:
                if kv not in seen_scal:
                    seen_scal.add(kv)
                    novel = True
            # [v13_3] Reveal-novelty atoms: 4×4 grid-cell regions of the
            # frame that were background in f0 but are now non-background.
            # Prevents IW from pruning exploratory moves that uncover hidden
            # board areas (partial observability fix — see RESEARCH.md §6.1).
            H4 = max(1, H // 4); W4 = max(1, W // 4)
            rev_mask = bg_mask_v3 & (frame != bg)
            if rev_mask.any():
                for ry in range(4):
                    for rx in range(4):
                        y0 = ry * H4; y1 = min((ry+1)*H4, H)
                        x0 = rx * W4; x1 = min((rx+1)*W4, W)
                        if rev_mask[y0:y1, x0:x1].any():
                            key = ('rev', ry*4+rx)
                            if key not in seen_extra:
                                seen_extra.add(key)
                                novel = True
            # [v13_3] Reactive-block atoms: non-player pixels that changed
            # between parent and child frames. Grouped into 8×8 grid cells +
            # new color — one coarse atom per block per state instead of 64
            # per-pixel atoms per rotation step. Keeps IW width tight for
            # levels with co-moving blocks (ls20 L5 rotating/color blocks).
            if parent_frame is not None:
                diff = (frame != parent_frame)
                if pcolor is not None:
                    diff = diff & (frame != pcolor) & (parent_frame != pcolor)
                if diff.any():
                    ys_d, xs_d = np.where(diff)
                    by_d = ys_d // 8; bx_d = xs_d // 8
                    bc_d = frame.reshape(-1)[ys_d * W + xs_d]
                    for by, bx, bc in set(zip(by_d.tolist(), bx_d.tolist(), bc_d.tolist())):
                        key = ('rb', by, bx, int(bc))
                        if key not in seen_extra:
                            seen_extra.add(key)
                            novel = True
            if not novel and mode == 'iw2':
                oat = [(c, int(cx) // 4, int(cy) // 4)
                       for c, cx, cy, _n in _frame_objs(frame, bg)]
                oat += list(scal)
                for i in range(len(oat)):
                    for j in range(i + 1, len(oat)):
                        pr = (oat[i], oat[j])
                        if pr not in seen_pairs:
                            seen_pairs.add(pr)
                            novel = True
            return novel

        dom = {} if dominance else None

        def _dominated(frame, scal, sig, depth):
            if dom is None:
                return False
            pos = self._player_pos(frame, pcolor)
            key = (None if pos is None else (int(pos[0]), int(pos[1])),
                   sig, scal)
            prev = dom.get(key)
            if prev is not None and prev <= depth:
                return True
            dom[key] = depth
            return False

        visited = set()
        visited.add(self._state_hash(game, f0, hidden_fields, mask=mask))
        sig0 = self._hist_sig(f0, sig_mask)
        _novel(f0, _scalar_state(game))
        heap = []
        counter = 0
        heapq.heappush(heap, (0.0, 0, 0, self._snap(game), [], sig0,
                              f0.astype(np.uint8).tobytes(), f0.shape))
        dyn = any(a == 6 for a, _ in actions)
        explored = 0
        pool = None
        if self.workers > 1 and self._snap(game)[0] == 'p':
            try:
                import multiprocessing as mp
                pool = mp.get_context(HW['mp_ctx']).Pool(
                    self.workers, initializer=_bfs_worker_init,
                    initargs=(self.game_path, mask, sig_mask, hidden_fields))
            except Exception:
                pool = None
        t0 = time.time()
        while heap and explored < max_states and (time.time() - t0) < time_budget:
            batch, metas, par_frames = [], [], []
            take = max(1, (self.workers * 4) if pool else 1)
            while heap and len(batch) < take:
                _prio, depth, _, snap, hist, sig, fbytes, fshape = heapq.heappop(heap)
                batch.append((snap, actions, level_idx, fbytes, fshape, dyn))
                metas.append((depth, hist, sig))
                par_frames.append((fbytes, fshape))
            if pool:
                try:
                    node_results = pool.map(_bfs_expand_node, batch,
                                            chunksize=max(1, len(batch) // self.workers))
                except Exception as e:
                    logger.warning(f"{mode} pool batch failed ({e})"); break
            else:
                _BFS_W['mask'] = mask; _BFS_W['sig_mask'] = sig_mask
                _BFS_W['hidden'] = hidden_fields
                node_results = [_bfs_expand_node(a) for a in batch]
            for node_out, (depth, hist, sig), (par_fb, par_fs) in zip(node_results, metas, par_frames):
                explored += len(node_out)
                for act_id, data, res in node_out:
                    if res is None:
                        continue
                    h, win, child_snap, child_sig, rfb, rfs, child_scal = res
                    if h in visited:
                        continue
                    visited.add(h)
                    new_hist = hist + [(act_id, data)]
                    if win:
                        if pool: pool.terminate()
                        elapsed = time.time() - t0
                        logger.info(f"BFS L{level_idx}: SOLVED ({mode}{(' '+tag) if tag else ''}) "
                                    f"in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                        return new_hist, (explored, len(visited), elapsed)
                    if depth + 1 >= 400:
                        continue
                    cf = np.frombuffer(rfb, dtype=np.uint8).reshape(rfs)
                    if mode in ('iw1', 'iw2'):
                        pf = np.frombuffer(par_fb, dtype=np.uint8).reshape(par_fs)
                        if not _novel(cf, child_scal, parent_frame=pf):
                            continue
                        if _dominated(cf, child_scal, child_sig, depth + 1):
                            continue
                        prio = float(depth + 1)
                    else:  # astar
                        if _dominated(cf, child_scal, child_sig, depth + 1):
                            continue
                        prio = (depth + 1) + w_h * _h(cf)
                    counter += 1
                    heapq.heappush(heap, (prio, depth + 1, counter, child_snap,
                                          new_hist, child_sig, rfb, rfs))
        if pool:
            pool.terminate()
        return None, (explored, len(visited), time.time() - t0)

    def _leg_search(self, start_snap, f_start, actions, level_idx,
                    hidden_fields, waypoint, pcolor, step, time_budget,
                    mask=None, max_nodes=4000, tol=2.0):
        """[v13_1] tiny sequential A* toward one waypoint. Succeeds when the
        player centroid is within `tol` of the waypoint (or the object there
        changes/disappears, or the level is WON outright). Returns
        (actions, end_snap, end_frame, won, explored) or None."""
        import heapq
        _BFS_W['mask'] = mask; _BFS_W['sig_mask'] = mask
        _BFS_W['hidden'] = hidden_fields
        wx, wy = waypoint
        tol = max(tol, step * 0.75)
        target_c = int(f_start[int(round(wy)) % f_start.shape[0],
                                int(round(wx)) % f_start.shape[1]])
        visited = set()
        g0 = self._restore(start_snap)
        visited.add(self._state_hash(g0, f_start, hidden_fields, mask=mask))
        heap = [(0.0, 0, 0, start_snap, [],
                 f_start.astype(np.uint8).tobytes(), f_start.shape)]
        counter, explored = 0, 0
        t0 = time.time()
        while heap and explored < max_nodes and (time.time() - t0) < time_budget:
            _prio, depth, _, snap, hist, fbytes, fshape = heapq.heappop(heap)
            out = _bfs_expand_node((snap, actions, level_idx, fbytes, fshape, False))
            explored += len(out)
            for act_id, data, res in out:
                if res is None:
                    continue
                h, win, child_snap, _sig, rfb, rfs, _scal = res
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                cf = np.frombuffer(rfb, dtype=np.uint8).reshape(rfs)
                if win:
                    return new_hist, child_snap, cf, True, explored
                pos = self._player_pos(cf, pcolor)
                if pos is None:
                    continue
                dist = abs(pos[0] - wx) + abs(pos[1] - wy)
                obj_changed = int(cf[int(round(wy)) % cf.shape[0],
                                     int(round(wx)) % cf.shape[1]]) != target_c
                if dist <= tol or (obj_changed and dist <= tol * 3):
                    return new_hist, child_snap, cf, False, explored
                if depth + 1 < 120:
                    counter += 1
                    heapq.heappush(heap, ((depth + 1) + 1.5 * dist / step,
                                          depth + 1, counter, child_snap,
                                          new_hist, rfb, rfs))
        return None

    def _waypoint_search(self, game, f0, actions, level_idx, hidden_fields,
                         time_budget, max_states, tag="", mask=None):
        """[v13_1] TSP-style hierarchical decomposition (RESEARCH.md #1).
        Movement games: enumerate orderings of object centroids (cheapest
        estimated tour first), A* a leg to each waypoint, then a short
        finish-BFS. Click games: macro BFS restricted to centroid clicks
        (branching = #objects instead of #scanned pixels).
        Returns (solution, stats) like the other rungs; solution UNVERIFIED."""
        t0 = time.time()
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
        pcolor, step = self._detect_player(game, f0, actions)
        has_dirs = any(1 <= a <= 4 for a, _ in actions)
        has_clicks = any(a == 6 for a, _ in actions)
        explored_total = 0

        # ---- click games: centroid-click macro BFS ----
        if has_clicks and (not has_dirs or pcolor is None):
            acts = [(a, d) for a, d in actions if a != 6]
            logger.info(f"BFS L{level_idx}: waypoint(click) — centroid-click macro BFS "
                        f"({len(acts)} simple actions + dynamic centroids)")
            return self._bfs_search_fifo(game, f0, acts, level_idx,
                                         hidden_fields, time_budget,
                                         max_states, tag="waypoint-click",
                                         mask=mask, force_dyn=True)

        # ---- movement games: waypoint tour ----
        if pcolor is None:
            return None, (0, 0, time.time() - t0)
        ppos = self._player_pos(f0, pcolor)
        objs = [o for o in _frame_objs(f0, bg) if o[0] != pcolor]
        objs.sort(key=lambda o: o[3])          # rare objects first
        wps = [(o[1], o[2]) for o in objs[:5]]  # k cap: 5 waypoints
        if not wps or ppos is None:
            return None, (0, 0, time.time() - t0)
        leg_actions = [(a, d) for a, d in actions if a != 6]

        from itertools import permutations

        def tour_len(order):
            cur, total = ppos, 0.0
            for w in order:
                total += abs(cur[0] - w[0]) + abs(cur[1] - w[1])
                cur = w
            return total

        orderings = []
        for r in range(1, len(wps) + 1):       # partial tours too: visit 1..k
            orderings.extend(permutations(wps, r))
        orderings.sort(key=tour_len)
        orderings = orderings[:24]
        logger.info(f"BFS L{level_idx}: waypoint — player c{pcolor} step {step:.1f}, "
                    f"{len(wps)} waypoints, {len(orderings)} tours")
        root_snap = self._snap(game)
        leg_budget = max(2.0, min(12.0, time_budget / max(len(orderings), 1) / 1.5))
        for order in orderings:
            if time.time() - t0 > time_budget - 1:
                break
            snap, cur_f, sol, ok = root_snap, f0, [], True
            for wp in order:
                remaining = time_budget - (time.time() - t0)
                if remaining < 1:
                    ok = False; break
                leg = self._leg_search(snap, cur_f, leg_actions, level_idx,
                                       hidden_fields, wp, pcolor, step,
                                       min(leg_budget, remaining), mask=mask)
                if leg is None:
                    ok = False; break
                leg_sol, snap, cur_f, won, exp = leg
                explored_total += exp
                sol += leg_sol
                if won:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED (waypoint tour{(' '+tag) if tag else ''}) "
                                f"in {len(sol)} actions ({explored_total} explored, {elapsed:.1f}s)")
                    return sol, (explored_total, explored_total, elapsed)
            if not ok or not sol:
                continue
            # tour complete without win: short finish-BFS from the end state
            remaining = time_budget - (time.time() - t0)
            if remaining < 2:
                break
            g_end = self._restore(snap)
            fres, fstats = self._bfs_search_fifo(
                g_end, cur_f, actions, level_idx, hidden_fields,
                min(remaining, max(5.0, time_budget * 0.25)),
                min(max_states, 60_000), tag="wp-finish", mask=mask)
            explored_total += fstats[0]
            if fres:
                elapsed = time.time() - t0
                logger.info(f"BFS L{level_idx}: SOLVED (waypoint+finish{(' '+tag) if tag else ''}) "
                            f"in {len(sol) + len(fres)} actions ({explored_total} explored, {elapsed:.1f}s)")
                return sol + fres, (explored_total, explored_total, elapsed)
        return None, (explored_total, explored_total, time.time() - t0)

    def _ehc_search(self, game, f0, actions, level_idx, hidden_fields,
                    time_budget, max_states, tag="", mask=None,
                    sig_mask=None, plateau_nodes=6000, max_commits=64):
        """[v13_2] Enforced Hill Climbing (FF-style, black-box adapted).
        From the committed state, plateau-BFS until the FIRST child whose
        masked color-histogram signature was never achieved before in this
        search ("progress event"), COMMIT to it (discard the frontier),
        append the plateau path to the running solution, restart. Win is
        checked on every expansion. Incomplete by design (commitment can
        trap) -> caller verifies by replay and the ladder falls back, the
        same EHC -> best-first structure FF uses. Returns (sol, stats)."""
        if sig_mask is None:
            sig_mask = mask
        t0 = time.time()
        _BFS_W['mask'] = mask; _BFS_W['sig_mask'] = sig_mask
        _BFS_W['hidden'] = hidden_fields
        dyn = any(a == 6 for a, _ in actions)
        cur_snap = self._snap(game)
        cur_f = f0
        total_sol = []
        explored = 0
        seen_sigs = {self._hist_sig(f0, sig_mask)}
        commits = 0
        while (time.time() - t0) < time_budget and commits < max_commits \
                and explored < max_states:
            # ---- one plateau: small exact BFS for the next progress event
            visited = set()
            g_cur = self._restore(cur_snap)
            visited.add(self._state_hash(g_cur, cur_f, hidden_fields,
                                         mask=mask))
            queue = deque([(cur_snap, [], 0,
                            cur_f.astype(np.uint8).tobytes(), cur_f.shape)])
            found = None
            p_nodes = 0
            while queue and found is None and p_nodes < plateau_nodes \
                    and (time.time() - t0) < time_budget:
                snap, hist, depth, fbytes, fshape = queue.popleft()
                out = _bfs_expand_node((snap, actions, level_idx,
                                        fbytes, fshape, dyn))
                explored += len(out); p_nodes += len(out)
                for act_id, data, res in out:
                    if res is None:
                        continue
                    h, win, child_snap, child_sig, rfb, rfs, _scal = res
                    if h in visited:
                        continue
                    visited.add(h)
                    new_hist = hist + [(act_id, data)]
                    if win:
                        sol = total_sol + new_hist
                        elapsed = time.time() - t0
                        logger.info(f"BFS L{level_idx}: SOLVED (ehc, "
                                    f"{commits} commits{(' '+tag) if tag else ''}) "
                                    f"in {len(sol)} actions ({explored} explored, {elapsed:.1f}s)")
                        return sol, (explored, explored, elapsed)
                    if child_sig not in seen_sigs:
                        found = (child_snap, new_hist, child_sig, rfb, rfs)
                        break
                    if depth + 1 < 60:
                        queue.append((child_snap, new_hist, depth + 1,
                                      rfb, rfs))
            if found is None:
                # plateau exhausted with no new signature — EHC dead end
                logger.info(f"BFS L{level_idx}: ehc plateau dead-end after "
                            f"{commits} commits ({explored} explored)")
                return None, (explored, explored, time.time() - t0)
            child_snap, plateau_path, child_sig, rfb, rfs = found
            seen_sigs.add(child_sig)
            total_sol += plateau_path
            cur_snap = child_snap
            cur_f = np.frombuffer(rfb, dtype=np.uint8).reshape(rfs)
            commits += 1
        return None, (explored, explored, time.time() - t0)

    def _verify_from_snap(self, root_snap, sol, level_idx):
        """[v13_1] replay `sol` from the search baseline; True iff it wins.
        Aggressive rungs (waypoint/astar/iw) are not complete/optimal, and
        waypoint COMPOSES sub-solutions — never bank an unverified answer."""
        try:
            g = self._restore(root_snap)
            for act_id, data in sol:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                r = g.perform_action(ai, raw=True)
                if r.levels_completed > level_idx or g._current_level_index > level_idx:
                    return True
            return False
        except Exception as e:
            logger.warning(f"verify replay failed: {e}")
            return False

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        """Transfer previous level's solution to current level."""
        try:
            # Try executing prev solution directly
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct replay, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
            # Try object-relative transfer
            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())
            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))
            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)
            if not objs_prev or not objs_curr:
                return None
            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))
            if not matched:
                return None
            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])
            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (offset dx={dx:.0f},dy={dy:.0f}, {i+1} actions)")
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None
def find_game_source_and_class(game_id, arc_env=None):
    """Find the game .py file and class name."""
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]
    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            import re
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m:
                        cls_name = m.group(1)
                    break
    if not src:
        import re
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"**/game_sources/**/{gid}.py",
            # [v19] local repo layout so the BFS is testable off-Kaggle too
            os.path.join(_root, "environment_files", gid, "*", f"{gid}.py"),
            os.path.join(_root, "arc-prize-2026-arc-agi-3", "environment_files", gid, "*", f"{gid}.py"),
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                break
    return src, cls_name
# ==================== CNN FALLBACK ====================
class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a
class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B,M=mem_actions.shape
        if M==0:return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)
class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64, mult=1):
        super().__init__()
        s.g=g
        # [v13] mult=1 (Mac/T4) -> 256 max channels; mult=4 (RTX 6000) -> 1024
        c1, c2, c3, c4 = 32*mult, 64*mult, 128*mult, 256*mult
        s.c1=nn.Conv2d(in_ch,c1,3,padding=1);s.c2=nn.Conv2d(c1,c2,3,padding=1)
        s.c3=nn.Conv2d(c2,c3,3,padding=1);s.c4=nn.Conv2d(c3,c4,3,padding=1)
        s.attn=CBAM(c4);s.ar=nn.Conv2d(c4,c2,1);s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(c2*16*16,c4);s.ah=nn.Linear(c4,5);s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(c4,c3,3,padding=1);s.cc2=nn.Conv2d(c3,c2,3,padding=1)
        s.cc3=nn.Conv2d(c2,32,1);s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1);s.gf=nn.Linear(c4,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x=F.relu(s.c1(x));x=F.relu(s.c2(x));x=F.relu(s.c3(x));f=F.relu(s.c4(x))
        f=s.attn(f);af=F.relu(s.ar(f));af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f));cf=F.relu(s.cc2(cf));cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)
def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs
# ==================== AGENT ====================
class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10
    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1))
        if TORCH_AVAILABLE: torch.manual_seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device(HW['device']) if TORCH_AVAILABLE else None
        s.G=64; s.IN=26
        s.net=None; s.opt=None
        # [v13] buffers/batching from the hardware profile
        s.buf=deque(maxlen=HW['buf_size']); s.buf_h=set()
        s.bsz=HW['bsz']; s.tfreq=HW['tfreq']
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256); s._aem_actions=deque(maxlen=256); s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._obj_moved=0
        # FIX 1: Initialize _visited_hashes so _reward() deduplication works correctly
        s._visited_hashes = set()
        # [v13-compat] numpy bandit fallback state (used only when torch missing)
        s._act_stats = {}
        s._noop_memory = set()
        # BFS solver
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))
    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]
    def _init_bfs(s):
        """Initialize BFS solver on first call."""
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            # [v13] V13_BFS_TIMEOUT lets the harness cap in-play solving when
            # levels are expected to come from the offline cache
            bfs_to = float(os.environ.get('V13_BFS_TIMEOUT', 180))
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=bfs_to,
                               workers=HW['workers'])
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
                # [v19] NO stored-solution answer-book. Every level is solved
                # LIVE this run (no-stored-answers rule). The disk hydrate of
                # v13_bfs_cache_*.json is intentionally removed.
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")
    def _try_bfs_solve(s, level_idx):
        """Try to solve current level with BFS, using previous solution for transfer."""
        if s._bfs is None:
            return None
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            # [v19] no disk persistence of solutions (no answer-book). The
            # in-memory self.solutions holds only THIS run's live-found paths,
            # used to execute the current level and transfer to the next.
            return sol
        return None
    def _tensor(s, fd):
        frame = s._raw(fd)
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        s._bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==s._bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        d1=torch.zeros(3,64,64,dtype=torch.float32)
        for i,prev in enumerate(reversed(list(s.fhist))):
            if i>=3:break
            d1[i]=torch.from_numpy((frame!=prev).astype(np.float32))
        d2=torch.zeros(2,64,64,dtype=torch.float32)
        h=list(s.fhist)
        if len(h)>=2:d2[0]=torch.from_numpy((h[-1]!=h[-2]).astype(np.float32))
        if len(h)>=4:d2[1]=torch.from_numpy((h[-2]!=h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh,aug,d1,d2],0).to(s.device)
    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        col_act=np.sum(frame!=s._bg,axis=0)
        for c in range(20,44):
            if col_act[c]<=2 and np.sum(col_act[:c]>0)>=5 and np.sum(col_act[c+1:]>0)>=5:
                for y in range(64):
                    for x in range(c+1):mask[y*64+x]=0.05
                return mask
        row_act=np.sum(frame!=s._bg,axis=1)
        for r in range(20,44):
            if row_act[r]<=2 and np.sum(row_act[:r]>0)>=5 and np.sum(row_act[r+1:]>0)>=5:
                for y in range(r+1):
                    for x in range(64):mask[y*64+x]=0.05
                return mask
        return mask
    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        # FIX 1: Use s._visited_hashes (now properly initialized) for deduplication.
        # Previously _visited_hashes was never created, so the hasattr() check always
        # returned True (not hasattr = True) meaning every state change got +1.5,
        # causing the CNN to loop endlessly without penalty.
        mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
        diff=(prev_raw!=curr_raw)&mask;changed=np.any(diff)
        r=0.0
        if curr_h != prev_h:
            if curr_h not in s._visited_hashes:
                r += 1.5
                s._visited_hashes.add(curr_h)
            else:
                r += 0.2  # small reward for revisiting — not zero, avoids cliff in sparse games
        else:
            r -= 0.1
        if changed:r+=0.5
        curr_objs=fast_objects(curr_raw,s._bg)
        if s._prev_objs and curr_objs:
            moved=0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0]==po[0]:
                        dist=abs(co[1]-po[1])+abs(co[2]-po[2])
                        if 2<dist<20:moved+=1;break
            if moved>0:r+=0.3*min(moved,3);s._obj_moved=moved
        s._prev_objs=curr_objs
        return r
    def _sample(s, logits, avail=None, temp=1.0):
        al=logits[:5].clone();cl=logits[5:5+4096].clone()
        if avail is not None and len(avail)>0:
            mask=torch.full_like(al,float('-inf'));a6=False
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:mask[aid-1]=0.0
                elif aid==6:a6=True
            al=al+mask
            if not a6:cl=cl+torch.full_like(cl,float('-inf'))
        if s._wm is not None:cl=cl+torch.log(s._wm.to(s.device).clamp(min=0.01))
        ap=torch.sigmoid(al/temp);cp=torch.sigmoid(cl/temp)/(s.G*s.G)
        allp=torch.cat([ap,cp]);sm=allp.sum()
        if sm<1e-8:allp=torch.ones_like(allp)/len(allp)
        else:allp=allp/sm
        idx=np.random.choice(len(allp),p=allp.cpu().numpy())
        if idx<5:return idx,None
        ci=idx-5;return 5,(ci//s.G,ci%s.G)
    def _heuristic(s, frame, avail, step):
        av=set(int(a.value) if hasattr(a,'value') else int(a) for a in avail)
        for d in[1,2,3,4]:
            if d in av and step<4:return d-1,None
        if 6 in av:
            cnt=np.bincount(frame.flatten(),minlength=16);targets=[]
            for c in range(16):
                if c==s._bg or cnt[c]==0 or cnt[c]>2000:continue
                ys,xs=np.where(frame==c)
                if len(ys)>=2:targets.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
            targets.sort(key=lambda t:t[2]);pidx=step-4
            if 0<=pidx<len(targets):return 5,(targets[pidx][1],targets[pidx][0])
        if 5 in av:return 4,None
        choices=[a for a in av if 1<=a<=5]
        if choices:return random.choice(choices)-1,None
        return 0,None
    def _frame_to_tensor(s, frame):
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        zeros=torch.zeros(5,64,64,dtype=torch.float32)
        return torch.cat([oh,aug,zeros],0)
    def _train(s):
        # [v13 FIX] adaptive batch: with the RTX profile's bsz=2048 most
        # levels never accumulate a full batch, so the old gate
        # (len(buf) < bsz -> return) silently disabled training exactly on
        # the biggest hardware. Train with what we have past a small floor.
        bsz = min(s.bsz, len(s.buf))
        if bsz < 64: return
        indices=np.random.choice(len(s.buf),bsz,replace=False)
        batch=[s.buf[i] for i in indices]
        # [v13] single H2D transfer: stack on CPU first (the old per-item
        # .to(device) issued bsz separate copies — brutal at bsz=2048)
        states=torch.stack([s._frame_to_tensor(e['s']) for e in batch]).to(s.device, non_blocking=True)
        acts=torch.tensor([e['a'] for e in batch],dtype=torch.long,device=s.device)
        rews=torch.tensor([e['r'] for e in batch],dtype=torch.float32,device=s.device)
        rews=torch.sigmoid(rews);s.opt.zero_grad()
        try:
            if HW['amp']:
                # RTX/T4: bfloat16 autocast on Tensor Cores
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits=s.net(states)
                    acts_c=acts.clamp(0,logits.size(1)-1)
                    sel=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
                    loss=F.binary_cross_entropy_with_logits(sel,rews)
                    p=torch.sigmoid(logits);loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
            else:
                # Mac M1 (MPS) / CPU: native FP32 for gradient stability
                logits=s.net(states)
                acts_c=acts.clamp(0,logits.size(1)-1)
                sel=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
                loss=F.binary_cross_entropy_with_logits(sel,rews)
                p=torch.sigmoid(logits);loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
            loss.backward();s.opt.step()
        except (RuntimeError, MemoryError) as e:
            if 'out of memory' not in str(e).lower():
                raise
            # [v13] OOM backoff: bsz=2048 x mult=4 ForgeNet stores ~40GB of
            # activations at 64x64 — too much even for big cards. Halve the
            # batch permanently and recover instead of crashing the run.
            s.bsz = max(64, bsz // 2)
            try:
                if s.device is not None and s.device.type == 'cuda':
                    torch.cuda.empty_cache()
            except Exception:
                pass
            logger.warning(f"_train: GPU OOM — batch size backed off to {s.bsz}")
    def _get_aem_tensors(s):
        if len(s._aem_diffs)<2:return None,None,None
        M=len(s._aem_diffs)
        diffs=torch.zeros(1,M,1,64,64,device=s.device)
        acts=torch.zeros(1,M,dtype=torch.long,device=s.device)
        rews=torch.zeros(1,M,device=s.device)
        for i,(d,a,r) in enumerate(zip(s._aem_diffs,s._aem_actions,s._aem_rewards)):
            diffs[0,i,0]=torch.from_numpy(d.astype(np.float32));acts[0,i]=min(a,4);rews[0,i]=r
        return diffs,acts,rews
    def _numpy_choose(s, lf, ch):
        """[v13-compat] torch-free fallback: experience-weighted bandit.
        Learns which actions actually change state, never repeats an action
        that was a no-op in the same state, clicks small-object centroids."""
        raw = s._raw(lf)
        avail = getattr(lf, 'available_actions', None) or []
        av = set(int(a.value) if hasattr(a, 'value') else int(a) for a in avail)
        if s.pr is not None and s.pai is not None:
            changed = bool(np.any(s.pr != raw))
            n, w = s._act_stats.get(s.pai, (0, 0.0))
            s._act_stats[s.pai] = (n + 1, w + (1.0 if changed else 0.0))
            if not changed and s.ph is not None:
                s._noop_memory.add((s.ph, s.pai))
            s._reward(s.pr, raw, s.ph, ch)
        candidates = []
        for aid in sorted(a for a in av if 1 <= a <= 5):
            candidates.append((aid, None))
        if 6 in av:
            cnt = np.bincount(raw.flatten(), minlength=16)
            bg = int(cnt.argmax())
            targets = []
            for c in range(16):
                if c == bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys, xs = np.where(raw == c)
                if len(ys) >= 2:
                    targets.append((int(np.median(ys)), int(np.median(xs)), int(cnt[c])))
            targets.sort(key=lambda t: t[2])
            for (y, x, _) in targets[:8]:
                candidates.append((6, (y, x)))
        if not candidates:
            a = GameAction.ACTION5 if 5 in av else GameAction.RESET
            a.reasoning = "np:none-avail"
            return a
        total_n = sum(n for n, _ in s._act_stats.values()) + 1
        scored = []
        for aid, coords in candidates:
            key = aid if aid != 6 else (6, coords)
            n, w = s._act_stats.get(key, (0, 0.0))
            prod = (w / n) if n else 0.6
            ucb = prod + 0.8 * np.sqrt(np.log(total_n + 1) / (n + 1))
            if (ch, key) in s._noop_memory:
                ucb -= 10.0
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
        s.fhist.append(raw.copy())
        s.pt = None; s.pai = key; s.pr = raw.copy(); s.ph = ch; s.la += 1
        return sel
    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True
    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)
            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                # Init BFS solver on first level
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()
                # Try BFS for this level
                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)
                # Init CNN fallback
                s.buf.clear(); s.buf_h.clear()
                if TORCH_AVAILABLE:
                    if HW['mode'] == 'RTX_6000':
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    if HW['device'] == 'cuda':
                        # fixed 64x64 input shapes → let cudnn pick best kernels
                        torch.backends.cudnn.benchmark = True
                    s.net = ForgeNet(s.IN, s.G, mult=HW['net_mult']).to(s.device)
                    # [v13 FIX] load pretrained weights BEFORE torch.compile:
                    # compile wraps the module and prefixes state_dict keys
                    # with '_orig_mod.', so loading after compile silently
                    # matches zero keys and drops the checkpoint.
                    for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                               'pretrained_weights.pt']:
                        try:
                            if os.path.exists(wp):
                                state=torch.load(wp,map_location=s.device,weights_only=True)
                                ms=s.net.state_dict()
                                for k in list(state.keys()):
                                    if k in ms and state[k].shape==ms[k].shape:ms[k]=state[k]
                                s.net.load_state_dict(ms);break
                        except: pass
                    if HW['compile']:
                        try: s.net = torch.compile(s.net)
                        except: pass
                    s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None
                s._aem_diffs.clear();s._aem_actions.clear();s._aem_rewards.clear()
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0
                # FIX 1: Reset visited hashes on every level change
                s._visited_hashes = set()
                # [v13-compat] reset bandit memory per level
                s._act_stats = {}; s._noop_memory = set()
                # FIX 4: Only reset epsilon if BFS didn't solve this level.
                # If BFS solved it, keep current eps so CNN fallback (if needed)
                # benefits from accumulated exploration knowledge.
                if not s._bfs_solution:
                    s._eps = 0.15
                # CLTI — inject BFS demos from previous level into CNN replay buffer
                # FIX 2: Use perform_action frame[-1] consistently with _raw(),
                # instead of get_pixels() which returns a different format.
                if lvl > 0 and s._bfs and s._bfs.solutions.get(lvl - 1):
                    prev_sol = s._bfs.solutions[lvl - 1]
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.set_level(lvl - 1)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            # Start from the post-reset frame, consistent with _raw()
                            prev_frame = np.array(r0.frame[-1], dtype=np.int64)
                            for act_id, data in prev_sol:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                result = replay_game.perform_action(ai, raw=True)
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                                s.buf.append({'s': prev_frame.copy(), 'a': action_idx, 'r': 2.0})
                                # Advance prev_frame using the action result, not get_pixels()
                                if result.frame:
                                    prev_frame = np.array(result.frame[-1], dtype=np.int64)
                            # [v13 FIX] same adaptive-batch reasoning as _train:
                            # demo counts are small; don't gate on full bsz
                            if TORCH_AVAILABLE and len(s.buf) >= 64:
                                for _ in range(min(20, max(1, len(s.buf) // min(s.bsz, len(s.buf))))):
                                    s._train()
                                logger.info(f"CLTI: injected {len(prev_sol)} expert demos from L{lvl-1}")
                    except Exception as e:
                        logger.warning(f"CLTI failed: {e}")
            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                a=GameAction.RESET;a.reasoning="reset";return a
            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel
            # ===== [v13-compat] NUMPY BANDIT (no torch) =====
            if not TORCH_AVAILABLE:
                raw = s._raw(lf)
                ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
                return s._numpy_choose(lf, ch)
            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)
            if s.pt is not None and s.pai is not None:
                mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
                diff_map=(s.pr!=raw)&mask;changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,'',ch)
                    s.buf.append({'s':s.pr.copy(),'a':s.pai,'r':r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai,4))
                        s._aem_rewards.append(r)
                if changed:s._ckpt_hash=ch;s._unproductive=0
                else:s._unproductive+=1
            avail_idx=[]
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:avail_idx.append(aid-1)
                elif aid==6:avail_idx.extend([5+i for i in range(0,4096,128)])
            if s._wm is None:s._wm=s._detect_template(raw)
            if s._undo_avail and s._unproductive>=30 and s._ckpt_hash:
                s._unproductive=0;a=GameAction.ACTION7;a.reasoning="undo"
                s.pt=tensor;s.pai=6;s.pr=raw.copy();s.ph=ch;s.la+=1;return a
            if not s._wd:
                if s.la<10:aidx,coords=s._heuristic(raw,avail,s.la)
                else:
                    s._wd=True
                    for _ in range(min(5,len(s.buf)//s.bsz)):s._train()
            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._get_aem_tensors()
                        if HW['amp']:
                            # [v13] bf16 inference on Tensor Cores (cuda only)
                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                if mem[0] is not None:logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                                else:logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                            logits=logits.float()
                        else:
                            if mem[0] is not None:logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                            else:logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:s._wd=True;aidx,coords=0,None
            if aidx<5:sel=s.al[aidx];sel.reasoning=f"cnn:a{aidx+1}"
            else:
                sel=GameAction.ACTION6;y,x=coords
                sel.set_data({"x":int(x),"y":int(y)});sel.reasoning=f"cnn:c({x},{y})"
            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            if s.action_counter%s.tfreq==0 and s._wd:s._train()
            return sel
        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a
