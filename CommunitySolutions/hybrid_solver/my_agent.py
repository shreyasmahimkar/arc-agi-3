# =====================================================================
# MASTER BASELINE v1 — Best-of-6 Hybrid Agent for ARC-AGI-3
#
# Built by merging the best parts of 6 top public notebooks:
#
# CORE: FORGE v19 (op_2) — most advanced BFS engine:
#   - A* search with game introspection heuristic (indicator sprites)
#   - Transient field detection (avoids state explosion from counters)
#   - _get_valid_actions() for correct click coordinate detection
#   - Dynamic action rescan BFS (for flood-fill games)
#   - Object model tracking (static/dynamic classification)
#   - _fast_deepcopy (skips camera for 2-3x faster copying)
#   - Level advancement by action replay (correct for multi-level)
#
# ADDITIONS from FORGE v17 (op_3):
#   - Beam search fallback (width 20-200, depth 60)
#   - Sprite permutation for click-only games ≤8 sprites
#   - Stride-1 neighbor click probing (catch odd-coordinate sprites)
#   - Prioritized experience replay (recent + high-reward weighted)
#   - Adaptive BFS time budget
#
# ADDITIONS from MCTS notebook (op_5):
#   - Click masking during CNN inference (only predict known-effective positions)
#   - Novelty-guided action selection during exploration phase
#
# ALL v19 BUG FIXES:
#   - _visited_hashes properly initialized in __init__
#   - 2 RESET calls (not 3) in BFS hidden retry
#   - Epsilon only resets when BFS actually failed
#   - FIX: frame extraction uses perform_action result throughout
# =====================================================================
import copy
import glob
import hashlib
import heapq
import importlib.util
import logging
import math
import os
import pickle
import random
import time
import traceback
from collections import defaultdict, deque
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)


# ==================== FAST DEEPCOPY ====================

def _fast_deepcopy(game):
    """Deepcopy game object, skipping the camera (rendering-only, never mutates)."""
    camera = getattr(game, '_camera', None)
    if camera is not None:
        game._camera = None
    g = pickle.loads(pickle.dumps(game, protocol=pickle.HIGHEST_PROTOCOL))
    if camera is not None:
        game._camera = camera
        g._camera = camera
    return g


# ==================== BFS SOLVER ====================

class BFSSolver:
    """
    Hybrid search engine: A* + dynamic rescan + IDDFS + beam + sprite permutation.
    This solver attempts to find an exact solution by exploring the state space of the game.
    It combines multiple search strategies to handle different types of games (e.g., directional, click-based).
    """

    def __init__(self, game_path, game_class_name, scan_timeout=4, bfs_timeout=180):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}
        self.timed_out_levels = set()

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    # ---- state hashing ----

    def _state_hash(self, g, frame, hidden_fields=None, transient_fields=None):
        """
        Creates a unique hash for a game state to avoid revisiting the same state.
        It hashes the visual frame and optionally includes internal game variables,
        while ignoring transient variables like action counters that change every turn.
        """
        fh = str(hash(frame.tobytes()))
        ignore = {'_action_count', '_full_reset', '_action_complete', '_debug', '_seed'}
        if transient_fields:
            ignore.update(transient_fields)
        extras = []
        for k, v in g.__dict__.items():
            if k.startswith('__') or k in ignore:
                continue
            if isinstance(v, (int, float, bool)):
                extras.append(f"{k}={v}")
            elif isinstance(v, (set, frozenset)) and len(v) < 50:
                extras.append(f"{k}={sorted(str(i) for i in v)}")
        if extras:
            eh = str(hash("|".join(sorted(extras))))
            return fh + "|" + eh
        return fh

    # ---- hidden / transient field detection ----

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')}
        changing = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = _fast_deepcopy(game)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                g.perform_action(ai, raw=True)
            except:
                continue
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing.add(k)
        return sorted(f for f in changing
                      if not f.startswith('_') or f in ('_current_level_index', '_score'))

    def _detect_transient_fields(self, game, actions):
        """Fields that change on EVERY action — e.g. budget counters. Exclude from hash."""
        if not actions:
            return set()
        ignore = {'_action_count', '_full_reset', '_action_complete'}
        initial = {k: v for k, v in game.__dict__.items()
                   if isinstance(v, (int, float, bool)) and not k.startswith('__')
                   and k not in ignore}
        changed_count = defaultdict(int)
        n_sampled = 0
        for act_id, data in actions[:min(12, len(actions))]:
            g = _fast_deepcopy(game)
            try:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                g.perform_action(ai, raw=True)
            except:
                continue
            n_sampled += 1
            for k in initial:
                if getattr(g, k, initial[k]) != initial[k]:
                    changed_count[k] += 1
        if n_sampled == 0:
            return set()
        transient = set()
        for k, cnt in changed_count.items():
            if cnt != n_sampled:
                continue
            if isinstance(initial[k], bool):
                continue  # boolean flags encode meaningful state
            transient.add(k)
        if transient:
            logger.info(f"BFS: transient fields (excluded from hash): {transient}")
        return transient

    # ---- goal heuristic (indicator introspection) ----

    def _build_goal_heuristic(self, f_init, f_prev_win, demo_model=None):
        def count_indicators(game):
            try:
                total, satisfied = 0, 0
                for av in game.__dict__.values():
                    if not isinstance(av, dict):
                        continue
                    for v in av.values():
                        if not isinstance(v, list):
                            continue
                        for item in v:
                            if hasattr(item, 'is_visible') and hasattr(item, 'pixels'):
                                total += 1
                                if item.is_visible:
                                    satisfied += 1
                return total, satisfied
            except:
                return 0, 0

        if self.game_cls:
            try:
                test = self.game_cls()
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                test.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                total, _ = count_indicators(test)
                if total > 0:
                    logger.info(f"BFS heuristic: introspection found {total} indicators")
                    def introspection_heuristic(f, game=None):
                        if game is None:
                            return 0
                        t, s = count_indicators(game)
                        return max(0, t - s)
                    return introspection_heuristic
            except:
                pass

        logger.info("BFS heuristic: uniform cost (no indicators found)")
        return lambda f, game=None: 0

    # ---- action scanning ----

    def _scan_actions(self, game, f0, bg):
        """
        Scan for effective actions. Uses _get_valid_actions() when available (fast + precise).
        It tests available actions to see which ones actually change the game state (visual frame).
        This drastically reduces the branching factor during BFS.
        """
        avail = game._available_actions
        actions = []

        # Directional / interact
        for a in [a for a in avail if a <= 5]:
            g = _fast_deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass

        if 6 not in avail:
            return actions

        seen_effects = set()

        # Primary: use game's own valid action list (exact click coords, much faster)
        if hasattr(game, '_get_valid_actions'):
            try:
                for ai_obj in game._get_valid_actions():
                    act_id = ai_obj.id._value_ if hasattr(ai_obj.id, '_value_') else int(ai_obj.id)
                    if act_id != 6:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(ai_obj, raw=True)
                        if r.frame:
                            f = np.array(r.frame[-1])
                            if np.sum(f0 != f) > 0:
                                eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if eh not in seen_effects:
                                    seen_effects.add(eh)
                                    actions.append((6, ai_obj.data))
                    except:
                        pass
            except:
                pass

        # Fallback: pixel scan (stride 2) if _get_valid_actions unavailable
        if not seen_effects:
            t0 = time.time()
            hit_positions = []
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True)
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0 != f) > 0:
                            eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if eh not in seen_effects:
                                seen_effects.add(eh)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                                hit_positions.append((x, y))
                    except:
                        pass

            # Stride-1 neighbors of hit positions (catch odd-coordinate sprites)
            tried = {(x, y) for x, y in hit_positions}
            for hx, hy in hit_positions:
                if time.time() - t0 > self.scan_timeout * 1.5:
                    break
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = hx + dx, hy + dy
                    if (nx, ny) in tried or not (0 <= nx < 64 and 0 <= ny < 64):
                        continue
                    tried.add((nx, ny))
                    if f0[ny, nx] == bg:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x': nx, 'y': ny, 'game_id': 'bfs'}),
                            raw=True)
                        if r.frame:
                            f = np.array(r.frame[-1])
                            if np.sum(f0 != f) > 0:
                                eh = hashlib.md5(f.tobytes()).hexdigest()[:12]
                                if eh not in seen_effects:
                                    seen_effects.add(eh)
                                    actions.append((6, {'x': nx, 'y': ny, 'game_id': 'bfs'}))
                    except:
                        pass

        # BG pixel scanning: stride-4 pass over background pixels (finds invisible click zones)
        if seen_effects:
            bg_budget = max(1, len(seen_effects) // 2)
            bg_added = 0
            t_bg = time.time()
            for y in range(0, 64, 4):
                if bg_added >= bg_budget or time.time() - t_bg > self.scan_timeout * 0.5:
                    break
                for x in range(0, 64, 4):
                    if bg_added >= bg_budget:
                        break
                    if f0[y, x] != bg:
                        continue
                    g = _fast_deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6,
                                        data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True)
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        if np.sum(f0 != f) > 0:
                            eh = str(hash(f.tobytes()))
                            if eh not in seen_effects:
                                seen_effects.add(eh)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                                bg_added += 1
                    except:
                        pass

        return actions

    def _probe_mover_target_colors(self, game):
        g = _fast_deepcopy(game)
        avail = [a for a in game._available_actions if 1 <= a <= 4]
        if not avail:
            return set(), set()
        try:
            r0 = g.perform_action(ActionInput(id=GameAction.from_id(avail[0])), raw=True)
            if not r0.frame:
                return set(), set()
            f0 = np.array(r0.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_centroids(frame):
                result = {}
                for c in range(16):
                    if c == bg:
                        continue
                    mask = (frame == c)
                    n = int(np.sum(mask))
                    if n < 2:
                        continue
                    ys, xs = np.where(mask)
                    result[c] = (float(np.mean(xs)), float(np.mean(ys)))
                return result

            movement = {}
            prev_c = get_centroids(f0)
            for _ in range(20):
                act = random.choice(avail)
                r2 = g.perform_action(ActionInput(id=GameAction.from_id(act)), raw=True)
                if not r2.frame:
                    break
                curr_c = get_centroids(np.array(r2.frame[-1]))
                for c in prev_c:
                    if c in curr_c:
                        movement[c] = (movement.get(c, 0.0)
                                       + abs(curr_c[c][0] - prev_c[c][0])
                                       + abs(curr_c[c][1] - prev_c[c][1]))
                prev_c = curr_c

            mover_colors = {c for c, m in movement.items() if m > 5}
            target_colors = {c for c, m in movement.items() if m == 0}
            return mover_colors, target_colors
        except:
            return set(), set()

    # ---- main solver ----

    def _init_game_at_level(self, level_idx):
        """Return (game, last_r) positioned at level_idx. Uses set_level() if available, else action replay."""
        game = self.game_cls()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        last_r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if level_idx == 0:
            return game, last_r
        if hasattr(game, 'set_level'):
            try:
                game.set_level(level_idx)
                game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                return game, last_r
            except Exception:
                pass
        # Fallback: action replay
        game = self.game_cls()
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        last_r = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        for prev_idx in range(level_idx):
            prev_sol = self.solutions.get(prev_idx)
            if not prev_sol:
                return None, None
            for act_id, data in prev_sol:
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                last_r = game.perform_action(ai, raw=True)
        return game, last_r

    def solve_level(self, level_idx, max_states=500000,
                    prev_solution=None, goal_heuristic=None):
        """
        The core method to solve a specific level.
        It progresses through multiple phases:
        1. A* Search: Uses a heuristic to find the shortest path.
        2. Dynamic Rescan: Handles flood-fill games where actions unlock over time.
        3. Hidden Fields Retry: Retries search by factoring in hidden state variables.
        4. IDDFS: Iterative Deepening DFS for deep directional games.
        5. Sprite Permutation: Tries all permutations for click-only games with few targets.
        6. Beam Search: Explores a fixed width of promising paths.
        """
        if not self.game_cls:
            return None

        # Advance to target level using set_level() or action replay
        game, last_r = self._init_game_at_level(level_idx)
        if game is None:
            return None

        if not last_r.frame:
            return None
        f0 = np.array(last_r.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        # Try solution transfer from previous level
        if prev_solution and level_idx > 0:
            transfer = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer:
                return transfer

        # Scan actions
        actions = self._scan_actions(game, f0, bg)

        # Warm-up unlock for frozen initial states
        if not actions:
            logger.info(f"BFS L{level_idx}: 0 actions found, trying warm-up unlock")
            avail = game._available_actions
            # Try click warm-up via _get_valid_actions if available
            if 6 in avail and hasattr(game, '_get_valid_actions'):
                try:
                    for va in game._get_valid_actions():
                        act_id = va.id._value_ if hasattr(va.id, '_value_') else int(va.id)
                        if act_id == 6:
                            g_warmup = _fast_deepcopy(game)
                            g_warmup.perform_action(va, raw=True)
                            r_after = g_warmup.perform_action(
                                ActionInput(id=GameAction.ACTION1), raw=True)
                            if r_after.frame:
                                f_after = np.array(r_after.frame[-1])
                                warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                                if warmup_actions:
                                    logger.info(f"BFS L{level_idx}: UNLOCKED with click!")
                                    game = g_warmup
                                    f0 = f_after
                                    actions = warmup_actions
                                    break
                except:
                    pass
            if not actions:
                for warmup_id in [a for a in avail if a <= 4]:
                    g_warmup = _fast_deepcopy(game)
                    try:
                        g_warmup.perform_action(
                            ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                        f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                        warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                        if warmup_actions:
                            logger.info(f"BFS L{level_idx}: UNLOCKED with ACTION{warmup_id}!")
                            game = g_warmup
                            f0 = f_after
                            actions = warmup_actions
                            break
                    except:
                        pass

        logger.info(f"BFS L{level_idx}: {len(actions)} effective actions")
        if not actions:
            return None

        transient_fields = self._detect_transient_fields(game, actions)
        hfn = goal_heuristic if goal_heuristic is not None else (lambda f, game=None: 0)
        _hfn_uses_game = goal_heuristic is not None

        # ---- Phase 1: A* search ----
        visited = set()
        base_game = _fast_deepcopy(game)
        h0 = self._state_hash(game, f0, transient_fields=transient_fields)
        visited.add(h0)
        counter = 0
        pq = [(hfn(f0, game) * 10, 0, counter, [], base_game)]
        t0 = time.time()
        explored = 0

        while pq and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            f_score, g_score, _, hist, node_game = heapq.heappop(pq)
            for act_id, data in actions:
                g2 = _fast_deepcopy(node_game)
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                          if data else ActionInput(id=GameAction.from_id(act_id)))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, transient_fields=transient_fields)
                if h in visited:
                    continue
                visited.add(h)
                new_hist = hist + [(act_id, data)]
                new_g = g_score + 1
                if (r.levels_completed > level_idx
                        or g2._current_level_index > level_idx):
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED (A*) in {len(new_hist)} actions "
                                f"({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist
                h_val = hfn(f, g2 if _hfn_uses_game else None) * 10
                counter += 1
                heapq.heappush(pq, (new_g + h_val, new_g, counter, new_hist, g2))

        elapsed_first = time.time() - t0
        logger.info(f"BFS L{level_idx}: A* timeout ({explored} explored, "
                    f"{len(visited)} unique, {elapsed_first:.1f}s)")
        self.timed_out_levels.add(level_idx)

        # ---- Phase 2: Dynamic rescan (flood-fill games) ----
        exhausted_quickly = len(pq) == 0 and elapsed_first < self.bfs_timeout * 0.5
        if exhausted_quickly:
            logger.info(f"BFS L{level_idx}: queue exhausted early — dynamic rescan")
            visited_d = {self._state_hash(base_game, f0, transient_fields=transient_fields)}
            queue_d = deque([([], 0, base_game)])
            current_actions = list(actions)
            t0_d = time.time()
            explored_d = 0
            remaining_d = max(30, self.bfs_timeout - elapsed_first)

            while queue_d and explored_d < max_states * 10 and (time.time() - t0_d) < remaining_d:
                hist_d, depth_d, node_game_d = queue_d.popleft()
                for act_id, data in current_actions:
                    g2_d = _fast_deepcopy(node_game_d)
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g2_d.perform_action(ai, raw=True)
                    except:
                        continue
                    explored_d += 1
                    if not r.frame:
                        continue
                    f2 = np.array(r.frame[-1])
                    h_d = self._state_hash(g2_d, f2, transient_fields=transient_fields)
                    if h_d in visited_d:
                        continue
                    visited_d.add(h_d)
                    # Rescan from child for newly unlocked actions
                    try:
                        new_acts = self._scan_actions(g2_d, f2, bg)
                        added = [a for a in new_acts if a not in current_actions]
                        if added:
                            logger.info(f"BFS L{level_idx}: rescan found {len(added)} new actions")
                            current_actions.extend(added)
                    except:
                        pass
                    new_hist_d = hist_d + [(act_id, data)]
                    if (r.levels_completed > level_idx
                            or g2_d._current_level_index > level_idx):
                        logger.info(f"BFS L{level_idx}: SOLVED (dynamic rescan) in "
                                    f"{len(new_hist_d)} actions")
                        self.solutions[level_idx] = new_hist_d
                        return new_hist_d
                    if depth_d < 30:
                        queue_d.append((new_hist_d, depth_d + 1, g2_d))

        # ---- Phase 3: Hidden fields retry ----
        elapsed_p2 = time.time() - t0
        if (explored > 0 and (len(visited) < 200 or explored / len(visited) > 5)
                and elapsed_p2 < self.bfs_timeout * 0.8):
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                logger.info(f"BFS L{level_idx}: RETRY with hidden fields: {hidden_fields}")
                game2, last_r2 = self._init_game_at_level(level_idx)
                if game2 is None or not last_r2.frame:
                    return None
                f0_2 = np.array(last_r2.frame[-1])
                visited2 = {self._state_hash(game2, f0_2, hidden_fields,
                                              transient_fields=transient_fields)}
                queue2 = deque([([], 0, _fast_deepcopy(game2))])
                t0_2 = time.time()
                explored2 = 0
                remaining2 = max(30, self.bfs_timeout - elapsed_p2)

                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining2:
                    hist, depth, node_game2 = queue2.popleft()
                    for act_id, data in actions:
                        g2 = _fast_deepcopy(node_game2)
                        try:
                            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                                  if data else ActionInput(id=GameAction.from_id(act_id)))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields,
                                             transient_fields=transient_fields)
                        if h in visited2:
                            continue
                        visited2.add(h)
                        new_hist = hist + [(act_id, data)]
                        if (r.levels_completed > level_idx
                                or g2._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (hidden retry) in "
                                        f"{len(new_hist)} actions")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        if depth < 30:
                            queue2.append((new_hist, depth + 1, g2))

        # ---- Phase 4: IDDFS (deep directional games, low branching) ----
        elapsed_p3 = time.time() - t0
        remaining_iddfs = max(30, self.bfs_timeout - elapsed_p3)
        if len(actions) <= 6 and remaining_iddfs > 30:
            logger.info(f"BFS L{level_idx}: trying IDDFS (branching={len(actions)}, "
                        f"{remaining_iddfs:.0f}s remaining)")
            game3, _ = self._init_game_at_level(level_idx)
            if game3 is None:
                game3 = _fast_deepcopy(game)
            t0_iddfs = time.time()
            for max_depth in range(10, 60):
                if time.time() - t0_iddfs > remaining_iddfs:
                    break
                stack = [(_fast_deepcopy(game3), [], set())]
                while stack and (time.time() - t0_iddfs) < remaining_iddfs:
                    g, hist, path_hashes = stack.pop()
                    if len(hist) >= max_depth:
                        continue
                    for act_id, data in actions:
                        g2 = _fast_deepcopy(g)
                        try:
                            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                                  if data else ActionInput(id=GameAction.from_id(act_id)))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        fh = hash(f.tobytes())
                        if fh in path_hashes:
                            continue
                        new_hist = hist + [(act_id, data)]
                        if (r.levels_completed > level_idx
                                or g2._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (IDDFS d={max_depth}) "
                                        f"in {len(new_hist)} actions")
                            self.solutions[level_idx] = new_hist
                            return new_hist
                        stack.append((g2, new_hist, path_hashes | {fh}))
            logger.info(f"BFS L{level_idx}: IDDFS exhausted")

        # ---- Phase 5: Sprite permutation (click-only games with ≤8 targets) ----
        elapsed_p4 = time.time() - t0
        remaining_perm = max(20, self.bfs_timeout - elapsed_p4)
        click_actions = [a for a in actions if a[0] == 6]
        non_click = [a for a in actions if a[0] != 6]
        if not non_click and 1 <= len(click_actions) <= 8 and remaining_perm > 10:
            logger.info(f"BFS L{level_idx}: trying sprite permutation "
                        f"({len(click_actions)} clicks)")
            t0_perm = time.time()
            perm_timeout = min(60, remaining_perm)
            for perm in permutations(range(len(click_actions))):
                if time.time() - t0_perm > perm_timeout:
                    break
                g_perm = _fast_deepcopy(game)
                hist_perm = []
                for idx in perm:
                    act_id, data = click_actions[idx]
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g_perm.perform_action(ai, raw=True)
                        hist_perm.append((act_id, data))
                        if (r.levels_completed > level_idx
                                or g_perm._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (permutation) "
                                        f"in {len(hist_perm)} actions")
                            self.solutions[level_idx] = hist_perm
                            return hist_perm
                    except:
                        break
            logger.info(f"BFS L{level_idx}: permutation exhausted")

        # ---- Phase 5.5: Random click ordering for medium click-only games (9-20 targets) ----
        elapsed_p55 = time.time() - t0
        remaining_p55 = max(20, self.bfs_timeout - elapsed_p55)
        if not non_click and 9 <= len(click_actions) <= 20 and remaining_p55 > 10:
            logger.info(f"BFS L{level_idx}: random click ordering "
                        f"({len(click_actions)} clicks, {remaining_p55:.0f}s)")
            t0_rand = time.time()
            rand_timeout = min(30, remaining_p55)
            tried_perms = set()
            while time.time() - t0_rand < rand_timeout:
                perm = tuple(random.sample(range(len(click_actions)), len(click_actions)))
                if perm in tried_perms:
                    continue
                tried_perms.add(perm)
                g_rand = _fast_deepcopy(game)
                hist_rand = []
                solved = False
                for idx in perm:
                    act_id, data = click_actions[idx]
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g_rand.perform_action(ai, raw=True)
                        hist_rand.append((act_id, data))
                        if (r.levels_completed > level_idx
                                or g_rand._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (random ordering, "
                                        f"{len(tried_perms)} tries) in {len(hist_rand)} actions")
                            self.solutions[level_idx] = hist_rand
                            solved = True
                            return hist_rand
                    except:
                        break
                if solved:
                    break
            logger.info(f"BFS L{level_idx}: random ordering exhausted "
                        f"({len(tried_perms)} tries)")

        # ---- Phase 6: Beam search (medium branching, medium depth) ----
        elapsed_p5 = time.time() - t0
        remaining_bs = max(20, self.bfs_timeout - elapsed_p5)
        if 2 <= len(actions) <= 20 and remaining_bs > 20:
            logger.info(f"BFS L{level_idx}: trying beam search "
                        f"(branching={len(actions)}, {remaining_bs:.0f}s)")
            bw = min(200, max(20, max_states // (len(actions) * 50)))
            game_b = _fast_deepcopy(game)
            f0_b = f0
            beam = [(_fast_deepcopy(game_b), [])]
            vis_b = {self._state_hash(game_b, f0_b, transient_fields=transient_fields)}
            t0_b = time.time()

            for bd in range(60):
                if time.time() - t0_b > remaining_bs or not beam:
                    break
                cands = []
                for g_b, hist_b in beam:
                    for act_id, data in actions:
                        g2 = _fast_deepcopy(g_b)
                        try:
                            ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                                  if data else ActionInput(id=GameAction.from_id(act_id)))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, transient_fields=transient_fields)
                        if h in vis_b:
                            continue
                        vis_b.add(h)
                        nh = hist_b + [(act_id, data)]
                        if (r.levels_completed > level_idx
                                or g2._current_level_index > level_idx):
                            logger.info(f"BFS L{level_idx}: SOLVED (beam d={bd}) "
                                        f"in {len(nh)} actions")
                            self.solutions[level_idx] = nh
                            return nh
                        pdiff = float(np.sum(f != f0_b)) / 4096.0
                        h_val = hfn(f, g2)
                        score = pdiff + 1.0 / (1.0 + h_val)
                        cands.append((score, g2, nh))
                if not cands:
                    break
                cands.sort(key=lambda x: x[0], reverse=True)
                beam = [(g_b, h_b) for _, g_b, h_b in cands[:bw]]

            logger.info(f"BFS L{level_idx}: beam done ({len(vis_b)} unique, "
                        f"{time.time()-t0_b:.1f}s)")

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            # Direct replay
            g = _fast_deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                try:
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = prev_solution[:i + 1]
                        self.solutions[level_idx] = sol
                        logger.info(f"BFS L{level_idx}: TRANSFER (direct replay, {i+1} actions)")
                        return sol
                except:
                    break

            # Object-relative offset transfer
            prev_game = self.game_cls()
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
                    n = int(np.sum(mask))
                    if n < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)),
                                 'cy': float(np.mean(ys)), 'n': n})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)
            if not objs_prev or not objs_curr:
                return None

            matched = []
            for op in objs_prev:
                best, best_dist = None, float('inf')
                for oc in objs_curr:
                    if (oc['color'] == op['color']
                            and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5):
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))
            if not matched:
                return None

            dx = float(np.mean([m[1]['cx'] - m[0]['cx'] for m in matched]))
            dy = float(np.mean([m[1]['cy'] - m[0]['cy'] for m in matched]))

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = _fast_deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                          if data else ActionInput(id=GameAction.from_id(act_id)))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = transferred[:i + 1]
                        self.solutions[level_idx] = sol
                        logger.info(f"BFS L{level_idx}: TRANSFER (offset dx={dx:.0f},"
                                    f"dy={dy:.0f}, {i+1} actions)")
                        return sol
                except:
                    break

            # Action multiplier transfer
            for multiplier in [2, 3, 4]:
                expanded = []
                for act_id, data in prev_solution:
                    for _ in range(multiplier):
                        if data:
                            new_data = dict(data)
                            new_data['x'] = max(0, min(63, int(data.get('x', 32) + dx)))
                            new_data['y'] = max(0, min(63, int(data.get('y', 32) + dy)))
                            expanded.append((act_id, new_data))
                        else:
                            expanded.append((act_id, data))
                g = _fast_deepcopy(game)
                for i, (act_id, data) in enumerate(expanded):
                    try:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        r = g.perform_action(ai, raw=True)
                        if r.levels_completed > level_idx or g._current_level_index > level_idx:
                            sol = expanded[:i + 1]
                            self.solutions[level_idx] = sol
                            logger.info(f"BFS L{level_idx}: TRANSFER (x{multiplier}, "
                                        f"{i+1} actions)")
                            return sol
                    except:
                        break
        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None


def find_game_source_and_class(game_id, arc_env=None):
    import re
    parts = game_id.split('-', 1)
    gid = parts[0]
    guid_suffix = parts[1] if len(parts) > 1 else ''

    # Primary: competition-structured path
    competition_path = (
        f"/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
        f"/environment_files/{gid}/{guid_suffix}/{gid}.py"
    )
    if os.path.exists(competition_path):
        src = competition_path
        m = re.search(r'class\s+(\w+)\s*\(', open(src).read()[:2000])
        cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
        logger.info(f"BFS: found {src} class={cls_name}")
        return src, cls_name

    # Fallback: environment_info or glob
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for cand in [ld / f"{gid}.py", ld / f"{gid.upper()}.py"]:
                if cand.exists():
                    m = re.search(r'class\s+(\w+)\s*\(', cand.read_text()[:2000])
                    cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
                    return str(cand), cls_name

    for pattern in [f"/kaggle/input/**/{gid}.py", f"/tmp/**/{gid}.py",
                    f"/kaggle/working/**/{gid}.py"]:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            src = matches[0]
            m = re.search(r'class\s+(\w+)\s*\(', open(src).read()[:2000])
            cls_name = m.group(1) if m else gid[0].upper() + gid[1:]
            logger.info(f"BFS: found {src} class={cls_name}")
            return src, cls_name

    logger.warning(f"BFS: game source not found for {game_id}")
    return None, gid[0].upper() + gid[1:]


# ==================== CNN MODULES ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1 = nn.Linear(ch, max(ch // r, 4))
        s.fc2 = nn.Linear(max(ch // r, 4), ch)
        s.sp = nn.Conv2d(2, 1, 7, padding=3)

    def forward(s, x):
        B, C, H, W = x.shape
        w = torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2, 3])))))
        x = x * w.view(B, C, 1, 1)
        a = torch.sigmoid(s.sp(torch.cat(
            [x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)))
        return x * a


class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim = mem_dim
        s.diff_enc = nn.Sequential(
            nn.Conv2d(1, 8, 8, stride=8), nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 2 * 2, mem_dim))
        s.q_proj = nn.Linear(feat_dim, mem_dim)
        s.v_proj = nn.Linear(mem_dim + 1 + n_actions, n_actions)
        s.scale = mem_dim ** 0.5

    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B, M = mem_actions.shape
        if M == 0:
            return torch.zeros(B, 5, device=cnn_feat.device)
        keys = s.diff_enc(mem_diffs.reshape(B * M, 1, 64, 64)).reshape(B, M, s.mem_dim)
        q = s.q_proj(cnn_feat).unsqueeze(1)
        attn = F.softmax(torch.bmm(q, keys.transpose(1, 2)) / s.scale, dim=-1)
        act_oh = F.one_hot(mem_actions.clamp(0, 4), 5).float()
        vals = torch.cat([keys, mem_rewards.unsqueeze(-1), act_oh], dim=-1)
        ctx = torch.bmm(attn, vals).squeeze(1)
        return s.v_proj(ctx)


class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g = g
        s.c1 = nn.Conv2d(in_ch, 32, 3, padding=1)
        s.c2 = nn.Conv2d(32, 64, 3, padding=1)
        s.c3 = nn.Conv2d(64, 128, 3, padding=1)
        s.c4 = nn.Conv2d(128, 256, 3, padding=1)
        s.attn = CBAM(256)
        s.ar = nn.Conv2d(256, 64, 1)
        s.ap = nn.MaxPool2d(4, 4)
        s.af = nn.Linear(64 * 16 * 16, 256)
        s.ah = nn.Linear(256, 5)
        s.dr = nn.Dropout(0.15)
        s.cc1 = nn.Conv2d(256, 128, 3, padding=1)
        s.cc2 = nn.Conv2d(128, 64, 3, padding=1)
        s.cc3 = nn.Conv2d(64, 32, 1)
        s.cc4 = nn.Conv2d(32, 1, 1)
        s.gp = nn.AdaptiveAvgPool2d(1)
        s.gf = nn.Linear(256, 64)
        s.aea = ActionEffectAttention(feat_dim=64, mem_dim=32, n_actions=5)

    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x = F.relu(s.c1(x))
        x = F.relu(s.c2(x))
        x = F.relu(s.c3(x))
        f = F.relu(s.c4(x))
        f = s.attn(f)
        af = F.relu(s.ar(f))
        af = s.ap(af).reshape(f.size(0), -1)
        al = s.ah(s.dr(F.relu(s.af(af))))
        cf = F.relu(s.cc1(f))
        cf = F.relu(s.cc2(cf))
        cf = F.relu(s.cc3(cf))
        cl = s.cc4(cf).reshape(f.size(0), -1)
        if mem_diffs is not None and mem_actions is not None:
            gf = s.gf(s.gp(f).reshape(f.size(0), -1))
            al = al + s.aea(gf, mem_diffs, mem_actions, mem_rewards)
        return torch.cat([al, cl], 1)


def fast_objects(frame, bg):
    objs = []
    for c in range(16):
        if c == bg:
            continue
        mask = (frame == c)
        npix = int(np.sum(mask))
        if npix < 4 or npix > 3000:
            continue
        ys, xs = np.where(mask)
        objs.append((c, float(np.mean(xs)), float(np.mean(ys)), npix))
    return objs


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time() * 1e6) + hash(s.game_id) % 1000000
        random.seed(seed)
        np.random.seed(seed % (2 ** 32 - 1))
        torch.manual_seed(seed % (2 ** 32 - 1))

        s.start_time = time.time()
        s.device = torch.device(
            'cuda' if torch.cuda.is_available() else
            ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G = 64
        s.IN = 26
        s.net = None
        s.opt = None

        # Replay buffer — prioritized (recent + high-reward weighted)
        s.buf = deque(maxlen=50000)
        s.buf_h = set()
        s.bsz = 64
        s.tfreq = 10

        # Per-step state tracking
        s.pt = None
        s.pai = None
        s.pr = None
        s.ph = None
        s.cl = -1
        s.fhist = deque(maxlen=6)
        s.la = 0

        s.al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3,
                GameAction.ACTION4, GameAction.ACTION5]
        s._wd = False
        s._bg = 0
        s._wm = None

        # AEA memory buffers
        s._aem_diffs = deque(maxlen=256)
        s._aem_actions = deque(maxlen=256)
        s._aem_rewards = deque(maxlen=256)

        # Exploration state
        s._ckpt_hash = None
        s._unproductive = 0
        s._undo_avail = False
        s._eps = 0.15
        s._eps_min = 0.03
        s._eps_decay = 0.9997
        # FIX: properly initialize _visited_hashes (was causing reward bug in older versions)
        s._visited_hashes = set()
        s._state_visit_counts = defaultdict(int)

        # Object movement tracking (for dense rewards)
        s._prev_objs = None
        s._obj_moved = 0

        # BFS solver
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        s._bfs_solved_last = False  # FIX: track if BFS solved previous level
        s._clti_demos = []

        # Scanned actions for CNN click masking (from op_5)
        s._scanned_actions = None
        s._visit_counts = defaultdict(int)  # for novelty-guided exploration

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES:
            s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid:
            s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json
            s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f):
        return getattr(f, 'score', None) or f.levels_completed

    def _raw(s, fd):
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id,
                                              s.arc_env if hasattr(s, 'arc_env') else None)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning("BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")

    def _capture_clti_demos(s, level_idx, sol):
        """Replay BFS solution and record (frame_before, action_idx, reward=2.0) tuples."""
        try:
            g, last_r = s._bfs._init_game_at_level(level_idx)
            if g is None or not last_r or not last_r.frame:
                return []
            demos = []
            for act_id, data in sol:
                frame_before = np.array(last_r.frame[-1], dtype=np.int64)
                if act_id <= 5:
                    pai = act_id - 1
                elif data and 'x' in data and 'y' in data:
                    pai = 5 + int(data['y']) * 64 + int(data['x'])
                else:
                    continue
                ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                      if data else ActionInput(id=GameAction.from_id(act_id)))
                last_r = g.perform_action(ai, raw=True)
                if not last_r.frame:
                    break
                demos.append({'s': frame_before, 'a': pai, 'r': 2.0})
            return demos
        except Exception as e:
            logger.warning(f"CLTI capture failed: {e}")
            return []

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None:
            return None

        # Adaptive time budget: if BFS solved the previous level, give it more time
        elapsed = time.time() - s.start_time
        total_budget = 6 * 3600 - 600
        remaining = max(60, total_budget - elapsed)
        if level_idx == 0:
            time_for_bfs = min(remaining * 0.35, 1200)
        elif s._bfs_solved_last:
            time_for_bfs = min(remaining * 0.20, 480)
        else:
            time_for_bfs = min(remaining * 0.08, 180)
        time_for_bfs = max(30, time_for_bfs)
        s._bfs.bfs_timeout = int(time_for_bfs)
        logger.info(f"BFS L{level_idx}: budget={time_for_bfs:.0f}s "
                    f"(remaining={remaining:.0f}s)")

        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None

        # Build goal heuristic from previous level solutions
        goal_heuristic = None
        if level_idx > 0 and s._bfs.game_cls is not None:
            try:
                g = s._bfs.game_cls()
                g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r = g.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                level_heuristics = []
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps:
                        break
                    f_init = np.array(last_r.frame[-1])
                    for act_id, data in ps:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        last_r = g.perform_action(ai, raw=True)
                    f_win = np.array(last_r.frame[-1])
                    hfn = s._bfs._build_goal_heuristic(f_init, f_win)
                    level_heuristics.append((hfn, pi + 1))
                if level_heuristics:
                    total_w = sum(w for _, w in level_heuristics)
                    def goal_heuristic(f, game=None,
                                       _h=level_heuristics, _t=total_w):
                        return sum(hfn(f, game) * w for hfn, w in _h) / _t
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: goal heuristic build failed: {e}")

        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol,
                                  goal_heuristic=goal_heuristic)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            s._bfs_solved_last = True
            s._clti_demos = s._capture_clti_demos(level_idx, sol)
            return sol

        # Retry with distance heuristic if flat
        if (level_idx in s._bfs.timed_out_levels
                and s._bfs.game_cls is not None):
            try:
                g_val = s._bfs.game_cls()
                g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                last_r_val = g_val.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                for pi in range(level_idx):
                    ps = s._bfs.solutions.get(pi)
                    if not ps:
                        break
                    for act_id, data in ps:
                        ai = (ActionInput(id=GameAction.from_id(act_id), data=data)
                              if data else ActionInput(id=GameAction.from_id(act_id)))
                        last_r_val = g_val.perform_action(ai, raw=True)
                mover_colors, target_colors = s._bfs._probe_mover_target_colors(g_val)
                if mover_colors and target_colors:
                    def dist_heuristic(f, game=None,
                                       _m=mover_colors, _t=target_colors):
                        centroids = {}
                        for c in range(16):
                            mask = (f == c)
                            n = int(np.sum(mask))
                            if n < 2:
                                continue
                            ys, xs = np.where(mask)
                            centroids[c] = (float(np.mean(xs)), float(np.mean(ys)))
                        targets = [(centroids[tc][0], centroids[tc][1])
                                   for tc in _t if tc in centroids]
                        if not targets:
                            return 0
                        return sum(
                            min(abs(centroids[mc][0] - tx) + abs(centroids[mc][1] - ty)
                                for tx, ty in targets)
                            for mc in _m if mc in centroids)
                    logger.info(f"BFS L{level_idx}: retrying with distance heuristic")
                    sol2 = s._bfs.solve_level(level_idx, prev_solution=prev_sol,
                                              goal_heuristic=dist_heuristic)
                    if sol2:
                        s._bfs_solution = sol2
                        s._bfs_step = 0
                        s._bfs_solved_last = True
                        s._clti_demos = s._capture_clti_demos(level_idx, sol2)
                        return sol2
            except Exception as e:
                logger.warning(f"BFS L{level_idx}: distance heuristic retry failed: {e}")

        s._bfs_solved_last = False
        return None

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        s._bg = int(cnt.argmax())
        mx = max(cnt.max(), 1)
        bg_m = (frame == s._bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) |
                (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        d1 = torch.zeros(3, 64, 64, dtype=torch.float32)
        for i, prev in enumerate(reversed(list(s.fhist))):
            if i >= 3:
                break
            d1[i] = torch.from_numpy((frame != prev).astype(np.float32))
        d2 = torch.zeros(2, 64, 64, dtype=torch.float32)
        h = list(s.fhist)
        if len(h) >= 2:
            d2[0] = torch.from_numpy((h[-1] != h[-2]).astype(np.float32))
        if len(h) >= 4:
            d2[1] = torch.from_numpy((h[-2] != h[-4]).astype(np.float32))
        s.fhist.append(frame.copy())
        return torch.cat([oh, aug, d1, d2], 0).to(s.device)

    def _detect_template(s, frame):
        mask = torch.ones(4096, dtype=torch.float32)
        col_act = np.sum(frame != s._bg, axis=0)
        for c in range(20, 44):
            if (col_act[c] <= 2 and np.sum(col_act[:c] > 0) >= 5
                    and np.sum(col_act[c + 1:] > 0) >= 5):
                for y in range(64):
                    for x in range(c + 1):
                        mask[y * 64 + x] = 0.05
                return mask
        row_act = np.sum(frame != s._bg, axis=1)
        for r in range(20, 44):
            if (row_act[r] <= 2 and np.sum(row_act[:r] > 0) >= 5
                    and np.sum(row_act[r + 1:] > 0) >= 5):
                for y in range(r + 1):
                    for x in range(64):
                        mask[y * 64 + x] = 0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        mask = np.ones((64, 64), dtype=bool)
        mask[:2] = False
        mask[62:] = False
        diff = (prev_raw != curr_raw) & mask
        changed = np.any(diff)
        r = 0.0
        if curr_h != prev_h:
            if curr_h not in s._visited_hashes:
                r += 1.5
                s._visited_hashes.add(curr_h)
            else:
                r += 0.2
        else:
            r -= 0.1
        if changed:
            r += 0.5
        curr_objs = fast_objects(curr_raw, s._bg)
        if s._prev_objs and curr_objs:
            moved = 0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0] == po[0]:
                        dist = abs(co[1] - po[1]) + abs(co[2] - po[2])
                        if 2 < dist < 20:
                            moved += 1
                            break
            if moved > 0:
                r += 0.3 * min(moved, 3)
                s._obj_moved = moved
        s._prev_objs = curr_objs
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al = logits[:5].clone()
        cl = logits[5:5 + 4096].clone()
        if avail is not None and len(avail) > 0:
            mask_al = torch.full_like(al, float('-inf'))
            a6 = False
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                if 1 <= aid <= 5:
                    mask_al[aid - 1] = 0.0
                elif aid == 6:
                    a6 = True
            al = al + mask_al
            if not a6:
                cl = cl + torch.full_like(cl, float('-inf'))
        # Template masking
        if s._wm is not None:
            cl = cl + torch.log(s._wm.to(s.device).clamp(min=0.01))
        # Click masking: only predict positions we know are effective (from op_5)
        if s._scanned_actions is not None:
            click_mask = torch.full((4096,), -5.0, device=s.device)
            for act_id, data in s._scanned_actions:
                if act_id == 6 and data:
                    x, y = data.get('x', 0), data.get('y', 0)
                    if 0 <= x < 64 and 0 <= y < 64:
                        click_mask[y * 64 + x] = 0.0
            cl = cl + click_mask
        ap = torch.sigmoid(al / temp)
        cp = torch.sigmoid(cl / temp) / (s.G * s.G)
        allp = torch.cat([ap, cp])
        sm = allp.sum()
        if sm < 1e-8:
            allp = torch.ones_like(allp) / len(allp)
        else:
            allp = allp / sm
        idx = np.random.choice(len(allp), p=allp.cpu().numpy())
        if idx < 5:
            return idx, None
        ci = idx - 5
        return 5, (ci // s.G, ci % s.G)

    def _sample_novelty_guided(s, frame, avail):
        """Exploration: pick from scanned actions weighted by inverse visit count."""
        if not s._scanned_actions:
            return s._heuristic(frame, avail, s.la)
        scored = []
        for act_id, data in s._scanned_actions:
            if data:
                key = f"{act_id}:{data.get('x', 0)}:{data.get('y', 0)}"
            else:
                key = str(act_id)
            cnt = s._visit_counts[key]
            score = 1.0 / math.sqrt(cnt + 1)
            scored.append((score, act_id, data))
        scored.sort(reverse=True)
        probs = np.array([x[0] for x in scored], dtype=np.float64)
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(scored), p=probs))
        _, act_id, data = scored[idx]
        key = (f"{act_id}:{data.get('x', 0)}:{data.get('y', 0)}"
               if data else str(act_id))
        s._visit_counts[key] += 1
        if act_id < 6:
            return act_id - 1, None
        return 5, (data['y'], data['x'])

    def _heuristic(s, frame, avail, step):
        av = set(int(a.value) if hasattr(a, 'value') else int(a) for a in avail)
        for d in [1, 2, 3, 4]:
            if d in av and step < 4:
                return d - 1, None
        if 6 in av:
            cnt = np.bincount(frame.flatten(), minlength=16)
            targets = []
            for c in range(16):
                if c == s._bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys, xs = np.where(frame == c)
                if len(ys) >= 2:
                    targets.append((int(np.median(xs)), int(np.median(ys)), len(ys)))
            targets.sort(key=lambda t: t[2])
            pidx = step - 4
            if 0 <= pidx < len(targets):
                return 5, (targets[pidx][1], targets[pidx][0])
        if 5 in av:
            return 4, None
        choices = [a for a in av if 1 <= a <= 5]
        if choices:
            return random.choice(choices) - 1, None
        return 0, None

    def _frame_to_tensor(s, frame):
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        bg = int(cnt.argmax())
        mx = max(cnt.max(), 1)
        bg_m = (frame == bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) |
                (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        zeros = torch.zeros(5, 64, 64, dtype=torch.float32)
        return torch.cat([oh, aug, zeros], 0)

    def _train(s):
        if len(s.buf) < s.bsz:
            return
        # Prioritized replay: weight recent + high-reward transitions more
        weights = np.array([abs(e['r']) + 0.1 for e in s.buf])
        n = len(weights)
        weights[max(0, n - 100):] *= 2.0
        weights /= weights.sum()
        indices = np.random.choice(n, s.bsz, replace=False, p=weights)
        batch = [s.buf[i] for i in indices]
        states = torch.stack(
            [s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts = torch.tensor([e['a'] for e in batch],
                            dtype=torch.long, device=s.device)
        rews = torch.tensor([e['r'] for e in batch],
                            dtype=torch.float32, device=s.device)
        rews = torch.sigmoid(rews)
        s.opt.zero_grad()
        logits = s.net(states)
        acts_c = acts.clamp(0, logits.size(1) - 1)
        sel = logits.gather(1, acts_c.unsqueeze(1)).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(sel, rews)
        p = torch.sigmoid(logits)
        loss = loss - 0.0001 * p[:, :5].mean() - 0.00001 * p[:, 5:].mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s.net.parameters(), 1.0)
        s.opt.step()

    def _get_aem_tensors(s):
        if len(s._aem_diffs) < 2:
            return None, None, None
        M = len(s._aem_diffs)
        diffs = torch.zeros(1, M, 1, 64, 64, device=s.device)
        acts = torch.zeros(1, M, dtype=torch.long, device=s.device)
        rews = torch.zeros(1, M, device=s.device)
        for i, (d, a, r) in enumerate(
                zip(s._aem_diffs, s._aem_actions, s._aem_rewards)):
            diffs[0, i, 0] = torch.from_numpy(d.astype(np.float32))
            acts[0, i] = min(a, 4)
            rews[0, i] = r
        return diffs, acts, rews

    def is_done(s, frames, lf):
        try:
            return (lf.state is GameState.WIN
                    or (time.time() - s.start_time) >= 6 * 3600 - 300)
        except:
            return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                # Save CLTI demos from previous level BEFORE running BFS for new level
                clti_to_inject = s._clti_demos
                s._clti_demos = []

                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)

                # Scan actions for CNN click masking (from op_5)
                s._scanned_actions = None
                s._visit_counts = defaultdict(int)
                if s._bfs is not None and s._bfs.game_cls is not None:
                    try:
                        g_scan = s._bfs.game_cls()
                        g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        g_scan.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        raw_init = s._raw(lf)
                        bg = int(np.bincount(raw_init.flatten(), minlength=16).argmax())
                        s._scanned_actions = s._bfs._scan_actions(g_scan, raw_init, bg)
                        logger.info(f"CNN: scanned {len(s._scanned_actions)} effective actions")
                    except Exception as e:
                        logger.warning(f"CNN action scan failed: {e}")

                # Init CNN
                s.buf.clear()
                s.buf_h.clear()
                # CLTI: inject previous level's BFS expert demos
                for demo in clti_to_inject:
                    key = hashlib.md5(demo['s'].tobytes() + str(demo['a']).encode()).hexdigest()[:16]
                    if key not in s.buf_h:
                        s.buf.append(demo)
                        s.buf_h.add(key)
                if clti_to_inject:
                    logger.info(f"CLTI: injected {len(clti_to_inject)} expert demos for L{lvl}")
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                           'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state = torch.load(wp, map_location=s.device, weights_only=True)
                            ms = s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape == ms[k].shape:
                                    ms[k] = state[k]
                            s.net.load_state_dict(ms)
                            break
                    except:
                        pass
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003, weight_decay=1e-5)
                s.pt = None
                s.pai = None
                s.pr = None
                s.ph = None
                s.cl = lvl
                s.fhist.clear()
                s.la = 0
                s._wd = False
                s._wm = None
                # FIX: only reset epsilon if BFS failed (don't waste good exploration)
                if not s._bfs_solved_last:
                    s._eps = 0.15
                s._aem_diffs.clear()
                s._aem_actions.clear()
                s._aem_rewards.clear()
                s._prev_objs = None
                s._obj_moved = 0
                s._ckpt_hash = None
                s._unproductive = 0
                s._visited_hashes = set()
                s._state_visit_counts = defaultdict(int)

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt = None
                s.pai = None
                s.pr = None
                s.ph = None
                a = GameAction.RESET
                a.reasoning = "reset"
                return a

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

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any(
                (a.value if hasattr(a, 'value') else int(a)) == 7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask = np.ones((64, 64), dtype=bool)
                mask[:2] = False
                mask[62:] = False
                diff_map = (s.pr != raw) & mask
                changed = np.any(diff_map)
                eh = hashlib.md5(
                    s.pr.tobytes()[:1000] + str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r = s._reward(s.pr, raw, s.ph, ch)
                    # Intrinsic reward: bonus for visiting novel states
                    state_key = ch
                    intrinsic = 0.5 / math.sqrt(s._state_visit_counts[state_key] + 1)
                    s._state_visit_counts[state_key] += 1
                    r = r + intrinsic
                    s.buf.append({'s': s.pr.copy(), 'a': s.pai, 'r': r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai, 4))
                        s._aem_rewards.append(r)
                if changed:
                    s._ckpt_hash = ch
                    s._unproductive = 0
                else:
                    s._unproductive += 1

            if s._wm is None:
                s._wm = s._detect_template(raw)

            if s._undo_avail and s._unproductive >= 30 and s._ckpt_hash:
                s._unproductive = 0
                a = GameAction.ACTION7
                a.reasoning = "undo"
                s.pt = tensor
                s.pai = 6
                s.pr = raw.copy()
                s.ph = ch
                s.la += 1
                return a

            if not s._wd:
                if s.la < 10:
                    # Novelty-guided exploration (from op_5)
                    aidx, coords = s._sample_novelty_guided(raw, avail)
                else:
                    s._wd = True
                    for _ in range(min(5, len(s.buf) // s.bsz)):
                        s._train()

            if s._wd:
                if random.random() < s._eps:
                    aidx, coords = s._sample_novelty_guided(raw, avail)
                else:
                    with torch.no_grad():
                        mem = s._get_aem_tensors()
                        if mem[0] is not None:
                            logits = s.net(tensor.unsqueeze(0), *mem).squeeze(0)
                        else:
                            logits = s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx, coords = s._sample(logits, avail, temp=0.5)
                s._eps = max(s._eps_min, s._eps * s._eps_decay)
            elif s.la >= 10:
                s._wd = True
                aidx, coords = 0, None

            if aidx < 5:
                sel = s.al[aidx]
                sel.reasoning = f"cnn:a{aidx + 1}"
            else:
                sel = GameAction.ACTION6
                y, x = coords
                sel.set_data({"x": int(x), "y": int(y)})
                sel.reasoning = f"cnn:c({x},{y})"

            s.pt = tensor
            s.pai = aidx if aidx < 5 else (5 + coords[0] * s.G + coords[1])
            s.pr = raw.copy()
            s.ph = ch
            s.la += 1
            if s.action_counter % s.tfreq == 0 and s._wd:
                s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a = random.choice(s.al)
            a.reasoning = f"err:{str(e)[:40]}"
            return a