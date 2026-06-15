"""v18 genuine solver — honest black-box best-first search.

This is v17's BFS/BFWS playbook with the cheating removed:
  * NO stored answers / harvest / v13 cache — it discovers the path itself.
  * NO engine `__dict__` progress signal — the ONLY goal signal is the observable
    `levels_completed` ticking up; dedup/novelty use the observable FRAME only.
  * NO in-process snapshot fork — it explores the ONE real env it is given by
    RESET + replaying an action prefix (exactly what you can do through the
    official API on a hidden game). Every probe is a real action and is counted.

Because it solves rather than recalls, it works on games it has never seen —
that is the whole point. Transferable pieces of the v17 playbook kept:
  * macro-actions ("repeat a move until the frame stops changing") — collapses
    corridors so search depth (in macro-steps) stays small.
  * frame dedup with PIXEL-level transient masking (frame-only) — kills the
    per-step counter/animation pixels that would otherwise make every state look
    novel and explode the search.
  * BFS over macro-nodes, depth limited in MACRO-STEPS (not raw actions).

Reported honestly: `actions_used` = real actions spent searching (the real-world
cost); `solution_len` = the short deployable path once found.
"""
from __future__ import annotations
from collections import deque
import hashlib
import numpy as np
from blackbox_env import MOVES, RESET, CLICK


def _click_targets(frame, limit=16):
    """Click targets from the VISIBLE frame — one per connected component of each
    non-background colour (so multiple same-colour objects each get their own
    click), largest first. Frame-only, honest. Click games (available=(6,)) need
    this density; per-colour centroids alone collapse distinct objects."""
    H, W = frame.shape
    flat = frame.flatten()
    bg = int(np.bincount(flat, minlength=16).argmax())
    seen = np.zeros((H, W), dtype=bool)
    comps = []
    for y in range(H):
        for x in range(W):
            if seen[y, x] or frame[y, x] == bg:
                continue
            c = frame[y, x]
            stack = [(y, x)]; seen[y, x] = True; xs = []; ys = []
            while stack:
                cy, cx = stack.pop(); xs.append(cx); ys.append(cy)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and frame[ny, nx] == c:
                        seen[ny, nx] = True; stack.append((ny, nx))
            comps.append((len(xs), int(sum(xs) / len(xs)), int(sum(ys) / len(ys))))
    comps.sort(key=lambda t: -t[0])           # largest components first
    return [(CLICK, {"x": cx, "y": cy, "game_id": "v18"})
            for _, cx, cy in comps[:limit]]


class SearchAgent:
    name = "search"
    drives_env = True   # the eval harness hands it the env and calls solve()

    def __init__(self, macro_cap=24, max_depth=40, seed=0, **kw):
        self.macro_cap = macro_cap        # max raw steps inside one macro (macro mode)
        self.max_depth = max_depth        # frontier depth headroom (rollout mode)

    # ---- frame-only state identity (transient pixels masked) ---------------
    def _probe_transient_mask(self, env):
        """Pixels that change under EVERY probe move are HUD/timer/counter —
        mask them so they don't (a) prevent wall detection or (b) make every
        state look novel. Frame-only; costs a handful of real actions."""
        base = env.reset()
        f0 = base.frame
        always = np.ones_like(f0, dtype=bool)
        used = 0
        for a in MOVES:
            env.reset()
            r = env.step(a); used += 1
            if r.frame.shape == f0.shape:
                always &= (r.frame != f0)
        return (always if always.any() else None), used

    def _h(self, frame, mask):
        f = frame
        if mask is not None:
            f = frame.copy(); f[mask] = 0
        return hashlib.md5(f.tobytes()).hexdigest()[:16]

    # ---- macro step: repeat a move until the (masked) frame stops changing --
    def _macro(self, env, action, data, mask):
        """Step `action` repeatedly until the masked frame stops changing (wall),
        a level is gained, or the game ends (or cap). Returns (raw_steps, obs)."""
        steps = [(action, data)]
        obs = env.step(action, data)
        if action == CLICK:
            return steps, obs
        prev = self._h(obs.frame, mask)
        start_levels = obs.levels_completed
        n = 1
        while n < self.macro_cap and not obs.done:
            o2 = env.step(action, data); n += 1
            if o2.levels_completed != start_levels:
                steps.append((action, data)); obs = o2; break
            h2 = self._h(o2.frame, mask)
            if h2 == prev:                 # masked frame unchanged -> hit a wall
                break                      # don't append the no-op step
            steps.append((action, data)); obs = o2; prev = h2
        return steps, obs

    def _replay(self, env, raw_path):
        obs = env.reset()
        for a, d in raw_path:
            obs = env.step(a, d)
        return obs, len(raw_path)

    def _candidates(self, obs):
        avail = list(obs.available_actions or MOVES)
        # simple actions (moves + ACTION5/ACTION7 etc.), excluding click/reset
        cands = [(a, None) for a in avail if a not in (CLICK, RESET)]
        if CLICK in avail:
            cands += _click_targets(obs.frame)
        # only fall back to raw MOVES if the game truly exposed nothing usable
        return cands or [(a, None) for a in (avail or MOVES) if a != RESET] or [(a, None) for a in MOVES]

    @staticmethod
    def _akey(a, d):
        """Hashable key for a candidate action (clicks distinguished by x,y)."""
        if d and "x" in d:
            return (a, d["x"], d["y"])
        return (a,)

    # ---- the search: rollout-based (v17-MCTS-lite, but honest) -------------
    # Honest reset+replay BFS costs O(nodes x depth) REAL actions (replay every
    # sibling) — intractable. Instead, like v17's imagination-MCTS: pick a
    # promising frontier node, RESET+replay to it ONCE, then roll a trajectory
    # FORWARD with light playouts, banking every newly-seen frame as a new
    # frontier node. One replay (depth) buys a whole rollout of new states.
    def solve(self, env, budget=150000, rollout_len=60, explore_p=0.30, seed=0):
        import random
        rng = random.Random(seed)
        root = env.reset()
        mask, used = self._probe_transient_mask(env)
        start_levels = root.levels_completed
        best = {"levels": 0, "win": False, "actions_to_first_level": None,
                "solution_len": None}
        solution = None
        r0, n = self._replay(env, []); used += n
        visited = {self._h(r0.frame, mask)}
        # Go-Explore archive of EXPLORE choices: frame-hash -> set of action keys
        # already tried from that observed state. Steers rollouts toward untried
        # actions (directed exploration) instead of re-treading randomly.
        tried = {}
        cand_cache = {}        # frame-hash -> candidate actions (component clicks are costly)
        # frontier node = (path, depth, levels_at_node)
        frontier = [([], 0, root.levels_completed)]
        sims = 0

        while frontier and used < budget:
            # best-first with exploration: usually expand the most-progressed,
            # shallowest node; sometimes a random one so a plateau can't starve.
            if rng.random() < explore_p:
                idx = rng.randrange(len(frontier))
            else:
                idx = max(range(len(frontier)),
                          key=lambda i: (frontier[i][2], -frontier[i][1]))
            path, depth, _ = frontier.pop(idx)
            cur, n = self._replay(env, path); used += n
            sims += 1
            cur_path = list(path)
            deaths = 0
            for t in range(rollout_len):
                if used >= budget:
                    break
                if cur.done:
                    if cur.state == "WIN" or deaths >= 2:
                        break
                    # GAME_OVER: return to the node and keep exploring (the tried
                    # archive will steer to a different action) instead of letting
                    # the rollout die early and starve the frontier.
                    deaths += 1
                    cur, n = self._replay(env, path); used += n
                    cur_path = list(path)
                    continue
                ch = self._h(cur.frame, mask)
                cands = cand_cache.get(ch)
                if cands is None:
                    cands = self._candidates(cur)
                    cand_cache[ch] = cands
                ts = tried.setdefault(ch, set())
                # Priority: untried SIMPLE (movement/ACTION5/7) > untried CLICK >
                # any simple > anything. Keeps movement games movement-focused (so
                # adding clicks for click-games doesn't dilute them), while pure
                # click games (no simple actions) naturally fall through to clicks.
                ut_simple = [c for c in cands if c[0] != CLICK and self._akey(c[0], c[1]) not in ts]
                ut_click = [c for c in cands if c[0] == CLICK and self._akey(c[0], c[1]) not in ts]
                simple = [c for c in cands if c[0] != CLICK]
                if rng.random() < 0.10:
                    a, d = rng.choice(cands)
                else:
                    for bucket in (ut_simple, ut_click, simple, cands):
                        if bucket:
                            a, d = rng.choice(bucket); break
                ts.add(self._akey(a, d))
                cur = env.step(a, d); used += 1
                cur_path.append((a, d))
                gained = cur.levels_completed - start_levels
                if gained > best["levels"] or (gained == best["levels"]
                                               and cur.state == "WIN" and not best["win"]):
                    best["levels"] = gained
                    best["win"] = cur.state == "WIN"
                    solution = list(cur_path)
                    best["solution_len"] = len(cur_path)
                    if best["actions_to_first_level"] is None and gained > 0:
                        best["actions_to_first_level"] = used
                hh = self._h(cur.frame, mask)
                if hh not in visited and not cur.done and depth + 1 + t <= self.max_depth + rollout_len:
                    visited.add(hh)
                    frontier.append((list(cur_path), depth + 1 + t, cur.levels_completed))

        best["actions_used"] = used
        best["sims"] = sims
        best["solution"] = solution
        return best
