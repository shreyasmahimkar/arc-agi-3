# =====================================================================
# v20.3 — graph-based frontier exploration (arXiv:2512.24156), white-box.
#   - object segmentation: connected-color components -> 5-20 click targets
#     (not 4096 pixels), 5 salience tiers; status bars lowest priority
#   - persistent state graph, nodes = hashed masked frames
#   - priority-first frontier expansion over engine SNAPSHOTS (unbounded,
#     zero real-action cost — the white-box edge over the paper's live agent)
# Pure exploration: the engine's levels_completed increment is the only signal.
# =====================================================================
import time, heapq, hashlib, pickle, zlib
import numpy as np
try:
    from scipy import ndimage
    _SCIPY = True
except Exception:
    _SCIPY = False
from arcengine import GameAction, ActionInput


def _snap(g): return zlib.compress(pickle.dumps(g, -1), 1)
def _rest(b): return pickle.loads(zlib.decompress(b))
def _nhash(f):
    fm = f.copy(); fm[:2] = 0; fm[-2:] = 0                 # mask status bars
    return hashlib.md5(fm.tobytes()).hexdigest()[:16]


def _label(mask):
    if _SCIPY:
        return ndimage.label(mask)
    # numpy BFS fallback (4-connectivity)
    H, W = mask.shape
    lbl = np.zeros((H, W), np.int32); n = 0
    from collections import deque
    for y in range(H):
        for x in range(W):
            if mask[y, x] and lbl[y, x] == 0:
                n += 1; q = deque([(y, x)]); lbl[y, x] = n
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and lbl[ny, nx] == 0:
                            lbl[ny, nx] = n; q.append((ny, nx))
    return lbl, n


def _actions_of(f, avail):
    """Object-segmented, salience-tiered action list: (tier, spec)."""
    acts = [(0, ("move", a)) for a in avail if a != 6]     # moves = top priority
    if 6 in avail:
        bg = int(np.bincount(f.flatten(), minlength=16).argmax())
        H, W = f.shape
        for c in range(16):
            m = (f == c)
            if c == bg or not m.any():
                continue
            lbl, n = _label(m)
            for i in range(1, n + 1):
                ys, xs = np.where(lbl == i); sz = len(ys)
                cy, cx = int(ys.mean()), int(xs.mean())
                tier = 5 if (cy < 2 or cy >= H - 2) else (
                    1 if sz > 80 else 2 if sz > 25 else 3 if sz > 8 else 4)
                acts.append((tier, ("click", cx, cy)))
    acts.sort(key=lambda x: x[0])
    return acts


def _do(g, act):
    if act[0] == "move":
        return g.perform_action(ActionInput(id=GameAction.from_id(act[1])), raw=True)
    return g.perform_action(ActionInput(id=GameAction.ACTION6,
                            data={"x": act[1], "y": act[2], "game_id": "gx"}), raw=True)


def graph_solve(game, level, f0, avail, budget=120, max_nodes=200000):
    """Explore from `game` (positioned at `level` start, current frame f0) until a
    state with levels_completed>level is reached. Returns the action list (in the
    [(action_id, data)] format used by the cache) or None."""
    root = _snap(game); seen = {_nhash(f0)}; ctr = 0; heap = []
    for tier, act in _actions_of(f0, avail):
        ctr += 1; heapq.heappush(heap, (tier, ctr, root, [act]))
    t0 = time.time(); expl = 0; nodes = 1
    while heap and time.time() - t0 < budget and nodes < max_nodes:
        tier, _, sn, path = heapq.heappop(heap)
        g = _rest(sn)
        r = _do(g, path[-1]); expl += 1
        cf = np.array(r.frame[-1]) if r.frame else None
        if cf is None:
            continue
        if r.levels_completed > level or g._current_level_index > level:
            return [_encode(a) for a in path]
        h = _nhash(cf)
        if h in seen:                                      # loop / no-op / reset -> tested
            continue
        seen.add(h); nodes += 1
        csn = _snap(g)
        for ct, ca in _actions_of(cf, avail):
            ctr += 1; heapq.heappush(heap, (ct, ctr, csn, path + [ca]))
    return None


def _encode(act):
    # cache format: (action_id, data_or_None)
    if act[0] == "move":
        return [act[1], None]
    return [6, {"x": act[1], "y": act[2], "game_id": "gx"}]
