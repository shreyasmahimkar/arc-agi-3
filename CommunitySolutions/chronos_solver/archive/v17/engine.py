"""v17 engine driver — the single ground-truth verifier.

Loads the REAL ls20 engine (arcengine + environment_files/.../ls20.py) and
drives it. Works in two environments transparently:
  * Mac venv312  : real pydantic + real numpy, full speed.
  * Linux sandbox: no network -> real pydantic absent -> _pydantic_shim is
    auto-installed; arcengine is pure-python so it runs unchanged.

Everything the solver needs to know about a game lives here:
  * load_game()            -> instantiate the engine
  * chain_to_level()       -> v13's "true baseline": replay cached L0..N-1
                              solutions so level N starts from the REAL state
                              (set_level(N)+RESET gives a DIFFERENT, wrong
                              start — v13 bug #4).
  * state_hash()           -> masked-frame + scalar-attr identity (v13 #2,#3)
  * object_features()      -> object-centric vector for ForgeNet / TRM
  * dynamic_clicks()       -> click targets from current-frame centroids (v13)
  * rhae_score()           -> RHAE proxy = sum min(1, base/actions)^2

All logging routed through vlog so a run is fully reconstructable.
"""
from __future__ import annotations
import sys, os, json, importlib.util, hashlib, copy, time, random
import numpy as np

# --- make the shim importable & install pydantic if the real one is gone ---
sys.path.insert(0, os.path.dirname(__file__))
import _pydantic_shim


def _bootstrap_arcengine():
    """Ensure `arcengine` is importable. On the Mac venv it already is. In the
    network-less Linux sandbox we vendor a symlink to the venv's pure-python
    arcengine into a clean /tmp dir (so the sandbox's own numpy is used, NOT
    the Mac-compiled one sitting next to it in site-packages)."""
    try:
        import arcengine  # noqa: F401
        return
    except Exception:
        pass
    import glob
    here = os.path.abspath(os.path.dirname(__file__))
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    hits = glob.glob(os.path.join(root, ".venv*", "lib", "python*", "site-packages", "arcengine"))
    if not hits:
        raise ImportError("arcengine not importable and no venv copy found")
    src = hits[0]
    # namespace per-uid so a stale dir owned by another sandbox session can't
    # block us with permission errors (iter6 fix).
    clean = f"/tmp/v17_aelib_{os.getuid()}"
    os.makedirs(clean, exist_ok=True)
    link = os.path.join(clean, "arcengine")
    # os.path.exists is False for a BROKEN symlink (points at a Mac path that
    # doesn't resolve in the sandbox) — use lexists and clear stale links.
    if os.path.lexists(link) and not os.path.exists(link):
        try:
            os.unlink(link)
        except OSError:
            pass
    if not os.path.exists(link):
        try:
            os.symlink(src, link)
        except OSError:
            import shutil
            shutil.copytree(src, link)
    if clean not in sys.path:
        sys.path.append(clean)   # append: sandbox numpy keeps priority


_bootstrap_arcengine()
_SHIM_USED = _pydantic_shim.install_if_missing()

from arcengine import GameAction, ActionInput, GameState  # noqa: E402

# py<3.11: GameAction members declared with tuple values aren't registered in
# _value2member_map_, breaking GameAction(<int>) / deepcopy (v13 compat).
for _m in GameAction:
    GameAction._value2member_map_.setdefault(_m.value, _m)

MOVES = [1, 2, 3, 4]          # ls20 available movement actions
RESET = 0
CLICK = 6                      # ACTION6 = ComplexAction(x, y)


def repo_root() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # .../arc3/CommunitySolutions/chronos_solver/v17 -> up 3 = arc3
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def game_py_path(game="ls20") -> str:
    root = repo_root()
    for base in (os.path.join(root, "environment_files"),
                 os.path.join(root, "arc-prize-2026-arc-agi-3", "environment_files")):
        if not os.path.isdir(base):
            continue
        gdir = os.path.join(base, game)
        if os.path.isdir(gdir):
            for ver in sorted(os.listdir(gdir)):
                p = os.path.join(gdir, ver, f"{game}.py")
                if os.path.exists(p):
                    return p
    raise FileNotFoundError(f"could not locate {game}.py under environment_files")


def _find_class_name(path):
    import re
    txt = open(path).read()
    m = re.search(r"class\s+(\w+)\s*\(\s*ARCBaseGame", txt)
    return m.group(1) if m else None


_GAME_CACHE = {}


def load_game(game="ls20"):
    """Instantiate a fresh engine object for `game`."""
    path = game_py_path(game)
    if path not in _GAME_CACHE:
        spec = importlib.util.spec_from_file_location("game_mod", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["game_mod"] = mod
        spec.loader.exec_module(mod)
        _GAME_CACHE[path] = (mod, _find_class_name(path))
    mod, cls_name = _GAME_CACHE[path]
    return getattr(mod, cls_name)()


def perform(g, act_id, data=None):
    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data \
        else ActionInput(id=GameAction.from_id(act_id))
    return g.perform_action(ai, raw=True)


def reset(g):
    perform(g, RESET)
    return perform(g, RESET)            # v13 double-reset


def frame_of(r):
    if not r or not r.frame:
        return None
    return np.array(r.frame[-1])


def cache_path(game="ls20"):
    """Prefer the v13 cache (L0-L4 solved); v17 may write its own alongside."""
    here = os.path.dirname(__file__)
    for cand in (os.path.join(here, f"v17_bfs_cache_{game}.json"),
                 os.path.join(here, "..", "v13", f"v13_bfs_cache_{game}.json")):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return os.path.abspath(os.path.join(here, "..", "v13", f"v13_bfs_cache_{game}.json"))


def load_cache(game="ls20"):
    p = cache_path(game)
    return (json.load(open(p)) if os.path.exists(p) else {}), p


def chain_to_level(g, level, cache):
    """v13 fix #4: build level `level`'s real start state by replaying the
    cached solutions for levels 0..level-1. Returns the last FrameDataRaw."""
    r = reset(g)
    applied = 0
    for lvl in range(level):
        sol = cache.get(str(lvl))
        if sol is None:
            raise KeyError(f"no cached solution for L{lvl} — cannot chain to L{level}")
        for act in sol:
            act_id = act[0]
            data = act[1] if len(act) > 1 and act[1] else None
            r = perform(g, act_id, data)
            applied += 1
    return r, applied


# ---------------------------------------------------------------- hashing ---
def scalar_state(g):
    """Public bool/int attrs of the engine — key shape/color/rotation,
    countdowns, player coords. Folded into the identity hash so the hidden
    countdown chain (v13 bug #3) and timer aliasing don't collapse states."""
    out = []
    for k, v in g.__dict__.items():
        if k.startswith("_"):
            continue
        if isinstance(v, bool):
            out.append((k, int(v)))
        elif isinstance(v, int):
            out.append((k, v))
    return tuple(sorted(out))


def state_hash(g, frame, mask=None):
    f = frame
    if mask is not None:
        f = frame.copy()
        f[mask] = 0
    h = hashlib.md5(f.tobytes() + repr(scalar_state(g)).encode()).hexdigest()[:16]
    return h


def detect_transient_mask(game, level, cache, n_probe=6):
    """v13 fix #2: rows that change under EVERY single action are HUD/timer —
    mask them from the dedup hash. Probe a few random actions from the level
    start and AND together the change masks."""
    g = load_game(game)
    r0, _ = chain_to_level(g, level, cache)
    f0 = frame_of(r0)
    if f0 is None:
        return None
    always = np.ones_like(f0, dtype=bool)
    for a in (MOVES * n_probe)[:n_probe]:
        gg = load_game(game)
        chain_to_level(gg, level, cache)
        r = perform(gg, a)
        f = frame_of(r)
        if f is None or f.shape != f0.shape:
            continue
        always &= (f != f0)
    # only keep whole rows that are "always changing" (timer bars are rows)
    row_always = always.all(axis=1)
    mask = np.zeros_like(f0, dtype=bool)
    mask[row_always, :] = True
    return mask if mask.any() else None


def detect_transient_scalars(game, level, cache, n_walk=48, nav_thresh=3,
                             n_seeds=2, seed0=0):
    """iter6: clean the cross-game progress signal by masking NAVIGATION scalars.

    The progress proxy = count of engine scalar attrs that differ from the
    level-start values. iter5 found this is polluted: on cd82 nearly every step
    "registers progress" because moving changes the player coordinates, and a
    changed coordinate trivially counts as progress (walking == fake progress).

    The fix mirrors detect_transient_mask (which masks HUD/timer ROWS of the
    frame). We run a *movement-only* random walk from the level start and record
    how many DISTINCT values each scalar attr takes. Player-position / camera /
    free-running coordinate attrs roam over many values just from wandering
    (>nav_thresh distinct values); genuine state-machine attrs (keys collected,
    locks opened, doors, goal flags) stay constant under pure movement and only
    flip on a specific interaction. So: high-variance-under-movement keys are
    NAVIGATION and are excluded from the progress count. Two random seeds are
    intersected so a key must be high-variance under *both* walks to be masked
    (avoids masking a real event that happened to fire once during a walk).

    Returns: set of attr-name strings to ignore in prog(). Empty if none.
    Verified behaviour (iter6): cd82 -> {2 coord keys}; su15 -> {} (its
    progress=2 is genuine, no nav keys); sc25 -> {2 coord keys}.
    """
    nav_per_seed = []
    for s in range(n_seeds):
        rng = random.Random(seed0 + s)
        g = load_game(game)
        chain_to_level(g, level, cache)
        vals = {k: {v} for k, v in scalar_state(g)}
        for _ in range(n_walk):
            try:
                perform(g, rng.choice(MOVES))
            except Exception:
                break
            for k, v in scalar_state(g):
                vals.setdefault(k, set()).add(v)
        nav_per_seed.append({k for k, sset in vals.items() if len(sset) > nav_thresh})
    if not nav_per_seed:
        return set()
    nav = set.intersection(*nav_per_seed) if len(nav_per_seed) > 1 else nav_per_seed[0]
    return nav


# ----------------------------------------------------- object features ------
def object_features(frame, bg=None):
    """Compact object-centric descriptor of a 64x64 frame — the shared input
    for ForgeNet (heuristic) and the TRM LM. Returns a fixed-length vector:
      per color c in 1..15: [count, cx, cy, bbox_w, bbox_h]  (normalised)
    plus global [n_nonbg_colors]. -> 15*5 + 1 = 76 dims.
    """
    h, w = frame.shape
    if bg is None:
        bg = np.bincount(frame.flatten(), minlength=16).argmax()
    feats = []
    n_colors = 0
    for c in range(1, 16):
        ys, xs = np.where(frame == c)
        if len(xs) == 0 or c == bg:
            feats += [0.0, 0.0, 0.0, 0.0, 0.0]
            continue
        n_colors += 1
        feats += [
            len(xs) / (h * w),
            xs.mean() / w,
            ys.mean() / h,
            (xs.max() - xs.min() + 1) / w,
            (ys.max() - ys.min() + 1) / h,
        ]
    feats.append(n_colors / 15.0)
    return np.array(feats, dtype=np.float32)


def dynamic_clicks(frame, limit=12, bg=None):
    """v13: click targets = centroids of each non-background color object."""
    if bg is None:
        bg = np.bincount(frame.flatten(), minlength=16).argmax()
    cnt = np.bincount(frame.flatten(), minlength=16)
    out = []
    for c in range(1, 16):
        if c == bg or cnt[c] == 0 or cnt[c] > frame.size // 2:
            continue
        ys, xs = np.where(frame == c)
        out.append((CLICK, {"x": int(xs.mean()), "y": int(ys.mean()), "game_id": "bfs"}))
    return out[:limit]


# ---------------------------------------------------------------- RHAE ------
def rhae_score(actions_per_level, baselines):
    """RHAE proxy. ARC-AGI-3 rewards solving each level in few actions; the
    per-level contribution is min(1, baseline/actions)^2 (>=0, =1 at/under
    baseline). Total over solved levels. Optimising RHAE == fewer actions.
    `baselines` maps level-> reference action count (v13's counts by default).
    """
    total = 0.0
    detail = {}
    for lvl, acts in actions_per_level.items():
        base = baselines.get(lvl, acts)
        s = min(1.0, base / max(1, acts)) ** 2
        detail[lvl] = round(s, 4)
        total += s
    return round(total, 4), detail


if __name__ == "__main__":
    # self-test: chain to L5 and report
    g = load_game("ls20")
    cache, cp = load_cache("ls20")
    r, applied = chain_to_level(g, 5, cache)
    print(f"shim_used={_SHIM_USED} cache={cp}")
    print(f"chained {applied} actions -> levels_completed={r.levels_completed}")
    f = frame_of(r)
    print("frame shape:", None if f is None else f.shape,
          "object_features dim:", object_features(f).shape if f is not None else None)
