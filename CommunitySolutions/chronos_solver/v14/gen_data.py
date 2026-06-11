#!/usr/bin/env python3
"""
v14 Phase 0 — the data factory.

Loads each LOCAL game engine directly (the v13 BFSSolver trick) and rolls
policies through it to harvest transitions for world-model training:

    shard fields (uint8/int16 npz):
      grids   (N+1, 64, 64)  frame sequence, camera-rendered
      actions (N, 3)         (id, x, y) — x=y=0 for simple actions
      rewards (N,)           0 neutral / 1 level-complete / 2 reset-detected
      game_id                for the held-out-game split in training

Policy mix per episode (important for coverage):
  - 70% epsilon-random over avail actions + object-centroid clicks
  - 30% replay of v13 cached solutions with random truncation+deviation
    (random play almost never reaches WIN transitions; expert replays do)

Usage (Mac or RTX box, repo root, venv312):
    python CommunitySolutions/chronos_solver/v14/gen_data.py \
        --out /tmp/v14_shards --episodes-per-game 200 --max-steps 120

UNTESTED SKELETON — validate one shard visually before a long run.
"""
import argparse
import glob
import importlib.util
import json
import logging
import os
import random
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'ARC-AGI-3-Agents'))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

from arcengine import GameAction, ActionInput  # noqa: E402
for _m in GameAction:  # py<3.11 enum compat (see v13 notes)
    GameAction._value2member_map_.setdefault(_m.value, _m)


def load_game(game_id):
    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    matches = glob.glob(os.path.join(env_dir, game_id, "**", f"{game_id}.py"),
                        recursive=True)
    if not matches:
        return None
    src = matches[0]
    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', open(src).read())
    spec = importlib.util.spec_from_file_location('game_mod', src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['game_mod'] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, m.group(1))


def act(g, aid, x=0, y=0):
    data = {'x': int(x), 'y': int(y), 'game_id': 'gen'} if aid == 6 else None
    ai = ActionInput(id=GameAction.from_id(aid), data=data) if data \
        else ActionInput(id=GameAction.from_id(aid))
    return g.perform_action(ai, raw=True)


def centroid_clicks(frame, k=10):
    cnt = np.bincount(frame.flatten(), minlength=16)
    bg = int(cnt.argmax())
    out = []
    for c in range(16):
        if c == bg or cnt[c] == 0 or cnt[c] > frame.size // 2:
            continue
        ys, xs = np.where(frame == c)
        out.append((int(cnt[c]), int(np.median(xs)), int(np.median(ys))))
    out.sort()
    return [(6, x, y) for _, x, y in out[:k]]


def expert_prefix(game_id):
    """Random prefix of a v13 cached solution (exercises WIN transitions)."""
    for vdir in ('v13', 'v12'):
        p = os.path.join(HERE, '..', vdir, f'{vdir}_bfs_cache_{game_id}.json')
        if os.path.exists(p):
            sols = json.load(open(p))
            if sols:
                sol = sols[random.choice(list(sols))]
                cut = random.randint(max(1, len(sol) // 2), len(sol))
                return [(a, (d or {}).get('x', 0), (d or {}).get('y', 0))
                        for a, d in sol[:cut]]
    return None


def roll_episode(game_cls, game_id, max_steps):
    g = game_cls()
    act(g, 0); r = act(g, 0)                       # double RESET baseline
    grids = [np.array(r.frame[-1], dtype=np.uint8)]
    actions, rewards = [], []
    lvl = r.levels_completed
    plan = expert_prefix(game_id) if random.random() < 0.3 else None
    avail = list(getattr(g, '_available_actions', [1, 2, 3, 4]))
    for t in range(max_steps):
        if plan and t < len(plan):
            aid, x, y = plan[t]
        else:
            cands = [(a, 0, 0) for a in avail if a <= 5 or a == 7]
            if 6 in avail:
                cands += centroid_clicks(grids[-1])
            aid, x, y = random.choice(cands)
        try:
            r = act(g, aid, x, y)
        except Exception:
            break
        if not r.frame:
            break
        f = np.array(r.frame[-1], dtype=np.uint8)
        # reward labeling: WIN beats everything; big frame jump back to the
        # level-start pattern ~ silent reset (label 2)
        if r.levels_completed > lvl or g._current_level_index > lvl:
            rew, lvl = 1, max(r.levels_completed, g._current_level_index)
        elif np.array_equal(f, grids[0]) and t > 3:
            rew = 2
        else:
            rew = 0
        grids.append(f)
        actions.append((aid, x, y))
        rewards.append(rew)
        if rew == 1 and lvl >= len(getattr(g, '_levels', [])):
            break
    return (np.stack(grids), np.array(actions, np.int16),
            np.array(rewards, np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/v14_shards")
    ap.add_argument("--games", default="all")
    ap.add_argument("--episodes-per-game", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=120)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env_dir = os.path.join(REPO, 'arc-prize-2026-arc-agi-3', 'environment_files')
    games = sorted(os.listdir(env_dir)) if args.games == "all" \
        else args.games.split(",")
    for gid in games:
        cls = load_game(gid)
        if cls is None:
            logger.warning(f"{gid}: no source"); continue
        eps = []
        for e in range(args.episodes_per_game):
            try:
                eps.append(roll_episode(cls, gid, args.max_steps))
            except Exception as ex:
                logger.warning(f"{gid} ep{e}: {ex}")
        if not eps:
            continue
        np.savez_compressed(
            os.path.join(args.out, f"{gid}.npz"),
            grids=np.concatenate([e[0] for e in eps]),
            lengths=np.array([len(e[1]) for e in eps]),
            actions=np.concatenate([e[1] for e in eps]),
            rewards=np.concatenate([e[2] for e in eps]))
        n = sum(len(e[1]) for e in eps)
        logger.info(f"{gid}: {len(eps)} episodes, {n} transitions saved")


if __name__ == "__main__":
    main()
