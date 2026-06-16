"""v18 agents — frame-only, game-agnostic. THE rule: `act(obs)` may use only
the fields on `Obs` (the public FrameData surface). No engine internals, no
simulating the current game. The SAME agent object plays every game; any
game_id-specific branch is cheating and defeats the transfer test.

Baseline: ReactiveExplorer — a novelty-seeking bandit over the available
actions, with click targets read from the *visible* frame. It is deliberately
simple; its job is to set an honest floor that learned agents (next iterations)
must beat on HELD-OUT games.
"""
from __future__ import annotations
import random
import numpy as np
from blackbox_env import MOVES, RESET, CLICK


def visible_click_targets(frame: np.ndarray, limit: int = 8):
    """Click targets = centroids of each non-background colour blob in the
    CURRENTLY VISIBLE frame. Uses only the frame -> honest."""
    flat = frame.flatten()
    bg = np.bincount(flat, minlength=16).argmax()
    cnt = np.bincount(flat, minlength=16)
    out = []
    for c in range(1, 16):
        if c == bg or cnt[c] == 0 or cnt[c] > frame.size // 2:
            continue
        ys, xs = np.where(frame == c)
        out.append({"x": int(xs.mean()), "y": int(ys.mean()), "game_id": "v18"})
    return out[:limit]


class BaseAgent:
    name = "base"

    def reset(self, game_id: str):
        """Called once at the start of each game episode."""

    def act(self, obs):
        """Return (action_id, data|None) using ONLY fields of obs."""
        raise NotImplementedError


class ReactiveExplorer(BaseAgent):
    """Novelty bandit. For each action it tracks how often that action produced
    a never-before-seen frame; picks the most-novelty-productive action with
    epsilon exploration. Tries a visible click target periodically when ACTION6
    is available. Resets on GAME_OVER. No lookahead, no simulation."""
    name = "reactive"

    def __init__(self, eps: float = 0.25, click_period: int = 7, seed: int = 0):
        self.eps = eps
        self.click_period = click_period
        self.rng = random.Random(seed)

    def reset(self, game_id: str):
        self.seen = set()
        self.last_hash = None
        self.last_action = None
        # per-action novelty stats: action -> [n_novel, n_tried]
        self.stats = {}
        self.t = 0
        self.click_idx = 0

    def _credit(self, novel: bool):
        if self.last_action is None:
            return
        s = self.stats.setdefault(self.last_action, [0, 0])
        s[0] += int(novel)
        s[1] += 1

    def _novelty_rate(self, a):
        s = self.stats.get(a)
        if not s or s[1] == 0:
            return 1.0  # optimistic: untried actions look maximally novel
        return s[0] / s[1]

    def act(self, obs):
        self.t += 1
        h = obs.frame_hash()
        novel = h not in self.seen
        self._credit(novel)
        self.seen.add(h)
        self.last_hash = h

        if obs.state == "GAME_OVER":
            self.last_action = RESET
            return RESET, None

        avail = list(obs.available_actions) or MOVES

        # periodically probe a visible click target if clicking is allowed
        if CLICK in avail and self.t % self.click_period == 0:
            tgts = visible_click_targets(obs.frame)
            if tgts:
                d = tgts[self.click_idx % len(tgts)]
                self.click_idx += 1
                self.last_action = CLICK
                return CLICK, d

        move_acts = [a for a in avail if a != CLICK] or MOVES
        if self.rng.random() < self.eps:
            a = self.rng.choice(move_acts)
        else:
            a = max(move_acts, key=lambda x: (self._novelty_rate(x), self.rng.random()))
        self.last_action = a
        return a, None


def make_agent(name: str = "reactive", **kw) -> BaseAgent:
    # NOTE: the old "clone" (memory-book lookup) agent was deleted by design —
    # v18 never uses stored answers. The genuine solver lives in search_agent.py.
    from search_agent import SearchAgent
    reg = {"reactive": ReactiveExplorer, "search": SearchAgent}
    return reg.get(name, ReactiveExplorer)(**kw)
