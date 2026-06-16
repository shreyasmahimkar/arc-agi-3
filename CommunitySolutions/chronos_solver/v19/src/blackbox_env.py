"""v18 black-box environment wrapper — the HONEST interface.

The whole point of v18: an agent may only see what the official ARC-AGI-3
`Agent.choose_action(frames, latest_frame)` sees — a rendered FrameData. NO
engine `__dict__` introspection, NO snapshot/restore, NO forward simulation of
the game it is playing. That is the only thing that transfers to the hidden
leaderboard games.

This wrapper loads the REAL game engine (so we can drive it and score locally,
no API key needed) but hands the agent ONLY an `Obs` carrying the public
FrameData fields:
    frame, levels_completed, win_levels, available_actions, state, game_id.

The engine handle never leaves this module. The eval harness has driving
access; the agent does not. That wall is what makes a "solve" honest.
"""
from __future__ import annotations
import os, sys, hashlib
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
# We reuse v17's proven engine LOADER ONLY (load/drive/render). The agent never
# touches it — it only receives Obs built from public FrameData fields below.
_V17 = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(_V17))
import engine as E  # noqa: E402

MOVES = E.MOVES        # [1,2,3,4]
RESET = E.RESET        # 0
CLICK = E.CLICK        # 6


@dataclass
class Obs:
    """Exactly the fields the official FrameData exposes to an agent."""
    frame: np.ndarray                 # (64,64) int grid (last frame)
    levels_completed: int
    win_levels: int
    available_actions: tuple
    state: str                        # 'NOT_FINISHED' | 'WIN' | 'GAME_OVER' | 'NOT_PLAYED'
    game_id: str
    action_counter: int = 0

    def frame_hash(self) -> str:
        return hashlib.md5(self.frame.tobytes()).hexdigest()[:16]

    @property
    def done(self) -> bool:
        return self.state in ("WIN", "GAME_OVER")


def _obs_from_raw(raw, game_id, action_counter) -> Obs:
    f = E.frame_of(raw)
    if f is None:
        f = np.zeros((64, 64), dtype=int)
    aa = tuple(getattr(raw, "available_actions", None) or [])
    st = getattr(raw, "state", None)
    st = getattr(st, "value", st) if st is not None else "NOT_FINISHED"
    return Obs(frame=f, levels_completed=int(raw.levels_completed or 0),
               win_levels=int(getattr(raw, "win_levels", 0) or 0),
               available_actions=aa, state=str(st), game_id=game_id,
               action_counter=action_counter)


class BlackBoxEnv:
    """Drive a real game; expose only Obs. One env instance == one game episode
    line (with RESET available like the real engine)."""

    def __init__(self, game: str):
        self.game = game
        self._g = None
        self.action_counter = 0

    def reset(self) -> Obs:
        self._g = E.load_game(self.game)
        raw = E.reset(self._g)
        self.action_counter = 0
        return _obs_from_raw(raw, self.game, self.action_counter)

    def step(self, action_id: int, data: Optional[dict] = None) -> Obs:
        if self._g is None:
            raise RuntimeError("call reset() before step()")
        raw = E.perform(self._g, action_id, data)
        self.action_counter += 1
        return _obs_from_raw(raw, self.game, self.action_counter)
