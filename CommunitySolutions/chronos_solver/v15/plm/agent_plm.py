"""v15 PLM — runtime agent loop.

This is the deployable inference path: no engine source, no solution
caches, no symbolic simulation of the game class. The agent learns the
game's dynamics from its OWN interactions, exactly as the hidden eval
requires:

    frame -> tokenize -> update belief -> (goose explores | planner plans)
          -> action -> observe surprise -> repeat

v15 change: the simulator is token-conditioned, so every sim call (the
prediction for goose grading, the goose's entropy probe, the planner's
rollouts) now carries the current frame's token ids.
"""
import logging
import os

import numpy as np
import torch

from .config import PLMConfig
from .encoder import Tokenizer, frame_to_tensor
from .trm import BeliefCore
from .world_model import BlockCausalSimulator
from .planner import latent_bfs
from .curiosity import Goose

logger = logging.getLogger(__name__)


def candidate_actions(frame, avail_ids, cfg):
    """Build the small discrete action set the PLM reasons over.

    Simple actions come straight from the env's available_actions.
    Click candidates are object centroids (v13's _dyn_clicks idea):
    never enumerate 4096 raw pixels.
    Returns list[(action_id, x, y)] — x=y=0 for non-click actions.
    """
    acts = [(a, 0, 0) for a in sorted(avail_ids) if 1 <= a <= 5 or a == 7]
    if 6 in avail_ids:
        cnt = np.bincount(frame.flatten(), minlength=cfg.n_colors)
        bg = int(cnt.argmax())
        targets = []
        for c in range(cfg.n_colors):
            if c == bg or cnt[c] == 0 or cnt[c] > frame.size // 2:
                continue
            ys, xs = np.where(frame == c)
            targets.append((int(cnt[c]), int(np.median(xs)), int(np.median(ys))))
        targets.sort()  # smallest objects first — usually the interactive ones
        for _, x, y in targets[: cfg.plan_topk_clicks]:
            acts.append((6, x, y))
    return acts or [(5, 0, 0)]  # always return something actionable


class PLMAgent:
    """Owns the four networks + goose state for one game episode stream."""

    def __init__(self, device=None, weights_path=None, cfg=None):
        self.cfg = cfg or PLMConfig()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available()
            else ("mps" if getattr(torch.backends, "mps", None)
                  and torch.backends.mps.is_available() else "cpu"))
        self.tok = Tokenizer(self.cfg).to(self.device).eval()
        self.belief_core = BeliefCore(self.cfg).to(self.device).eval()
        self.sim = BlockCausalSimulator(self.cfg).to(self.device).eval()
        self.loaded = self._load_weights(weights_path)
        self.reset_episode()

    # ---------------- weights ----------------
    def _load_weights(self, explicit_path):
        """Search order: explicit arg > env var > next to this package >
        the Kaggle dataset mount. Missing weights are survivable (the
        caller should then prefer its non-PLM fallback)."""
        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        candidates = [
            explicit_path,
            os.environ.get("V15_PLM_WEIGHTS"),
            os.path.join(here, "plm_weights.pt"),
            "/kaggle/input/v15-plm/plm_weights.pt",
        ]
        for p in candidates:
            if p and os.path.exists(p):
                try:
                    state = torch.load(p, map_location=self.device,
                                       weights_only=True)
                    self.tok.load_state_dict(state["tokenizer"])
                    self.belief_core.load_state_dict(state["belief"])
                    self.sim.load_state_dict(state["world_model"])
                    logger.info(f"PLM: loaded weights from {p}")
                    return True
                except Exception as e:
                    logger.warning(f"PLM: failed loading {p}: {e}")
        logger.warning("PLM: no weights found — model is untrained "
                       "(goose-only behavior; caller should prefer fallback)")
        return False

    # ---------------- episode state ----------------
    def reset_episode(self):
        """New game: empty belief, hungry goose."""
        self.h = self.belief_core.initial(1, self.device)
        self.goose = Goose(self.cfg)
        self.last_action = (0, 0, 0)       # RESET
        self.pred_tokens = None            # prediction made for the NEXT frame
        self.plan_queue = []               # committed action sequence

    def on_level_change(self):
        """Same game, next level: rules persist, layout doesn't.
        Keep the belief (it encodes the rules) but let the goose re-verify."""
        self.goose.err = max(self.goose.err, self.cfg.explore_threshold * 1.5)
        self.plan_queue = []

    # ---------------- main step ----------------
    @torch.no_grad()
    def step(self, frame, avail_ids):
        """frame: (64,64) int numpy. Returns ((id,x,y), info_str)."""
        x = frame_to_tensor(frame, self.cfg.n_colors).unsqueeze(0).to(self.device)
        _, tok_ids = self.tok.encode(x)                       # (1, 8, 8)
        cur = tok_ids.reshape(1, -1)                          # (1, 64)

        # 1) score last prediction vs reality -> goose learning signal
        if self.pred_tokens is not None:
            self.goose.observe(self.pred_tokens.flatten(),
                               cur.flatten())

        # 2) fold the new observation into the recursive belief
        aid = torch.tensor([self.last_action[0]], device=self.device)
        ax = torch.tensor([self.last_action[1]], device=self.device)
        ay = torch.tensor([self.last_action[2]], device=self.device)
        self.h = self.belief_core.step(self.h, tok_ids, aid, ax, ay)

        cands = candidate_actions(frame, avail_ids, self.cfg)

        # 3) committed plan in flight? keep executing it
        if self.plan_queue:
            action = self.plan_queue.pop(0)
            info = f"plm:plan({len(self.plan_queue)} left)"
        # 4) world model still confused -> goose pokes the environment
        elif self.goose.should_explore():
            action = self.goose.pick(self.h, cur, self.sim, cands, self.device)
            info = f"plm:goose(err={self.goose.err:.2f})"
        # 5) model trusts itself -> plan in imagination
        else:
            seq, stats = latent_bfs(self.h, cur, self.sim, self.belief_core,
                                    cands, self.cfg, self.device)
            if seq:
                action = seq[0]
                self.plan_queue = seq[1:]
                if stats.get("hard"):
                    info = f"plm:bfs(win@{stats['depth']},{stats['explored']}sims)"
                else:   # committed prefix toward the most win-promising leaf
                    info = f"plm:bfs-soft(p={stats['p']:.2f},commit={len(seq)})"
            else:  # reward head sees nothing promising — curious poking
                action = self.goose.pick(self.h, cur, self.sim, cands,
                                         self.device)
                info = f"plm:bfs-miss(p={stats['p']:.2f})->goose"

        # 6) record the prediction for this action so the NEXT frame can
        #    grade it (the goose's only supervision at test time)
        aid = torch.tensor([action[0]], device=self.device)
        ax = torch.tensor([action[1]], device=self.device)
        ay = torch.tensor([action[2]], device=self.device)
        tok_logits, _, _ = self.sim(self.h, cur, aid, ax, ay)
        self.pred_tokens = tok_logits.argmax(-1)              # (1, 64)

        self.last_action = action
        return action, info
