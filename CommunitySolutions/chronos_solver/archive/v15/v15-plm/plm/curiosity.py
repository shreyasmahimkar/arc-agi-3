"""v15 PLM — the Stochastic Goose: explore while the world model is wrong.

Tracks an EMA of next-frame prediction error. While error is high, picks
the action whose outcome the model is LEAST sure about (max expected
information). Once error drops under threshold, the goose retires and
the latent-BFS planner takes over.

Two failure modes observed live (ar25, 2026-06-12) and fixed here:

1. ACTION FIXATION: deterministic argmax-surprise pressed ACTION7 (undo)
   hundreds of times in a row — undo's outcome depends on history the
   belief can't fully carry, so it is PERMANENTLY the most surprising
   action: a curiosity trap. Fix: a repetition penalty that grows with
   every consecutive repeat until any other action wins.

2. NO PLATEAU EXIT: retirement required err < threshold (0.12), but the
   model's irreducible error floor on this game was ~0.17 — the goose
   could never retire and the planner never ran. Fix: if err has stopped
   improving for `plateau_window` observations, hand over anyway —
   further poking teaches nothing; let the planner work with the model
   we have.
"""
import torch
import torch.nn.functional as F


class Goose:
    PLATEAU_WINDOW = 15      # observations with <PLATEAU_EPS improvement
    PLATEAU_EPS = 0.01       # -> hand over to the planner
    REPEAT_PENALTY = 0.5     # per consecutive repeat of the same action

    def __init__(self, cfg):
        self.cfg = cfg
        self.err = 1.0          # running token error rate (start pessimistic)
        self.steps = 0
        self.hist = []          # err trace for plateau detection
        self.last_pick = None   # (id,x,y) of previous pick
        self.repeats = 0        # consecutive identical picks

    def observe(self, predicted_tokens, actual_tokens):
        """Call after each real step with (64,) predicted vs actual ids."""
        e = (predicted_tokens != actual_tokens).float().mean().item()
        self.err = self.cfg.surprise_ema * self.err + \
            (1 - self.cfg.surprise_ema) * e
        self.steps += 1
        self.hist.append(self.err)
        if len(self.hist) > 50:
            self.hist = self.hist[-50:]

    def should_explore(self):
        if self.steps < self.cfg.min_explore_steps:
            return True
        if self.err <= self.cfg.explore_threshold:
            return False                      # model is trusted — plan
        # plateau: surprise stopped falling; more poking won't teach the
        # frozen model anything. The planner should try with what we have.
        if (len(self.hist) >= self.PLATEAU_WINDOW
                and self.hist[-self.PLATEAU_WINDOW] - self.err
                < self.PLATEAU_EPS):
            return False
        return True

    @torch.no_grad()
    def pick(self, belief, cur_tokens, sim, candidate_actions, device):
        """Choose the action with maximum predicted outcome-entropy,
        penalizing consecutive repeats (the ACTION7 curiosity-trap fix)."""
        aid = torch.tensor([a[0] for a in candidate_actions], device=device)
        ax = torch.tensor([a[1] for a in candidate_actions], device=device)
        ay = torch.tensor([a[2] for a in candidate_actions], device=device)
        h = belief.expand(len(candidate_actions), -1)
        t = cur_tokens.reshape(1, -1).expand(len(candidate_actions), -1)
        tok_logits, _, change, _ = sim(h, t, aid, ax, ay)
        probs = F.softmax(tok_logits, -1)
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean(-1)  # (A,)
        # prefer uncertain AND likely-to-do-something actions
        score = ent + 0.5 * torch.sigmoid(change)
        if self.last_pick is not None and self.repeats > 0:
            for i, a in enumerate(candidate_actions):
                if a == self.last_pick:
                    score[i] -= self.REPEAT_PENALTY * self.repeats
        choice = candidate_actions[int(score.argmax())]
        self.repeats = self.repeats + 1 if choice == self.last_pick else 0
        self.last_pick = choice
        return choice
