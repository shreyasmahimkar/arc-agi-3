"""v14 PLM — the Stochastic Goose: explore while the world model is wrong.

Tracks an EMA of next-frame prediction error. While error is high, picks
the action whose outcome the model is LEAST sure about (max token-entropy
= max expected information). Once error drops under threshold, the goose
retires and the latent-BFS planner takes over. UNTESTED SKELETON.
"""
import torch
import torch.nn.functional as F


class Goose:
    def __init__(self, cfg):
        self.cfg = cfg
        self.err = 1.0          # running token error rate (start pessimistic)
        self.steps = 0

    def observe(self, predicted_tokens, actual_tokens):
        """Call after each real step with (64,) predicted vs actual ids."""
        e = (predicted_tokens != actual_tokens).float().mean().item()
        self.err = self.cfg.surprise_ema * self.err + \
            (1 - self.cfg.surprise_ema) * e
        self.steps += 1

    def should_explore(self):
        return (self.steps < self.cfg.min_explore_steps
                or self.err > self.cfg.explore_threshold)

    @torch.no_grad()
    def pick(self, belief, sim, candidate_actions, device):
        """Choose the action with maximum predicted outcome-entropy."""
        aid = torch.tensor([a[0] for a in candidate_actions], device=device)
        ax = torch.tensor([a[1] for a in candidate_actions], device=device)
        ay = torch.tensor([a[2] for a in candidate_actions], device=device)
        h = belief.expand(len(candidate_actions), -1)
        tok_logits, _, change = sim(h, aid, ax, ay)
        probs = F.softmax(tok_logits, -1)
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean(-1)  # (A,)
        # prefer uncertain AND likely-to-do-something actions
        score = ent + 0.5 * torch.sigmoid(change)
        return candidate_actions[int(score.argmax())]
