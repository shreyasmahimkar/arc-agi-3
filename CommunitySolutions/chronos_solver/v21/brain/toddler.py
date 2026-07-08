# =====================================================================
# v21 brain/toddler.py — the TODDLER (Epic C3): the intuitive "System-1"
# action scorer that every teacher (BFS / blitz / Go-Explore / coder) helps
# raise, behind the FIXED `order_actions(game, frame)` interface.
#
# The metaphor (TEACHER_STUDENT.md): teachers leave lessons on the blackboard
# (action_effects — did an action change the frame / win). The toddler WATCHES
# those lessons and forms an intuition about which action to try first — but it
# is not born blank: it starts from the corpus-frequency `IntuitionPrior`
# (what worked on solved levels) and shifts toward what it OBSERVES online as
# effects accumulate. It is also FRAME-AWARE in its first, GPU-free form: it
# keeps a per-coarse-frame effect memory (self-supervised "from a frame like
# this, action a tends to change things"), so a StochasticGoose-lite CNN / TRM
# scorer (R9/R11) can later drop in behind `_effect_score` without touching the
# interface or any caller.
#
# INVARIANT: additive + env-gated (V21_TODDLER, default OFF); pure/offline
# (stdlib + optional numpy via blackboard.cell_key); it only ORDERS actions —
# it never invents actions, never verifies, never risks the corpus. When it has
# seen nothing it degrades exactly to the corpus prior (and then to canonical
# order), so wiring it on is strictly a re-ranking of the SAME candidate set.
# =====================================================================
from brain.blackboard import ALL_ACTIONS, cell_key

# effect scores from a win-weighted change-rate live in ~[0, 4]; squash to ~[0,1]
_EFF_SCALE = 4.0


class Toddler:
    """Intuitive action-orderer distilled ONLINE from blackboard action_effects,
    blended with the corpus-frequency `IntuitionPrior`. Drop-in for
    `IntuitionPrior.order_actions` (same `(game, frame)` signature) plus an
    optional `actions=` candidate restriction used by the search callers."""

    def __init__(self, blackboard=None, prior=None, alpha=0.7):
        self.bb = blackboard          # Blackboard (global action_effects) or None
        self.prior = prior            # IntuitionPrior (corpus frequencies) or None
        self.alpha = float(alpha)     # weight on learned effects vs corpus prior
        self._frame_eff = {}          # cell_key(frame) -> {action: {changed,tried,won}}

    # -- self-supervised online update (teacher-agnostic) ---------------------
    def observe(self, action, changed, won=False, frame=None, bins=8):
        """A teacher just tried `action` from `frame` and saw `changed`/`won`.
        Records it both globally (on the blackboard, if present) and, when a
        frame is given, in the per-coarse-frame effect memory."""
        a = int(action)
        if self.bb is not None:
            self.bb.teach_action_effect(a, changed, won=won, source="toddler")
        if frame is not None:
            d = self._frame_eff.setdefault(cell_key(frame, bins=bins), {}) \
                    .setdefault(a, {"changed": 0, "tried": 0, "won": 0})
            d["tried"] += 1
            d["changed"] += int(bool(changed))
            d["won"] += int(bool(won))
        return self

    # -- scoring --------------------------------------------------------------
    def _effect_score(self, a, frame=None, bins=8):
        """Observed effectiveness of `a`: frame-conditioned first (what happened
        from frames like this one), else the blackboard's global effect. Returns
        None when the toddler has never seen `a` in a relevant context."""
        a = int(a)
        if frame is not None:
            fe = self._frame_eff.get(cell_key(frame, bins=bins), {}).get(a)
            if fe and fe["tried"]:
                return 3.0 * fe["won"] / fe["tried"] + fe["changed"] / fe["tried"]
        if self.bb is not None:
            g = self.bb.data.get("action_effects", {}).get(str(a))
            if g and g["tried"]:
                return 3.0 * g["won"] / g["tried"] + g["changed"] / g["tried"]
        return None

    def _prior_weights(self, game=None, actions=None):
        """Corpus-prior weight per candidate action (per-game over global)."""
        acts = list(actions) if actions else list(ALL_ACTIONS)
        if self.prior is None:
            return {int(a): 0.0 for a in acts}
        p = getattr(self.prior, "p", {}) or {}
        w = dict(p.get("global", {}))
        gw = p.get("per_game", {}).get(game, {}) if game else {}
        for k, v in gw.items():
            w[k] = w.get(k, 0.0) + v
        return {int(a): float(w.get(str(int(a)), 0.0)) for a in acts}

    def order_actions(self, game=None, frame=None, actions=None, bins=8):
        """Rank the candidate actions best-first. Blend, per action:
            seen:   alpha*effect(0..1) + (1-alpha)*prior(0..1)
            unseen: prior(0..1)            (lean on the corpus until we learn)
        Ties break on canonical action id, so the ordering is deterministic and,
        with no lessons + no prior, is exactly the canonical order (a no-op)."""
        acts = list(actions) if actions else list(ALL_ACTIONS)
        pri = self._prior_weights(game, acts)
        pmax = max(pri.values()) or 1.0

        def score(a):
            p = pri[int(a)] / pmax                      # 0..1
            eff = self._effect_score(a, frame, bins)
            if eff is None:                             # never seen -> corpus prior
                return p
            e = max(0.0, eff) / _EFF_SCALE              # ~0..1
            return self.alpha * e + (1.0 - self.alpha) * p

        return sorted(acts, key=lambda a: (-score(a), int(a)))

    # convenience: same as IntuitionPrior.first_guess but effect-aware
    def first_guess(self, game=None, frame=None, actions=None):
        order = self.order_actions(game=game, frame=frame, actions=actions)
        return order[0] if order else None
