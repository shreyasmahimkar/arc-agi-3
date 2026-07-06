# =====================================================================
# v21 intuition — the amortized "System-1" prior distilled from the corpus.
#
# Every cadence cycle we distill all verified solutions into an action-ORDERING
# prior: given a game/frame, which action to try first in blitz/search. This is
# what makes next cycle's exploration cheaper (the compounding the self-learning
# research attributes ~2.6x to). Starts corpus-frequency-simple and is pluggable
# to a trained policy net later — the interface (`order_actions`) stays fixed.
# =====================================================================
import os, json, glob, logging
from collections import Counter, defaultdict

logger = logging.getLogger("v21.intuition")
ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]


def distill(solutions_dir, out_path):
    """Read every solutions/<gid>.json and learn, per game + globally, an action
    weight that favors actions appearing in winning plans, weighted toward EARLY
    positions (early actions carry more signal about the core mechanic)."""
    per_game, global_c = {}, Counter()
    first_move = defaultdict(Counter)
    for p in glob.glob(os.path.join(solutions_dir, "*.json")):
        gid = os.path.splitext(os.path.basename(p))[0]
        try:
            sol = json.load(open(p))
        except Exception:
            continue
        c = Counter()
        for lvl, plan in sol.items():
            for i, step in enumerate(plan):
                a = step[0] if isinstance(step, (list, tuple)) else step
                w = 1.0 + 2.0 / (1 + i)          # early-move upweight
                c[int(a)] += w; global_c[int(a)] += w
                if i == 0:
                    first_move[gid][int(a)] += 1
        if c:
            per_game[gid] = _normalize(c)
    prior = {"global": _normalize(global_c),
             "per_game": per_game,
             "first_move": {g: dict(v) for g, v in first_move.items()},
             "actions": ALL_ACTIONS}
    json.dump(prior, open(out_path, "w"), indent=1)
    logger.info("intuition distilled -> %s (%d games)", out_path, len(per_game))
    return prior


def _normalize(counter):
    tot = sum(counter.values()) or 1.0
    return {str(k): round(v / tot, 4) for k, v in counter.items()}


class IntuitionPrior:
    """Loaded in the agent; biases blitz/search action order. frame is optional —
    when a trained net replaces this, it will consume the frame; today it uses the
    per-game/global corpus weights."""
    def __init__(self, path):
        self.p = json.load(open(path)) if path and os.path.exists(path) else \
                 {"global": {}, "per_game": {}, "first_move": {}, "actions": ALL_ACTIONS}

    def order_actions(self, game=None, frame=None):
        w = dict(self.p.get("global", {}))
        gw = self.p.get("per_game", {}).get(game, {}) if game else {}
        for k, v in gw.items():                       # per-game overrides global
            w[k] = w.get(k, 0) + v
        return sorted(self.p.get("actions", ALL_ACTIONS),
                      key=lambda a: -w.get(str(a), 0.0))

    def first_guess(self, game):
        fm = self.p.get("first_move", {}).get(game, {})
        return int(max(fm, key=fm.get)) if fm else None
