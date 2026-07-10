# =====================================================================
# v21 evolve — the between-rounds code-writer / champion-challenger loop.
#
# This is the part that improves the AGENT, not just its answer cache. Each
# cadence cycle:
#   1. read recent failures (wall levels) from the scorecard
#   2. the LLM code-writer PROPOSES challengers: config patches AND/OR new
#      heuristic code snippets aimed at the walls
#   3. each challenger is EVALUATED in-sandbox on the 3 games + a HELD-OUT probe
#   4. promote a challenger ONLY if it strictly beats the champion on the
#      held-out set without regressing the train games (generalization-gated)
#   5. write the new champion + append an evolution-history row
#
# Honesty: nothing the LLM writes ships unverified. Challenger heuristics run in
# the same restricted sandbox as runtime_coder; eval RHAE comes from replay-verified
# solves only. Champion is a JSON config the live agent reads at load.
# =====================================================================
import os, json, time, logging, copy

logger = logging.getLogger("v21.evolve")

DEFAULT_CHAMPION = {
    "version": 0,
    "blitz_K": 200,                       # max repeat-action length in blitz probe
    "action_order": [6, 1, 2, 3, 4, 5, 7],
    "abandon_deaths": 2,
    "bfs_prefer_optimal": True,
    "heuristics": [],                     # list of vetted code snippets (name+code)
    "rhae": {},                           # last measured per-game RHAE
    "notes": "seed champion",
}

PROPOSE_SYSTEM = ("You improve an ARC-AGI-3 solver. Output ONLY strict JSON: a list "
                  "of <=N challenger patches. Each patch may set blitz_K (int), "
                  "action_order (list of 1..7), abandon_deaths (int), or add a "
                  "heuristic {name, code}. Target the listed WALL levels. No prose.")

PROPOSE_PROMPT = ("Champion config:\n{champ}\n\nWall levels still unsolved (game:level "
                  "with human baseline):\n{walls}\n\nRecent per-game RHAE:\n{rhae}\n\n"
                  "Propose up to {n} challenger patches most likely to crack the walls "
                  "or shorten loose solves. JSON list only.")


def load_champion(path):
    if os.path.exists(path):
        try:
            return json.load(open(path))
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_CHAMPION)


def save_champion(path, champ):
    json.dump(champ, open(path, "w"), indent=1)


def propose_challengers(champion, walls, rhae, llm, n=4):
    prompt = PROPOSE_PROMPT.format(champ=json.dumps(champion)[:1500],
                                   walls=json.dumps(walls)[:1200],
                                   rhae=json.dumps(rhae), n=n)
    raw = llm.complete(prompt, system=PROPOSE_SYSTEM, max_tokens=800, stop=["```"])
    patches = _parse_json_list(raw)
    challengers = []
    for patch in patches[:n]:
        if not isinstance(patch, dict):
            continue
        c = copy.deepcopy(champion)
        for k in ("blitz_K", "action_order", "abandon_deaths", "bfs_prefer_optimal"):
            if k in patch:
                c[k] = patch[k]
        if "heuristic" in patch and isinstance(patch["heuristic"], dict):
            c.setdefault("heuristics", []).append(patch["heuristic"])
        elif "name" in patch and "code" in patch:
            c.setdefault("heuristics", []).append({"name": patch["name"], "code": patch["code"]})
        c["notes"] = patch.get("note", "challenger")
        challengers.append(c)
    return challengers


def evolve_step(champion_path, history_path, walls, cur_rhae, llm, eval_fn,
                train_games, heldout_games, n=4, cost_fn=None):
    """eval_fn(config, games) -> {game: mean_rhae}. Promote a challenger only if it
    beats champion on HELD-OUT without regressing TRAIN. Returns (champion, promoted).

    R7(b) action-frugality (DREAMTEAM arXiv:2605.09650 — 31% fewer env-actions per
    game): when an optional `cost_fn(config, games) -> total_env_actions` is supplied,
    a challenger that TIES the current best on held-out RHAE but solves those walls with
    STRICTLY fewer env-actions is promoted too (lower cost = more frugal). This extends
    R4's quadratic-RHAE pressure to challengers that already hit RHAE 1.0 but do it in
    fewer steps. Purely additive: with `cost_fn=None` the tie-break is inert and behaviour
    is identical to before (a challenger must strictly beat held-out RHAE to promote)."""
    champ = load_champion(champion_path)
    champ_train = eval_fn(champ, train_games)
    champ_held = eval_fn(champ, heldout_games)
    base_train = _mean(champ_train); base_held = _mean(champ_held)
    best_cost = _cost(cost_fn, champ, heldout_games)
    logger.info("[evolve] champion v%s train=%.3f held=%.3f cost=%s",
                champ.get("version"), base_train, base_held,
                "n/a" if best_cost is None else round(best_cost, 1))

    best, best_held, best_train, promoted = champ, base_held, base_train, False
    for i, cand in enumerate(propose_challengers(champ, walls, cur_rhae, llm, n)):
        ct = _mean(eval_fn(cand, train_games))
        ch = _mean(eval_fn(cand, heldout_games))
        cc = _cost(cost_fn, cand, heldout_games)
        no_regress = ct >= base_train - 1e-6                        # generalization-gated
        beats = ch > best_held + 1e-6                               # strictly better RHAE
        frugal = (best_cost is not None and cc is not None and
                  abs(ch - best_held) <= 1e-6 and cc < best_cost - 1e-6)  # tie -> fewer actions
        ok = no_regress and (beats or frugal)
        why = "PROMOTE(rhae)" if (ok and beats) else ("PROMOTE(frugal)" if ok else "reject")
        logger.info("[evolve] challenger %d train=%.3f held=%.3f cost=%s -> %s",
                    i, ct, ch, "n/a" if cc is None else round(cc, 1), why)
        if ok:
            cand["version"] = champ.get("version", 0) + 1
            cand["rhae"] = cur_rhae
            best, best_held, best_train, promoted = cand, ch, ct, True
            if cc is not None:
                best_cost = cc

    if promoted:
        save_champion(champion_path, best)
    with open(history_path, "a") as f:
        f.write(json.dumps({"t": int(time.time()), "champion_v": best.get("version"),
                            "train": round(best_train, 4), "held": round(best_held, 4),
                            "held_actions": (None if best_cost is None else round(best_cost, 1)),
                            "promoted": promoted, "notes": best.get("notes")}) + "\n")
    return best, promoted


def _cost(cost_fn, config, games):
    """Total env-actions of `config` over `games` (lower = more frugal), or None when
    no cost_fn is supplied. Any failure degrades to None so frugality stays inert."""
    if cost_fn is None:
        return None
    try:
        c = cost_fn(config, games)
        return float(c) if c is not None else None
    except Exception:
        return None


def _rhae_level(human, ai_actions):
    """RHAE for one level: (min(1, human/ai))**2. Mirrors cadence_runner.rhae_level
    so config-aware wall probes score on the same scale as the shipped corpus."""
    if not ai_actions or ai_actions <= 0 or not human or human <= 0:
        return 0.0
    return min(1.0, (human / ai_actions)) ** 2


def config_aware_eval_fn(corpus_rhae, walls_by_game, probe_fn=None):
    """P2 upgrade (BACKLOG #1): the config-SENSITIVE evaluator that lets challengers
    actually PROMOTE. Returns eval_fn(config, games) -> {game: score in [floor, 1]}.

    For each game the score starts at the verified-corpus floor (never regresses) and
    RISES as the challenger's config cracks that game's still-unsolved WALL levels:

        score(g) = floor(g) + (1 - floor(g)) * mean(wall RHAE under this config)

    A wall's RHAE comes from `probe_fn(config, game, level) -> actions|None` — a real
    engine rollout that APPLIES the challenger's blitz_K/action_order (wired on the
    Mac; see cadence_runner._make_evolve_probe). Solving a budget-gated wall a bigger
    blitz_K unlocks strictly raises the score, so the generalization gate promotes it.

    When `probe_fn` is None (offline / no engine) it degrades to the config-insensitive
    corpus floor — identical to the old _corpus_eval_fn, so nothing promotes on noise.
    """
    walls_by_game = walls_by_game or {}

    def _f(config, games):
        out = {}
        for g in games:
            floor = float(corpus_rhae.get(g, 0.0))
            walls = walls_by_game.get(g, [])
            if not probe_fn or not walls:
                out[g] = floor
                continue
            gained = []
            for w in walls:
                lvl, base = w.get("level"), w.get("baseline")
                try:
                    actions = probe_fn(config, g, lvl)
                except Exception:
                    actions = None
                gained.append(_rhae_level(base, actions) if actions else 0.0)
            wall_score = sum(gained) / len(gained) if gained else 0.0
            out[g] = floor + (1.0 - floor) * wall_score
        return out

    return _f


def config_aware_cost_fn(walls_by_game, probe_fn=None, miss_penalty=100000.0):
    """R7(b) companion to config_aware_eval_fn: the env-action COST of a config over the
    walls (lower = more frugal). `probe_fn(config, game, level) -> actions|None` is the
    SAME real-engine rollout the eval uses; cost = sum of solved-wall action counts, with
    each still-UNSOLVED wall charged a fixed `miss_penalty` so a config that solves FEWER
    walls can never look cheaper than one that solves more. Returns cost_fn(config, games)
    -> float, wired into evolve_step(cost_fn=...) as the equal-RHAE tie-break.

    When `probe_fn` is None (offline / no engine) it returns 0.0 for every config, so the
    frugality tie-break is inert and nothing promotes on noise — mirroring the eval's
    degrade-to-floor contract."""
    walls_by_game = walls_by_game or {}

    def _c(config, games):
        if not probe_fn:
            return 0.0
        total = 0.0
        for g in games:
            for w in walls_by_game.get(g, []):
                lvl = w.get("level")
                try:
                    actions = probe_fn(config, g, lvl)
                except Exception:
                    actions = None
                total += float(actions) if actions else miss_penalty
        return total

    return _c


def _mean(d):
    vs = list(d.values()) if isinstance(d, dict) else list(d)
    return sum(vs) / len(vs) if vs else 0.0


def _parse_json_list(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").lstrip("json").strip()
    try:
        v = json.loads(raw)
        return v if isinstance(v, list) else [v]
    except Exception:
        # salvage the first [...] block
        i, j = raw.find("["), raw.rfind("]")
        if 0 <= i < j:
            try:
                return json.loads(raw[i:j + 1])
            except Exception:
                pass
    return []
