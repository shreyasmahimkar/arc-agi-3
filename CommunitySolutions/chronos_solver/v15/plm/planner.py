"""v15 PLM — P(win)-guided latent beam search.

Lessons from the first live run (ar25, 2026-06-11):
  - ar25 L0 needs 15 actions; a depth-5 horizon can't contain a win from
    the start position, so requiring a HARD predicted win (argmax==WIN)
    meant `bfs-miss` forever and the agent dissolved back to random
    poking every step.
  - An underconfident reward head (few WIN examples) may NEVER argmax to
    WIN even adjacent to one.

Fixes, in order of effect:
  1. SOFT GUIDANCE: track P(win)=softmax(reward)[WIN] for every imagined
     node; the beam keeps the most win-promising states (not insertion
     order), making this best-first search with a learned heuristic
     instead of blind BFS.
  2. PREFIX COMMIT: if no hard win is imagined but the best leaf has
     P(win) >= cfg.plan_win_prob, return the first cfg.plan_commit
     actions of that path — directed progress, then replan from closer.
  3. Depth raised via cfg.plan_depth (8): commit+replan extends the
     effective horizon far past a single search's depth.
"""
import torch

from .world_model import REWARD_WIN


@torch.no_grad()
def latent_bfs(belief, cur_tokens, sim, belief_core, candidate_actions,
               cfg, device):
    """belief: (1, belief_dim)  cur_tokens: (1, 64) or (1, 8, 8) ids.
    Returns (action_sequence | None, stats). stats['hard'] tells whether
    the sequence ends in a predicted win or is a committed best-prefix."""
    A = len(candidate_actions)
    aid = torch.tensor([a[0] for a in candidate_actions], device=device)
    ax = torch.tensor([a[1] for a in candidate_actions], device=device)
    ay = torch.tensor([a[2] for a in candidate_actions], device=device)

    frontier_h = belief.expand(A, -1)                       # (A, belief)
    frontier_t = cur_tokens.reshape(1, -1).expand(A, -1)    # (A, 64)
    histories = [[i] for i in range(A)]
    visited = set()
    explored = 0
    best_p, best_hist = 0.0, None       # most win-promising leaf seen

    for depth in range(cfg.plan_depth):
        tok_logits, rew_logits, change = sim(frontier_h, frontier_t,
                                             aid, ax, ay)
        explored += frontier_h.shape[0]
        toks = tok_logits.argmax(-1)                        # (N, 64)
        pwin = rew_logits.softmax(-1)[:, REWARD_WIN]        # (N,)

        wins = (rew_logits.argmax(-1) == REWARD_WIN).nonzero(as_tuple=True)[0]
        if len(wins):
            best = min(wins.tolist(), key=lambda i: len(histories[i]))
            seq = [candidate_actions[j] for j in histories[best]]
            return seq, {"explored": explored, "depth": depth + 1,
                         "hard": True, "p": float(pwin[best])}

        i_best = int(pwin.argmax())
        if float(pwin[i_best]) > best_p:
            best_p, best_hist = float(pwin[i_best]), histories[i_best]

        # beam: rank by win-promise (+ a nudge toward states that change),
        # dedup by predicted tokens, clip
        order = torch.argsort(pwin + 0.1 * torch.sigmoid(change),
                              descending=True).tolist()
        keep = []
        for i in order:
            hsh = toks[i].cpu().numpy().tobytes()
            if hsh in visited:
                continue
            visited.add(hsh)
            keep.append(i)
            if len(keep) >= cfg.plan_beam:
                break
        if not keep:
            break

        # advance belief for survivors, then branch over all actions again
        ki = torch.tensor(keep, device=device)
        side = cfg.grid // cfg.patch
        tok_grid = toks[ki].view(len(keep), side, side)
        h_next = belief_core.step(frontier_h[ki], tok_grid,
                                  aid[ki], ax[ki], ay[ki])
        frontier_h = h_next.repeat_interleave(A, 0)
        frontier_t = toks[ki].repeat_interleave(A, 0)
        histories = [histories[k] + [j] for k in keep for j in range(A)]
        aid = torch.tensor([a[0] for a in candidate_actions] * len(keep), device=device)
        ax = torch.tensor([a[1] for a in candidate_actions] * len(keep), device=device)
        ay = torch.tensor([a[2] for a in candidate_actions] * len(keep), device=device)

    # no hard win imagined — commit toward the most promising leaf if the
    # reward head shows any conviction at all
    if best_hist is not None and best_p >= cfg.plan_win_prob:
        seq = [candidate_actions[j] for j in best_hist][:cfg.plan_commit]
        return seq, {"explored": explored, "depth": cfg.plan_depth,
                     "hard": False, "p": best_p}
    return None, {"explored": explored, "depth": cfg.plan_depth,
                  "hard": False, "p": best_p}
