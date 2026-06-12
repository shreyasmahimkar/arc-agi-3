"""v15 PLM — latent BFS planner: the v13 search transplanted into imagination.

v15 change: the simulator is token-conditioned, so a frontier node is now
(belief, predicted_tokens) instead of belief alone — imagined rollouts
feed each step's predicted tokens into both the next belief update AND
the next simulator call.
"""
import torch

from .world_model import REWARD_WIN


@torch.no_grad()
def latent_bfs(belief, cur_tokens, sim, belief_core, candidate_actions,
               cfg, device):
    """belief: (1, belief_dim)  cur_tokens: (1, 64) or (1, 8, 8) ids.
    candidate_actions: list[(id,x,y)].
    Returns (action_sequence | None, stats)."""
    A = len(candidate_actions)
    aid = torch.tensor([a[0] for a in candidate_actions], device=device)
    ax = torch.tensor([a[1] for a in candidate_actions], device=device)
    ay = torch.tensor([a[2] for a in candidate_actions], device=device)

    frontier_h = belief.expand(A, -1)                       # (A, belief)
    frontier_t = cur_tokens.reshape(1, -1).expand(A, -1)    # (A, 64)
    histories = [[i] for i in range(A)]
    visited = set()
    explored = 0

    for depth in range(cfg.plan_depth):
        tok_logits, rew_logits, _ = sim(frontier_h, frontier_t, aid, ax, ay)
        explored += frontier_h.shape[0]
        toks = tok_logits.argmax(-1)                        # (N, 64)
        rew = rew_logits.argmax(-1)                         # (N,)

        wins = (rew == REWARD_WIN).nonzero(as_tuple=True)[0]
        if len(wins):
            best = min(wins.tolist(), key=lambda i: len(histories[i]))
            seq = [candidate_actions[j] for j in histories[best]]
            return seq, {"explored": explored, "depth": depth + 1}

        # dedup + beam clip
        keep = []
        for i in range(toks.shape[0]):
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
                                  aid[ki % A], ax[ki % A], ay[ki % A])
        frontier_h = h_next.repeat_interleave(A, 0)
        frontier_t = toks[ki].repeat_interleave(A, 0)       # imagined tokens ride along
        histories = [histories[k] + [j] for k in keep for j in range(A)]
        aid = torch.tensor([a[0] for a in candidate_actions] * len(keep), device=device)
        ax = torch.tensor([a[1] for a in candidate_actions] * len(keep), device=device)
        ay = torch.tensor([a[2] for a in candidate_actions] * len(keep), device=device)

    return None, {"explored": explored, "depth": cfg.plan_depth}
