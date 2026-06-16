"""v14 PLM — latent BFS planner: the v13 search transplanted into imagination.

Batched breadth-first rollout inside the frozen world model. State dedup
by predicted-token bytes (the v13 visited-hash idea). Returns the shortest
imagined action sequence that reaches a predicted WIN. UNTESTED SKELETON.
"""
import torch

from .world_model import REWARD_WIN


@torch.no_grad()
def latent_bfs(belief, sim, belief_core, candidate_actions, cfg, device):
    """belief: (1, belief_dim). candidate_actions: list[(id,x,y)].
    Returns (action_sequence | None, stats)."""
    A = len(candidate_actions)
    aid = torch.tensor([a[0] for a in candidate_actions], device=device)
    ax = torch.tensor([a[1] for a in candidate_actions], device=device)
    ay = torch.tensor([a[2] for a in candidate_actions], device=device)

    frontier_h = belief.expand(A, -1)                       # (A, belief)
    histories = [[i] for i in range(A)]
    visited = set()
    explored = 0

    for depth in range(cfg.plan_depth):
        tok_logits, rew_logits, _ = sim(frontier_h, aid, ax, ay)
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
        tok_grid = toks[ki].view(len(keep), cfg.grid // cfg.patch,
                                 cfg.grid // cfg.patch)
        h_next = belief_core.step(frontier_h[ki], tok_grid,
                                  aid[ki % A], ax[ki % A], ay[ki % A])
        frontier_h = h_next.repeat_interleave(A, 0)
        histories = [histories[k] + [j] for k in keep for j in range(A)]
        aid = torch.tensor([a[0] for a in candidate_actions] * len(keep), device=device)
        ax = torch.tensor([a[1] for a in candidate_actions] * len(keep), device=device)
        ay = torch.tensor([a[2] for a in candidate_actions] * len(keep), device=device)

    return None, {"explored": explored, "depth": cfg.plan_depth}
