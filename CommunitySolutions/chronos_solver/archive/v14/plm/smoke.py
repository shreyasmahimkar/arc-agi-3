"""v14 PLM shape/smoke test — run me FIRST on any live machine:

    cd CommunitySolutions/chronos_solver/v14 && python -m plm.smoke

Verifies every module's tensor contract end-to-end with random data.
No weights needed. Should finish in <30s on CPU.
"""
import numpy as np
import torch

from .config import PLMConfig
from .encoder import Tokenizer, frame_to_tensor, object_channels
from .trm import BeliefCore
from .world_model import BlockCausalSimulator
from .planner import latent_bfs
from .curiosity import Goose
from .agent_plm import candidate_actions


def main():
    cfg = PLMConfig()
    dev = "cpu"
    frame = np.random.randint(0, 10, (64, 64))

    # encoder
    obj = object_channels(frame)
    assert obj.shape == (2, 64, 64), obj.shape
    x = frame_to_tensor(frame).unsqueeze(0)
    assert x.shape == (1, cfg.n_colors + 2, 64, 64), x.shape
    tok = Tokenizer(cfg)
    loss, recon, idx = tok(x, torch.from_numpy(frame).unsqueeze(0).long())
    assert idx.shape == (1, 8, 8), idx.shape
    print(f"tokenizer OK  (recon loss {recon.item():.3f})")

    # belief
    core = BeliefCore(cfg)
    h = core.initial(1, dev)
    a = torch.tensor([1]), torch.tensor([0]), torch.tensor([0])
    h = core.step(h, idx, *a)
    assert h.shape == (1, cfg.belief_dim), h.shape
    print("belief core OK")

    # world model
    sim = BlockCausalSimulator(cfg)
    tl, rl, ch = sim(h, *a)
    assert tl.shape == (1, cfg.tokens_per_frame, cfg.codebook), tl.shape
    assert rl.shape == (1, 3) and ch.shape == (1,)
    print("world model OK")

    # planner + goose (tiny budget so the smoke test stays fast)
    cfg.plan_depth, cfg.plan_beam = 2, 8
    cands = candidate_actions(frame, {1, 2, 3, 4, 6}, cfg)
    seq, stats = latent_bfs(h, sim, core, cands, cfg, dev)
    print(f"planner OK    (explored {stats['explored']}, win={'yes' if seq else 'no'})")
    g = Goose(cfg)
    act = g.pick(h, sim, cands, dev)
    g.observe(torch.zeros(64, dtype=torch.long), idx.flatten())
    print(f"goose OK      (picked {act}, err {g.err:.2f})")
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
