"""Lever #1 — offline-pretrain ChangeNet on the PUBLIC TRAIN games.

The winning preview agent (StochasticGoose) ran on a CNN that predicts which
action changes the frame. Its edge was that the prior generalised. We reproduce
that: drive the FORGE agent's own exploration on the TRAIN games, harvest its
ground-truth-labelled (frame, action -> changed/novel) transitions, and train a
fresh ChangeNet on the union. Ship the weights as a warm start for unseen games.

HELD-OUT (cn04, ka59, sk48, tu93, wa30) is NEVER touched — that is the transfer
test. lf52/tn36 skipped (load errors). Labels come from the real engine (did the
frame change), not from the model, so the supervision is honest.

Output: v19/pretrained_weights.pt
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "v18")))
from blackbox_env import BlackBoxEnv                       # noqa: E402
from forge_agent import ForgeAgent, ChangeNet, featurize, N_SIMPLE, CLICK_BASE, GRID  # noqa: E402

HERE = os.path.dirname(__file__)
HELDOUT = {"cn04", "ka59", "sk48", "tu93", "wa30"}
KNOWN_BAD = {"lf52", "tn36"}
ALL = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
       "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
       "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]
TRAIN = [g for g in ALL if g not in HELDOUT and g not in KNOWN_BAD]


def harvest(game, n_actions, device):
    """Run FORGE exploration; return its (frame, akey, target) buffer."""
    env = BlackBoxEnv(game); obs = env.reset()
    ag = ForgeAgent(seed=0, device=device); ag.reset(game)
    for _ in range(n_actions):
        aid, data = ag.act(obs)
        obs = env.step(aid, data)
        if obs.state == "WIN":
            break
    return list(ag.buf)


def train(samples, device, epochs, bsz=128, lr=3e-4):
    net = ChangeNet().to(device)
    opt = optim.Adam(net.parameters(), lr=lr)
    frames = np.stack([s[0] for s in samples]).astype(np.int64)
    keys = np.array([s[1] for s in samples], dtype=np.int64)
    targs = np.array([s[2] for s in samples], dtype=np.float32)
    n = len(samples)
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, bsz):
            idx = perm[i:i + bsz]
            x = featurize(torch.from_numpy(frames[idx]).to(device))
            a_logits, c_logits = net(x)
            k = keys[idx]
            sel = torch.empty(len(idx), device=device)
            for j, kk in enumerate(k):
                if kk < N_SIMPLE:
                    sel[j] = a_logits[j, kk]
                else:
                    ci = kk - CLICK_BASE; sel[j] = c_logits[j, ci // GRID, ci % GRID]
            t = torch.from_numpy(targs[idx]).to(device)
            loss = F.binary_cross_entropy_with_logits(sel, t)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"  epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}")
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_game", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[pretrain] device={device} TRAIN={len(TRAIN)} games per_game={args.per_game}")
    samples, t0 = [], time.time()
    for g in TRAIN:
        try:
            s = harvest(g, args.per_game, device)
            samples += s
            print(f"  {g}: +{len(s)} samples (total {len(samples)})")
        except Exception as e:
            print(f"  {g}: ERROR {repr(e)[:70]}")
    print(f"[pretrain] harvested {len(samples)} samples in {time.time()-t0:.0f}s; training…")
    pos = sum(1 for s in samples if s[2] > 0.5)
    print(f"[pretrain] label balance: {pos}/{len(samples)} changed ({pos/max(len(samples),1):.0%})")
    net = train(samples, device, args.epochs)
    out = os.path.join(HERE, "pretrained_weights.pt")
    torch.save(net.state_dict(), out)
    print(f"[pretrain] saved -> {out}")


if __name__ == "__main__":
    main()
