#!/usr/bin/env python3
"""v19 world-model trainer (ExIt step 3) — the tractable PLM.

Trains a forward world model on the harvested transitions: given (frame, action)
predict (next_frame, reward). This is the v15 idea made Mac-tractable — knowledge
distilled from BFS solutions lives in the WEIGHTS, carried to hidden games (no
stored answers). Research-informed: OPTIMISTIC weighting on reward steps
(2602.10044) so the sparse level-completion signal isn't drowned.

v17-style logging: per epoch we log train loss AND held-out next-frame accuracy
on games kept OUT of training — the honest generalisation metric (does it predict
dynamics of games it never trained on?). Each epoch's row goes to WM_LOG.md.

Output: wm_weights.pt
"""
from __future__ import annotations
import os, sys, time, argparse
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_agent import featurize, ResBlock          # reuse 21-ch featurizer

HERE = os.path.dirname(os.path.abspath(__file__))
# held-out games (never trained on) — the transfer test for the world model
HELDOUT_GAMES = {"cn04", "ka59", "sk48", "tu93", "wa30"}
GAME_ORDER = ['ar25', 'cd82', 'cn04', 'dc22', 'ft09', 'g50t', 'ka59', 'lp85',
              'ls20', 'm0r0', 'r11l', 's5i5', 'sk48', 'sp80', 'tu93', 'vc33']


class WorldModel(nn.Module):
    """featurized frame (21) + action-id embedding (8, broadcast) + click map (1)
    -> next-frame logits (16 per pixel) + reward logit."""
    def __init__(self):
        super().__init__()
        self.act_emb = nn.Embedding(8, 8)
        self.stem = nn.Sequential(nn.Conv2d(21 + 8 + 1, 64, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.res1 = ResBlock(128); self.res2 = ResBlock(128)
        self.nf_head = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(64, 16, 1))
        self.r_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, frames, actions):
        B = frames.shape[0]
        feat = featurize(frames)                                  # (B,21,64,64)
        aid = actions[:, 0].clamp(0, 7)
        ae = self.act_emb(aid).view(B, 8, 1, 1).expand(B, 8, 64, 64)
        clk = torch.zeros(B, 1, 64, 64, device=frames.device)
        is_click = actions[:, 0] == 6
        if is_click.any():
            idx = torch.nonzero(is_click).squeeze(1)
            ys = actions[idx, 2].clamp(0, 63); xs = actions[idx, 1].clamp(0, 63)
            clk[idx, 0, ys, xs] = 1.0
        x = torch.cat([feat, ae, clk], dim=1)
        h = self.res2(self.res1(self.stem(x)))
        return self.nf_head(h), self.r_head(h).squeeze(1)


def load():
    d = np.load(os.path.join(HERE, "wm_data.npz"))
    return (d["frames"].astype(np.int64), d["next_frames"].astype(np.int64),
            d["actions"].astype(np.int64), d["rewards"].astype(np.float32),
            d["game_ids"].astype(np.int64))


def evaluate(net, frames, nframes, actions, device, bsz=128):
    """Return overall + changed-pixel next-frame accuracy."""
    net.eval(); ok = tot = okc = totc = 0
    with torch.no_grad():
        for i in range(0, len(frames), bsz):
            fr = torch.from_numpy(frames[i:i+bsz]).to(device)
            nf = torch.from_numpy(nframes[i:i+bsz]).to(device)
            ac = torch.from_numpy(actions[i:i+bsz]).to(device)
            logits, _ = net(fr, ac)
            pred = logits.argmax(1)
            ok += (pred == nf).sum().item(); tot += nf.numel()
            ch = (fr != nf)
            okc += ((pred == nf) & ch).sum().item(); totc += ch.sum().item()
    net.train()
    return ok / max(tot, 1), okc / max(totc, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bsz", type=int, default=128)
    ap.add_argument("--reward-weight", type=float, default=20.0)  # optimistic upweight
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    F_, NF, A, R, G = load()
    heldout_idx = {GAME_ORDER.index(g) for g in HELDOUT_GAMES if g in GAME_ORDER}
    is_ho = np.isin(G, list(heldout_idx))
    tr = ~is_ho
    print(f"[wm] device={device} train={tr.sum()} held-out={is_ho.sum()} "
          f"(games {sorted(GAME_ORDER[i] for i in heldout_idx)})")
    Ftr, NFtr, Atr, Rtr = F_[tr], NF[tr], A[tr], R[tr]
    Fho, NFho, Aho = F_[is_ho], NF[is_ho], A[is_ho]

    net = WorldModel().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    n = len(Ftr); logp = os.path.join(HERE, "WM_LOG.md")
    if not os.path.exists(logp):
        with open(logp, "w") as f:
            f.write("# v19 world-model training log (v17-style)\n\n")
            f.write("Held-out = games never trained on. **chg-acc** = next-frame "
                    "accuracy on CHANGED pixels (the real dynamics metric).\n\n")
            f.write("| epoch | train_loss | HO pix-acc | HO chg-acc | best | secs |\n")
            f.write("|---|---|---|---|---|---|\n")
    best_chg = 0.0; t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, args.bsz):
            idx = perm[i:i+args.bsz]
            fr = torch.from_numpy(Ftr[idx]).to(device)
            nf = torch.from_numpy(NFtr[idx]).to(device)
            ac = torch.from_numpy(Atr[idx]).to(device)
            rw = torch.from_numpy(Rtr[idx]).to(device)
            logits, rlog = net(fr, ac)
            loss_nf = F.cross_entropy(logits, nf)
            w = 1.0 + (args.reward_weight - 1.0) * rw          # optimistic upweight
            loss_r = (F.binary_cross_entropy_with_logits(rlog, rw, reduction="none") * w).mean()
            loss = loss_nf + 0.5 * loss_r
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        pix, chg = evaluate(net, Fho, NFho, Aho, device)
        better = chg > best_chg
        if better:
            best_chg = chg; torch.save(net.state_dict(), os.path.join(HERE, "wm_weights.pt"))
        if ep % 5 == 0 or ep == 1 or better:
            with open(logp, "a") as f:
                f.write(f"| {ep} | {tot/nb:.4f} | {pix:.3f} | {chg:.3f} | "
                        f"{'*' if better else ''} | {time.time()-t0:.0f} |\n")
            print(f"  ep{ep}: loss={tot/nb:.4f} HO pix-acc={pix:.3f} chg-acc={chg:.3f}"
                  f"{'  <- best, saved' if better else ''}")
    with open(logp, "a") as f:
        f.write(f"\n<!-- done@{datetime.now():%H:%M:%S} best HO chg-acc={best_chg:.3f} -->\n")
    print(f"[wm] done. best held-out changed-pixel acc={best_chg:.3f} -> wm_weights.pt")


if __name__ == "__main__":
    main()
