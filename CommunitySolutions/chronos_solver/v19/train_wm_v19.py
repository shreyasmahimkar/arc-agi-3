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
import os, sys, time, argparse, hashlib
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forge_agent import (featurize, ResBlock, auto_bsz,   # reuse 21-ch featurizer
                         make_grad_scaler, amp_autocast)

HERE = os.path.dirname(os.path.abspath(__file__))
# Held-out games (never trained on) — the WM transfer test. Scales to any game
# set: the 5 official held-out PLUS a stable ~10% hash bucket of every other game.
HELDOUT_GAMES = {"cn04", "ka59", "sk48", "tu93", "wa30"}


def is_heldout(game: str) -> bool:
    if game in HELDOUT_GAMES:
        return True
    return int(hashlib.md5(game.encode()).hexdigest(), 16) % 10 == 0


class WorldModel(nn.Module):
    """featurized frame (21) + action-id embedding (8, broadcast) + click map (1)
    -> next-frame logits (16 per pixel) + reward logit.

    base stem width = 64; `mult` scales the trunk for a higher-capacity model on a
    big GPU. Saved as a plain state_dict; mult_of/from_state_dict let the trainer
    warm-start and the planner load whatever width was trained."""
    BASE = 64

    def __init__(self, mult=1):
        super().__init__()
        c1, c2 = 64 * mult, 128 * mult
        self.act_emb = nn.Embedding(8, 8)
        self.stem = nn.Sequential(nn.Conv2d(21 + 8 + 1, c1, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(c1, c2, 3, padding=1), nn.ReLU())
        self.res1 = ResBlock(c2); self.res2 = ResBlock(c2)
        self.nf_head = nn.Sequential(nn.Conv2d(c2, c1, 3, padding=1), nn.ReLU(),
                                     nn.Conv2d(c1, 16, 1))
        self.r_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                    nn.Linear(c2, c1), nn.ReLU(), nn.Linear(c1, 1))

    @staticmethod
    def mult_of(sd):
        return max(1, int(sd["stem.0.weight"].shape[0]) // WorldModel.BASE)

    @classmethod
    def from_state_dict(cls, sd):
        net = cls(mult=cls.mult_of(sd)); net.load_state_dict(sd); return net

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
    d = np.load(os.path.join(HERE, "wm_data.npz"), allow_pickle=True)
    games = list(d["games"]) if "games" in d else None
    return (d["frames"].astype(np.int64), d["next_frames"].astype(np.int64),
            d["actions"].astype(np.int64), d["rewards"].astype(np.float32),
            d["game_ids"].astype(np.int64), games)


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
    ap.add_argument("--bsz", type=int, default=-1, help="-1 = auto from VRAM")
    ap.add_argument("--net-mult", type=int, default=1, dest="net_mult",
                    help="trunk width multiplier (4 = beast model for a big GPU)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (cuda only)")
    ap.add_argument("--patience", type=int, default=12,
                    help="held-out chg-acc plateau early-stop (more epochs won't help)")
    ap.add_argument("--reward-weight", type=float, default=20.0)  # optimistic upweight
    ap.add_argument("--scratch", action="store_true",
                    help="fresh random init instead of warm-starting from wm_weights.pt")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    bsz = args.bsz if args.bsz > 0 else auto_bsz(device)
    use_amp = bool(args.amp) and device.type == "cuda"
    F_, NF, A, R, G, games = load()
    if games is None:
        games = [str(i) for i in range(int(G.max()) + 1)]
    heldout_idx = {i for i, g in enumerate(games) if is_heldout(g)}
    is_ho = np.isin(G, list(heldout_idx)) if heldout_idx else np.zeros(len(G), bool)
    tr = ~is_ho
    print(f"[wm] device={device} bsz={bsz} net_mult={args.net_mult} amp={use_amp} "
          f"games={len(games)} train={tr.sum()} held-out={is_ho.sum()} "
          f"(HO games {sorted(games[i] for i in heldout_idx)})")
    Ftr, NFtr, Atr, Rtr = F_[tr], NF[tr], A[tr], R[tr]
    Fho, NFho, Aho = F_[is_ho], NF[is_ho], A[is_ho]

    net = WorldModel(mult=args.net_mult).to(device)
    # WARM-START: continue from the previous world model so knowledge ACCUMULATES
    # across cycles instead of re-learning from scratch each run — but only when the
    # widths match (--net-mult change starts a fresh lineage). (--scratch forces fresh.)
    wpath = os.path.join(HERE, "wm_weights.pt")
    warm = False
    if not args.scratch and os.path.exists(wpath):
        try:
            sd = torch.load(wpath, map_location=device, weights_only=True)
            if WorldModel.mult_of(sd) == args.net_mult:
                net.load_state_dict(sd); warm = True
            else:
                print(f"[wm] wm_weights.pt is mult={WorldModel.mult_of(sd)}, requested "
                      f"{args.net_mult} -> fresh init (new lineage)")
        except Exception:
            pass
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    scaler = make_grad_scaler(use_amp)
    n = len(Ftr); logp = os.path.join(HERE, "WM_LOG.md")
    if not os.path.exists(logp):
        with open(logp, "w") as f:
            f.write("# v19 world-model training log (v17-style)\n\n")
            f.write("Held-out = games never trained on. **chg-acc** = next-frame "
                    "accuracy on CHANGED pixels (the real dynamics metric).\n\n")
            f.write("| epoch | train_loss | HO pix-acc | HO chg-acc | best | secs |\n")
            f.write("|---|---|---|---|---|---|\n")
    # only KEEP this run's weights if held-out generalisation beats the warm
    # baseline — so the saved model improves monotonically across cycles.
    best_chg = 0.0; since = 0; t0 = time.time()
    if warm:
        _, best_chg = evaluate(net, Fho, NFho, Aho, device)
        print(f"[wm] warm-start: prior held-out chg-acc={best_chg:.3f} (keep only if beaten)")
    for ep in range(1, args.epochs + 1):
        perm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, bsz):
            idx = perm[i:i+bsz]
            fr = torch.from_numpy(Ftr[idx]).to(device)
            nf = torch.from_numpy(NFtr[idx]).to(device)
            ac = torch.from_numpy(Atr[idx]).to(device)
            rw = torch.from_numpy(Rtr[idx]).to(device)
            with amp_autocast(use_amp):
                logits, rlog = net(fr, ac)
                loss_nf = F.cross_entropy(logits, nf)
                w = 1.0 + (args.reward_weight - 1.0) * rw      # optimistic upweight
                loss_r = (F.binary_cross_entropy_with_logits(rlog, rw, reduction="none") * w).mean()
                loss = loss_nf + 0.5 * loss_r
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nb += 1
        pix, chg = evaluate(net, Fho, NFho, Aho, device)
        better = chg > best_chg
        if better:
            best_chg = chg; since = 0
            torch.save(net.state_dict(), os.path.join(HERE, "wm_weights.pt"))
        else:
            since += 1
        if ep % 5 == 0 or ep == 1 or better:
            with open(logp, "a") as f:
                f.write(f"| {ep} | {tot/nb:.4f} | {pix:.3f} | {chg:.3f} | "
                        f"{'*' if better else ''} | {time.time()-t0:.0f} |\n")
            print(f"  ep{ep}: loss={tot/nb:.4f} HO pix-acc={pix:.3f} chg-acc={chg:.3f}"
                  f"{'  <- best, saved' if better else ''}")
        if since >= args.patience:
            print(f"  PLATEAU: held-out chg-acc {best_chg:.3f} not beaten for {args.patience} "
                  f"epochs — more compute won't help; grow the corpus or the model.")
            break
    with open(logp, "a") as f:
        f.write(f"\n<!-- done@{datetime.now():%H:%M:%S} best HO chg-acc={best_chg:.3f} "
                f"mult={args.net_mult} amp={use_amp} -->\n")
    print(f"[wm] done. best held-out changed-pixel acc={best_chg:.3f} -> wm_weights.pt")


if __name__ == "__main__":
    main()
