#!/usr/bin/env python3
"""Exp A — colour-permutation (+ optional D4) augmentation for the world model.

Hypothesis: the WM's held-out chg-acc plateaus ~0.17 because pixel features
memorise each game's specific COLOURS instead of colour-invariant mechanics. ARC
colours are arbitrary labels, so permuting them (consistently in frame+next_frame,
action unchanged) should force the WM to learn transferable dynamics.

This is a STANDALONE A/B harness (doesn't touch the parallel session's
train_wm_v19.py): it trains an identical WorldModel on RAW vs AUGMENTED training
transitions and compares held-out chg-acc on the SAME raw held-out games.

    python wm_augment.py --epochs 14 --ncolor 4        # colour perm x4
    python wm_augment.py --epochs 14 --ncolor 4 --d4    # + D4
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_wm_v19 import WorldModel, evaluate, is_heldout, load

HERE = os.path.dirname(os.path.abspath(__file__))


def color_perm(frames, nframes, rng):
    """Random consistent permutation of non-bg colours 1..15 (per sample)."""
    of, on = frames.copy(), nframes.copy()
    for i in range(len(frames)):
        lut = np.arange(16)
        lut[1:16] = rng.permutation(np.arange(1, 16))
        of[i] = lut[frames[i]]; on[i] = lut[nframes[i]]
    return of, on


def d4(g, op):
    return [g, np.rot90(g, 1), np.rot90(g, 2), np.rot90(g, 3),
            np.fliplr(g), np.flipud(g), g.T, np.rot90(g, 2).T][op]


def d4_click(x, y, op, n=64):
    pts = {0: (x, y), 1: (y, n - 1 - x), 2: (n - 1 - x, n - 1 - y),
           3: (n - 1 - y, x), 4: (n - 1 - x, y), 5: (x, n - 1 - y),
           6: (y, x), 7: (n - 1 - y, n - 1 - x)}
    return pts[op]


def augment(F_, NF, A, ncolor, do_d4, seed=0):
    rng = np.random.default_rng(seed)
    Fs, NFs, As = [F_], [NF], [A]
    for _ in range(ncolor):                       # colour permutations
        cf, cn = color_perm(F_, NF, rng)
        Fs.append(cf); NFs.append(cn); As.append(A)
    if do_d4:
        for op in range(1, 8):                    # 7 non-identity D4 ops
            of = np.stack([d4(f, op) for f in F_]).copy()
            on = np.stack([d4(f, op) for f in NF]).copy()
            aa = A.copy()
            clk = A[:, 0] == 6
            for i in np.nonzero(clk)[0]:
                aa[i, 1], aa[i, 2] = d4_click(A[i, 1], A[i, 2], op)
            Fs.append(of); NFs.append(on); As.append(aa)
    return np.concatenate(Fs), np.concatenate(NFs), np.concatenate(As)


def train_eval(Ftr, NFtr, Atr, Rtr, Fho, NFho, Aho, device, epochs, bsz=128):
    net = WorldModel().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    n = len(Ftr); best = 0.0
    for ep in range(epochs):
        perm = np.random.permutation(n)
        for i in range(0, n, bsz):
            idx = perm[i:i+bsz]
            fr = torch.from_numpy(Ftr[idx]).to(device)
            nf = torch.from_numpy(NFtr[idx]).to(device)
            ac = torch.from_numpy(Atr[idx]).to(device)
            rw = torch.from_numpy(Rtr[idx]).to(device)
            logits, rlog = net(fr, ac)
            loss = F.cross_entropy(logits, nf) + 0.5 * F.binary_cross_entropy_with_logits(rlog, rw)
            opt.zero_grad(); loss.backward(); opt.step()
        _, chg = evaluate(net, Fho, NFho, Aho, device)
        best = max(best, chg)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--ncolor", type=int, default=4)
    ap.add_argument("--d4", action="store_true")
    args = ap.parse_args()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    F_, NF, A, R, G, games = load()
    if games is None:
        games = [str(i) for i in range(int(G.max()) + 1)]
    ho = np.array([is_heldout(games[g]) for g in G])
    tr = ~ho
    Fho, NFho, Aho = F_[ho], NF[ho], A[ho]
    Ftr, NFtr, Atr, Rtr = F_[tr].astype(np.int64), NF[tr].astype(np.int64), A[tr].astype(np.int64), R[tr]
    print(f"[augA] device={device} train={tr.sum()} held-out={ho.sum()} "
          f"games={len(games)} | ncolor={args.ncolor} d4={args.d4}")

    t0 = time.time()
    base = train_eval(Ftr, NFtr, Atr, Rtr, Fho, NFho, Aho, device, args.epochs)
    print(f"[augA] BASELINE (raw) held-out chg-acc = {base:.3f}  ({time.time()-t0:.0f}s)")

    aF, aNF, aA = augment(Ftr, NFtr, Atr, args.ncolor, args.d4)
    aR = np.tile(R[tr], len(aF) // len(Ftr))
    t1 = time.time()
    aug = train_eval(aF.astype(np.int64), aNF.astype(np.int64), aA.astype(np.int64), aR,
                     Fho, NFho, Aho, device, args.epochs)
    print(f"[augA] AUGMENTED ({len(aF)} samples) held-out chg-acc = {aug:.3f}  ({time.time()-t1:.0f}s)")
    print(f"[augA] >>> lift: {aug-base:+.3f}  ({'AUG HELPS' if aug>base+0.01 else 'no clear lift'})")
    with open(os.path.join(HERE, "WM_REPR_EXPERIMENT.md"), "a") as f:
        f.write(f"\n<!-- ExpA {time.strftime('%m-%d %H:%M')}: baseline={base:.3f} "
                f"aug(ncolor={args.ncolor},d4={args.d4})={aug:.3f} lift={aug-base:+.3f} -->\n")


if __name__ == "__main__":
    main()
