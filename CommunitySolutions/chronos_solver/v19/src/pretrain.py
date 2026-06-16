"""Lever #1 — offline-pretrain ChangeNet on the PUBLIC TRAIN games.

The winning preview agent (StochasticGoose) ran on a CNN that predicts which
action changes the frame. Its edge was that the prior generalised. We reproduce
that: drive the FORGE agent's own exploration on the TRAIN games, harvest its
ground-truth-labelled (frame, action -> changed/novel) transitions, and train a
fresh ChangeNet on the union. Ship the weights as a warm start for unseen games.

HELD-OUT (cn04, ka59, sk48, tu93, wa30) is NEVER trained on — it is the transfer
test, and now also the SAVE GATE: each epoch we measure held-out change-prediction
accuracy and only overwrite pretrained_weights.pt when transfer improves. If it
plateaus for --patience epochs we stop early and say so — that is the signal that
more compute/epochs won't help and you should grow the corpus or the features.
lf52/tn36 skipped (load errors). Labels come from the real engine, not the model.

GPU scaling (for a big offline box like the RTX PRO 6000):
  --net-mult 4   wider, higher-capacity prior (auto-loaded at inference)
  --bsz -1       batch auto-scales with VRAM (1024 @96GB, 512 @48GB, 256 @T4)
  --amp          mixed precision (cuda only)

Output: v19/pretrained_weights.pt        Smoke test: python pretrain.py --selftest
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from forge_agent import (ForgeAgent, ChangeNet, featurize, auto_bsz,   # noqa: E402
                         make_grad_scaler, amp_autocast,
                         N_SIMPLE, CLICK_BASE, GRID)

HERE = os.path.dirname(__file__)
HELDOUT = {"cn04", "ka59", "sk48", "tu93", "wa30"}
KNOWN_BAD = {"lf52", "tn36"}
ALL = ["ar25", "bp35", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "lf52",
       "lp85", "ls20", "m0r0", "r11l", "re86", "s5i5", "sb26", "sc25", "sk48",
       "sp80", "su15", "tn36", "tr87", "tu93", "vc33", "wa30"]
TRAIN = [g for g in ALL if g not in HELDOUT and g not in KNOWN_BAD]


def harvest(game, n_actions, device):
    """Run FORGE exploration; return its (frame, akey, target) buffer."""
    from blackbox_env import BlackBoxEnv          # lazy: only offline needs the engine
    env = BlackBoxEnv(game); obs = env.reset()
    ag = ForgeAgent(seed=0, device=device); ag.reset(game)
    for _ in range(n_actions):
        aid, data = ag.act(obs)
        obs = env.step(aid, data)
        if obs.state == "WIN":
            break
    return list(ag.buf)


def _logit_for(a_logits, c_logits, j, kk):
    """The model's change-logit for sample j's taken action key kk."""
    if kk < N_SIMPLE:
        return a_logits[j, kk]
    ci = kk - CLICK_BASE
    return c_logits[j, ci // GRID, ci % GRID]


def eval_changenet(net, samples, device, bsz=256):
    """Held-out change-prediction accuracy (logit>0 vs the engine's changed/not)."""
    if not samples:
        return 0.0
    frames = np.stack([s[0] for s in samples]).astype(np.int64)
    keys = np.array([s[1] for s in samples], dtype=np.int64)
    targs = np.array([s[2] for s in samples], dtype=np.float32)
    net.eval(); ok = tot = 0
    with torch.no_grad():
        for i in range(0, len(samples), bsz):
            x = featurize(torch.from_numpy(frames[i:i + bsz]).to(device))
            a_logits, c_logits = net(x)
            for j, kk in enumerate(keys[i:i + bsz]):
                pred = 1.0 if float(_logit_for(a_logits, c_logits, j, int(kk))) > 0 else 0.0
                ok += int(pred == targs[i + j]); tot += 1
    net.train()
    return ok / max(tot, 1)


def train(samples, device, epochs, bsz=128, lr=3e-4, net_mult=1, amp=False,
          ho_samples=None, patience=4, out_path=None):
    """Train ChangeNet. Gated: when ho_samples is given, save to out_path only on
    held-out improvement and early-stop on a --patience plateau. Ungated: save the
    final net. Returns (net, best_heldout_acc)."""
    use_amp = bool(amp) and getattr(device, "type", "") == "cuda"
    scaler = make_grad_scaler(use_amp)
    net = ChangeNet(mult=net_mult).to(device)
    # warm-start from the existing prior when widths match, so knowledge accumulates
    if out_path and os.path.exists(out_path):
        try:
            sd = torch.load(out_path, map_location=device, weights_only=True)
            if ChangeNet.mult_of(sd) == net_mult:
                net.load_state_dict(sd); print("  warm-started from existing prior")
        except Exception:
            pass
    opt = optim.Adam(net.parameters(), lr=lr)
    frames = np.stack([s[0] for s in samples]).astype(np.int64)
    keys = np.array([s[1] for s in samples], dtype=np.int64)
    targs = np.array([s[2] for s in samples], dtype=np.float32)
    n = len(samples)
    gated = bool(ho_samples)
    best_acc = -1.0; since = 0
    if gated and out_path and os.path.exists(out_path):
        best_acc = eval_changenet(net, ho_samples, device)
        print(f"  held-out transfer (warm baseline) acc={best_acc:.3f} — keep only if beaten")
    for ep in range(epochs):
        perm = np.random.permutation(n); tot = 0.0; nb = 0
        for i in range(0, n, bsz):
            idx = perm[i:i + bsz]
            x = featurize(torch.from_numpy(frames[idx]).to(device))
            with amp_autocast(use_amp):
                a_logits, c_logits = net(x)
                sel = torch.empty(len(idx), device=device)
                for j, kk in enumerate(keys[idx]):
                    sel[j] = _logit_for(a_logits, c_logits, j, int(kk))
                t = torch.from_numpy(targs[idx]).to(device)
                loss = F.binary_cross_entropy_with_logits(sel, t)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nb += 1
        if gated:
            acc = eval_changenet(net, ho_samples, device)
            better = acc > best_acc
            if better:
                best_acc = acc; since = 0
                if out_path:
                    torch.save(net.state_dict(), out_path)
            else:
                since += 1
            print(f"  epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}  HO acc={acc:.3f}"
                  f"{'  <- best, saved' if better else ''}")
            if since >= patience:
                print(f"  PLATEAU: held-out acc {best_acc:.3f} not beaten for {patience} "
                      f"epochs — more compute/epochs won't help; grow the corpus/features.")
                break
        else:
            print(f"  epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}")
    if not gated and out_path:
        torch.save(net.state_dict(), out_path)
    return net, best_acc


def _selftest():
    """No engine needed: verify width-scaling, AMP-off path, gate, and the
    save -> infer-width -> reload round-trip the shipped agent relies on."""
    import tempfile
    dev = torch.device("cpu"); rng = np.random.default_rng(0)
    def mk(n):
        return [(rng.integers(0, 16, (64, 64)).astype(np.int64),
                 int(rng.integers(0, N_SIMPLE)), float(rng.integers(0, 2))) for _ in range(n)]
    out = os.path.join(tempfile.gettempdir(), "pretrain_selftest.pt")
    if os.path.exists(out):
        os.remove(out)
    net, best = train(mk(48), dev, epochs=2, bsz=8, net_mult=2, amp=False,
                      ho_samples=mk(16), patience=5, out_path=out)
    assert os.path.exists(out), "gate did not save a checkpoint"
    sd = torch.load(out, map_location="cpu", weights_only=True)
    assert ChangeNet.mult_of(sd) == 2, f"width infer wrong: {ChangeNet.mult_of(sd)}"
    ChangeNet.from_state_dict(sd)                       # must reconstruct mult=2 + load
    ForgeAgent(weights=out).reset("selftest")           # shipped path must load it
    print(f"[pretrain] selftest OK: mult=2 trained+saved, width-inferred reload OK, "
          f"agent loads it, HO acc={best:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_game", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bsz", type=int, default=-1, help="-1 = auto from VRAM")
    ap.add_argument("--net-mult", type=int, default=1, dest="net_mult",
                    help="trunk width multiplier (4 = beast prior for a big GPU)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (cuda only)")
    ap.add_argument("--patience", type=int, default=4, help="held-out plateau early-stop")
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the held-out transfer gate (save the final net)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return _selftest()
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    bsz = args.bsz if args.bsz > 0 else auto_bsz(device)
    print(f"[pretrain] device={device} bsz={bsz} net_mult={args.net_mult} amp={args.amp} "
          f"TRAIN={len(TRAIN)} games per_game={args.per_game}")
    samples, t0 = [], time.time()
    for g in TRAIN:
        try:
            s = harvest(g, args.per_game, device)
            samples += s
            print(f"  {g}: +{len(s)} samples (total {len(samples)})")
        except Exception as e:
            print(f"  {g}: ERROR {repr(e)[:70]}")
    print(f"[pretrain] harvested {len(samples)} samples in {time.time()-t0:.0f}s")
    pos = sum(1 for s in samples if s[2] > 0.5)
    print(f"[pretrain] label balance: {pos}/{len(samples)} changed ({pos/max(len(samples),1):.0%})")
    ho = []
    if not args.no_gate:
        for g in sorted(HELDOUT):
            try:
                ho += harvest(g, max(120, args.per_game // 4), device)
            except Exception as e:
                print(f"  held-out {g}: ERROR {repr(e)[:60]}")
        print(f"[pretrain] held-out transfer set: {len(ho)} samples from {sorted(HELDOUT)}")
    out = os.path.join(HERE, "pretrained_weights.pt")
    print("[pretrain] training…")
    net, best = train(samples, device, args.epochs, bsz=bsz, lr=3e-4,
                      net_mult=args.net_mult, amp=args.amp,
                      ho_samples=(ho or None), patience=args.patience, out_path=out)
    print(f"[pretrain] saved -> {out}  (held-out acc={best:.3f}, mult={args.net_mult}, "
          f"bsz={bsz}, amp={args.amp})")


if __name__ == "__main__":
    main()
