"""Puzzle-LM v1 — a cross-game model of puzzle dynamics, in the spirit of v15.

Two components, both trained on the POOLED transitions from all 25 games
(gen_pooled.py), both pure numpy:

  1. WorldModel : (state_features, action) -> next_state_features + P(progress).
     This is the v15 BlockCausalSimulator analogue, but in the shared
     object-feature "language" (76-d) instead of pixel tokens — so it is
     genuinely cross-game and cheap. It predicts the DELTA on top of a copy of
     the current state, so "nothing moved" is the easy default (the v15 lesson)
     and dynamics are learned as residuals.

  2. Apprentice : the existing TRM (recursive policy/value core), trained on the
     SAME pool so its policy prior = "actions that make progress here" and its
     value = "is this a progress-promising state", across all games.

The world model is the centrepiece (it's the "Puzzle Language Model"); the TRM
is the policy/value head used as the MCTS apprentice. Reported metric, v15-style:
next-state prediction error vs the copy baseline.
"""
from __future__ import annotations
import os, numpy as np
from vlog import get_logger

N_ACT = 5            # action classes [1,2,3,4,CLICK]
FEAT = 76


class WorldModel:
    """(feat[76] + action_onehot[5]) -> next_feat[76] (residual) + progress logit."""

    def __init__(self, hidden=128, seed=0):
        rng = np.random.RandomState(seed)
        din = FEAT + N_ACT
        self.W1 = (rng.randn(din, hidden) * np.sqrt(2.0/din)).astype(np.float32)
        self.b1 = np.zeros(hidden, np.float32)
        self.Wd = (rng.randn(hidden, FEAT) * 0.01).astype(np.float32)   # residual delta
        self.bd = np.zeros(FEAT, np.float32)
        self.Wp = (rng.randn(hidden, 1) * np.sqrt(1.0/hidden)).astype(np.float32)
        self.bp = np.zeros(1, np.float32)
        self._adam = {}
        self.x_mean = None; self.x_std = None

    def _enc(self, feat, act):
        oh = np.zeros((len(act), N_ACT), np.float32); oh[np.arange(len(act)), act] = 1.0
        return np.concatenate([feat, oh], 1)

    def _norm(self, X):
        return X if self.x_mean is None else (X - self.x_mean) / self.x_std

    def forward(self, feat, act):
        X = self._norm(self._enc(np.atleast_2d(feat).astype(np.float32),
                                 np.atleast_1d(act)))
        z1 = X @ self.W1 + self.b1; a1 = np.maximum(z1, 0.0)
        delta = a1 @ self.Wd + self.bd
        prog = 1.0/(1.0+np.exp(-(a1 @ self.Wp + self.bp)))
        nxt = np.atleast_2d(feat).astype(np.float32) + delta       # residual on copy
        return nxt, prog[:, 0], (X, z1, a1)

    def predict_next(self, feat, act):
        return self.forward(feat, act)[0]

    def fit(self, S, A, NS, P, epochs=120, lr=3e-3, bsz=128, iter_tag="puzzle", l2=1e-5):
        lg = get_logger("worldmodel", iter_tag)
        S = np.asarray(S, np.float32); NS = np.asarray(NS, np.float32)
        A = np.asarray(A, np.int64); P = np.asarray(P, np.float32)
        Xraw = self._enc(S, A)
        self.x_mean = Xraw.mean(0); self.x_std = Xraw.std(0) + 1e-6
        target_delta = NS - S
        n = len(S)
        copy_mse = float((target_delta**2).mean())     # baseline: predict no change
        lg.info(f"[WM-FIT] n={n} copy_baseline_mse(delta)={copy_mse:.5f} epochs={epochs}")
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, bsz):
                bi = idx[s:s+bsz]
                feat = S[bi]; act = A[bi]
                nxt, prog, (X, z1, a1) = self.forward(feat, act)
                m = len(bi)
                # delta regression
                gdelta = (2.0/m) * (nxt - NS[bi])            # d/d(delta) = d/d(nxt)
                gWd = a1.T @ gdelta + l2*self.Wd; gbd = gdelta.sum(0)
                # progress BCE
                gp = (1.0/m) * (prog - P[bi])[:, None]
                gWp = a1.T @ gp + l2*self.Wp; gbp = gp.sum(0)
                ga1 = gdelta @ self.Wd.T + gp @ self.Wp.T
                gz1 = ga1 * (z1 > 0)
                gW1 = X.T @ gz1 + l2*self.W1; gb1 = gz1.sum(0)
                self._step(lr, dict(W1=gW1, b1=gb1, Wd=gWd, bd=gbd, Wp=gWp, bp=gbp))
            if ep % max(1, epochs//6) == 0 or ep == epochs-1:
                nxt, prog, _ = self.forward(S, A)
                mse = float(((nxt - NS)**2).mean())
                pacc = float(((prog > 0.5) == (P > 0.5)).mean())
                lg.info(f"[WM-FIT] ep={ep:3d} next_mse={mse:.5f} "
                        f"(copy={copy_mse:.5f}, {100*(1-mse/copy_mse):.1f}% better) "
                        f"progress_acc={pacc:.3f}")
        return self

    def _step(self, lr, grads, b1=0.9, b2=0.999, eps=1e-8):
        for k, g in grads.items():
            p = getattr(self, k)
            st = self._adam.setdefault(k, {"m": np.zeros_like(p), "v": np.zeros_like(p), "t": 0})
            st["t"] += 1
            st["m"] = b1*st["m"] + (1-b1)*g
            st["v"] = b2*st["v"] + (1-b2)*(g*g)
            mh = st["m"]/(1-b1**st["t"]); vh = st["v"]/(1-b2**st["t"])
            setattr(self, k, p - lr*mh/(np.sqrt(vh)+eps))

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in
                          ("W1", "b1", "Wd", "bd", "Wp", "bp", "x_mean", "x_std")})

    @classmethod
    def load(cls, path):
        d = np.load(path); m = cls()
        for k in ("W1", "b1", "Wd", "bd", "Wp", "bp", "x_mean", "x_std"):
            setattr(m, k, d[k])
        return m
