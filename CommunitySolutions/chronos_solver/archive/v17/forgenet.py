"""v17 ForgeNet — the CNN cost-to-go heuristic (Track A of BFSLLM_RESEARCH).

Pure numpy (the dev sandbox has no torch). It is a genuine CNN feature
extractor + a trained MLP readout:

  frame 64x64 --downsample--> 16x16 --one-hot(16)--> [16,16,16]
        |--> fixed conv bank (K random 3x3xC filters, ReLU) --> +avg/+max pool
        |--> object_features (76-dim, from engine.object_features)
        concat --> MLP(hidden, ReLU) --> scalar cost-to-go (steps-to-goal)

Why fixed conv + trained head: in numpy a fully-trained deep CNN is slow and
fragile to hand-backprop; a fixed random conv bank ("convolutional random
features") gives translation-aware spatial features for free, and the MLP
head (which IS backpropped here, Adam) learns the cost-to-go mapping. This
is the cheapest faithful port of "BFS depth labels -> a heuristic" and plugs
straight into search.solve_level(strategy='greedy'/'astar') as heuristic_fn.

Trained on solution-path states: for a level solved in S actions, the state
after i actions has cost-to-go = S - i. Lower heuristic == closer to winning.
"""
from __future__ import annotations
import numpy as np
import json, os
from vlog import get_logger

K_FILTERS = 16
DS = 16            # downsample grid
C = 16            # colors


def _downsample_onehot(frame):
    """64x64 int frame -> (DS,DS,C) one-hot via block-mode downsample."""
    h, w = frame.shape
    bh, bw = h // DS, w // DS
    out = np.zeros((DS, DS, C), dtype=np.float32)
    for i in range(DS):
        for j in range(DS):
            block = frame[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            c = np.bincount(block.flatten(), minlength=C).argmax()
            out[i, j, int(c)] = 1.0
    return out


class _ConvBank:
    """Fixed (seeded) 3x3xC conv filters -> ReLU -> global avg+max pool."""

    def __init__(self, k=K_FILTERS, seed=17):
        rng = np.random.RandomState(seed)
        self.W = rng.randn(k, 3, 3, C).astype(np.float32) * 0.3
        self.k = k

    def features(self, onehot):
        DS_ = onehot.shape[0]
        # valid conv -> (DS-2, DS-2, k)
        out = np.zeros((DS_-2, DS_-2, self.k), dtype=np.float32)
        for fi in range(self.k):
            f = self.W[fi]
            acc = np.zeros((DS_-2, DS_-2), dtype=np.float32)
            for di in range(3):
                for dj in range(3):
                    acc += np.sum(onehot[di:di+DS_-2, dj:dj+DS_-2, :] * f[di, dj], axis=2)
            out[:, :, fi] = np.maximum(acc, 0.0)        # ReLU
        avg = out.mean(axis=(0, 1))
        mx = out.max(axis=(0, 1))
        return np.concatenate([avg, mx]).astype(np.float32)   # 2k dims


class ForgeNet:
    def __init__(self, obj_dim=76, hidden=64, seed=0):
        self.conv = _ConvBank()
        self.in_dim = 2 * K_FILTERS + obj_dim
        rng = np.random.RandomState(seed)
        self.W1 = (rng.randn(self.in_dim, hidden) * np.sqrt(2.0/self.in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.randn(hidden, 1) * np.sqrt(2.0/hidden)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)
        self._adam = {}
        self.x_mean = None
        self.x_std = None

    # --- featurization ---
    def featurize(self, frame, obj_feat):
        oh = _downsample_onehot(frame)
        cf = self.conv.features(oh)
        return np.concatenate([cf, obj_feat]).astype(np.float32)

    def _norm(self, X):
        if self.x_mean is None:
            return X
        return (X - self.x_mean) / self.x_std

    # --- forward ---
    def _fwd(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0.0)
        y = a1 @ self.W2 + self.b2
        return y[:, 0], (X, z1, a1)

    def predict(self, frame, obj_feat):
        X = self._norm(self.featurize(frame, obj_feat)[None, :])
        return float(self._fwd(X)[0][0])

    # --- training (Adam on MSE of cost-to-go) ---
    def fit(self, X, y, epochs=300, lr=3e-3, bsz=64, iter_tag="v17", l2=1e-5):
        lg = get_logger("forgenet", iter_tag)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.x_mean = X.mean(0); self.x_std = X.std(0) + 1e-6
        Xn = (X - self.x_mean) / self.x_std
        n = len(Xn)
        lg.info(f"[FIT] ForgeNet n={n} in_dim={self.in_dim} epochs={epochs} "
                f"lr={lr} bsz={bsz} y_range=[{y.min():.1f},{y.max():.1f}]")
        for ep in range(epochs):
            idx = np.random.permutation(n)
            tot = 0.0
            for s in range(0, n, bsz):
                bi = idx[s:s+bsz]
                xb, yb = Xn[bi], y[bi]
                pred, (Xc, z1, a1) = self._fwd(xb)
                err = pred - yb
                tot += float((err**2).mean())
                m = len(bi)
                gy = (2.0/m) * err[:, None]
                gW2 = a1.T @ gy + l2 * self.W2
                gb2 = gy.sum(0)
                ga1 = gy @ self.W2.T
                gz1 = ga1 * (z1 > 0)
                gW1 = Xc.T @ gz1 + l2 * self.W1
                gb1 = gz1.sum(0)
                self._adam_step(lr, gW1, gb1, gW2, gb2)
            if ep % max(1, epochs//8) == 0 or ep == epochs-1:
                full = self._fwd(Xn)[0]
                rmse = float(np.sqrt(((full - y)**2).mean()))
                mae = float(np.abs(full - y).mean())
                lg.info(f"[FIT] ep={ep:3d} train_mse={tot/max(1,(n//bsz+1)):.3f} "
                        f"full_rmse={rmse:.3f} full_mae={mae:.3f}")
        return self

    def _adam_step(self, lr, gW1, gb1, gW2, gb2, b1=0.9, b2=0.999, eps=1e-8):
        for name, g, p in (("W1", gW1, self.W1), ("b1", gb1, self.b1),
                           ("W2", gW2, self.W2), ("b2", gb2, self.b2)):
            st = self._adam.setdefault(name, {"m": np.zeros_like(p), "v": np.zeros_like(p), "t": 0})
            st["t"] += 1
            st["m"] = b1*st["m"] + (1-b1)*g
            st["v"] = b2*st["v"] + (1-b2)*(g*g)
            mh = st["m"]/(1-b1**st["t"]); vh = st["v"]/(1-b2**st["t"])
            p -= lr * mh/(np.sqrt(vh)+eps)

    def save(self, path):
        np.savez(path, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 x_mean=self.x_mean, x_std=self.x_std, convW=self.conv.W)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls()
        m.W1, m.b1, m.W2, m.b2 = d["W1"], d["b1"], d["W2"], d["b2"]
        m.x_mean, m.x_std = d["x_mean"], d["x_std"]
        m.conv.W = d["convW"]
        return m
