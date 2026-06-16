"""Puzzle-LM tokenizer — v15-style patch/codebook, in pure numpy.

The iter-2/3 finding: the 76-d object-feature summary is the ceiling (color
counts/centroids throw away geometry). This replaces it with a spatial token
representation, the way v15 used a CNN+VQ codebook over 8x8 patches:

  frame 64x64  -> 8x8 = 64 patches, each an 8x8 colour block
  patch        -> 16-d colour histogram
  codebook     -> K cluster centroids (k-means, fit offline on pooled frames)
  tokenize     -> 64 token ids (an 8x8 token grid) = the frame's "spelling"
  token_feature-> coarse RxR region token-histograms (spatial layout preserved)

Unlike the 76-d summary, token_feature knows WHICH region holds WHICH token —
the geometry dynamics and policy actually depend on. k-means stands in for v15's
straight-through VQ (no gradients needed, Kaggle-legal, CPU-cheap).
"""
from __future__ import annotations
import os, numpy as np

PATCH = 8
GRID = 8                # 64/8
NCOL = 16
DEFAULT_K = 24
REGIONS = 4             # token_feature uses a 4x4 region grid


def _patch_hist(frame):
    """frame(64,64) -> (64, 16) per-patch colour histograms."""
    H = np.zeros((GRID * GRID, NCOL), np.float32)
    idx = 0
    for i in range(GRID):
        for j in range(GRID):
            block = frame[i*PATCH:(i+1)*PATCH, j*PATCH:(j+1)*PATCH]
            H[idx] = np.bincount(block.flatten(), minlength=NCOL)[:NCOL]
            idx += 1
    return H / (PATCH * PATCH)


class Tokenizer:
    def __init__(self, K=DEFAULT_K):
        self.K = K
        self.cb = None          # (K, 16) centroids

    # ---- offline codebook fit (k-means / Lloyd) ----
    def fit(self, frames, iters=15, seed=0):
        rng = np.random.RandomState(seed)
        P = np.concatenate([_patch_hist(f) for f in frames], 0)   # (N*64, 16)
        # init from random distinct patches
        c = P[rng.choice(len(P), self.K, replace=False)].copy()
        for _ in range(iters):
            d = ((P[:, None, :] - c[None, :, :]) ** 2).sum(2)     # (N, K)
            a = d.argmin(1)
            for k in range(self.K):
                m = a == k
                if m.any():
                    c[k] = P[m].mean(0)
        self.cb = c.astype(np.float32)
        return self

    def tokenize(self, frame):
        H = _patch_hist(frame)                                    # (64,16)
        d = ((H[:, None, :] - self.cb[None, :, :]) ** 2).sum(2)
        return d.argmin(1).reshape(GRID, GRID)                    # (8,8) token ids

    def token_feature(self, frame):
        """RxR region token-histograms -> spatial token feature (R*R*K dims)."""
        tg = self.tokenize(frame)
        feat = np.zeros((REGIONS, REGIONS, self.K), np.float32)
        step = GRID // REGIONS
        for ri in range(REGIONS):
            for rj in range(REGIONS):
                blk = tg[ri*step:(ri+1)*step, rj*step:(rj+1)*step].flatten()
                feat[ri, rj] = np.bincount(blk, minlength=self.K)[:self.K]
        return (feat / (step * step)).reshape(-1)                 # 16*K

    @property
    def feat_dim(self):
        return REGIONS * REGIONS * self.K

    def save(self, path):
        np.savez(path, cb=self.cb, K=self.K)

    @classmethod
    def load(cls, path):
        d = np.load(path); t = cls(K=int(d["K"])); t.cb = d["cb"]; return t


def combined_feature(frame, tok, obj_fn):
    """token_feature (spatial) ++ object_features (summary) = strict superset."""
    return np.concatenate([tok.token_feature(frame), obj_fn(frame)]).astype(np.float32)
