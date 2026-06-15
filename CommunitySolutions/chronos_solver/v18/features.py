"""Frame-only feature vector — the SINGLE representation the learned agent uses
at test time. Pure function of the 64x64 frame: no engine internals, so it is
identical for a train game and a never-seen held-out game.

  object block : per colour 1..15 -> [count, cx, cy, bbox_w, bbox_h] (normalised)
                 + n_nonbg_colours                                    = 76 dims
  coarse block : frame downsampled 64x64 -> 16x16, /15                = 256 dims
                                                                 total = 332 dims
"""
from __future__ import annotations
import numpy as np

DS = 4  # downsample stride -> 16x16


def frame_features(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame)
    h, w = f.shape
    flat = f.flatten()
    bg = np.bincount(flat, minlength=16).argmax()
    obj = []
    n_colors = 0
    for c in range(1, 16):
        ys, xs = np.where(f == c)
        if len(xs) == 0 or c == bg:
            obj += [0.0, 0.0, 0.0, 0.0, 0.0]
            continue
        n_colors += 1
        obj += [len(xs) / (h * w), xs.mean() / w, ys.mean() / h,
                (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h]
    obj.append(n_colors / 15.0)
    coarse = (f[::DS, ::DS].astype(np.float32) / 15.0).flatten()
    return np.concatenate([np.array(obj, dtype=np.float32), coarse])
