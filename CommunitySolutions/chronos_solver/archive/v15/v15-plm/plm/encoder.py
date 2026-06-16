"""v15 PLM — spatial encoder: BFS object channels + CNN + VQ tokenizer.

UNCHANGED from v14 — and that is deliberate: v14's tokenizer trained to
0.9862 pixel accuracy and its weights are reused directly (copy v14's
plm_weights.pt into the v15 dir; the wm phase resumes the tokenizer and
retrains only belief + world_model with the new architecture).

Diagnosed but deferred: the codebook is degenerate (700/1024 codes are
near-duplicates, only ~35 in active use). The ids are STABLE (zero flips
under 1e-3 perturbation, measured), so this is waste, not a bug —
dead-code revival is queued for the next tokenizer retrain, not this one.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# scipy is optional but makes featurization ~100x faster — the pure-python
# fallback below was the training-throughput bottleneck on GPU boxes.
try:
    from scipy import ndimage as _ndi
except Exception:
    _ndi = None


# ---------- programmatic object channel (numpy BFS — the v13 trick) ----------

def object_channels(frame: np.ndarray) -> np.ndarray:
    """Connected components per color. Returns float32 (2, H, W):
    channel 0 = normalized component id, channel 1 = normalized component size.
    Gives the CNN object-permanence for free."""
    if _ndi is not None:
        # vectorized path: scipy labels each color's components in C
        comp = np.zeros(frame.shape, np.int32)
        size = np.zeros(frame.shape, np.float32)
        nxt = 0
        for c in np.unique(frame):
            lab, n = _ndi.label(frame == c)
            if n == 0:
                continue
            m = lab > 0
            comp[m] = lab[m] + nxt
            counts = np.bincount(lab.ravel())
            size[m] = counts[lab[m]]
            nxt += n
        return np.stack([comp.astype(np.float32) / max(nxt, 1),
                         size / frame.size])
    # pure-python fallback (slow; fine for inference, not for training)
    H, W = frame.shape
    comp = np.zeros((H, W), np.int32)
    size = np.zeros((H, W), np.float32)
    cur = 0
    for y in range(H):
        for x in range(W):
            if comp[y, x]:
                continue
            cur += 1
            c = frame[y, x]
            stack = [(y, x)]
            cells = []
            comp[y, x] = cur
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy-1, cx), (cy+1, cx), (cy, cx-1), (cy, cx+1)):
                    if 0 <= ny < H and 0 <= nx < W and not comp[ny, nx] \
                            and frame[ny, nx] == c:
                        comp[ny, nx] = cur
                        stack.append((ny, nx))
            n = len(cells)
            for cy, cx in cells:
                size[cy, cx] = n
    out = np.stack([comp.astype(np.float32) / max(cur, 1),
                    size / (H * W)])
    return out


def frame_to_tensor(frame: np.ndarray, n_colors: int = 16) -> torch.Tensor:
    """(H,W) int grid -> (n_colors+2, H, W) float tensor."""
    oh = np.eye(n_colors, dtype=np.float32)[frame.clip(0, n_colors - 1)]
    oh = oh.transpose(2, 0, 1)                       # (C,H,W)
    obj = object_channels(frame)                     # (2,H,W)
    return torch.from_numpy(np.concatenate([oh, obj], 0))


# ---------- CNN encoder / decoder ----------

class GridEncoder(nn.Module):
    """(B, n_colors+2, 64, 64) -> (B, code_dim, 8, 8)"""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.enc_ch
        self.net = nn.Sequential(
            nn.Conv2d(cfg.n_colors + 2, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GELU(),      # 32
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, stride=2, padding=1), nn.GELU(),      # 16
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.Conv2d(c, cfg.code_dim, 3, stride=2, padding=1),      # 8
        )

    def forward(self, x):
        return self.net(x)


class GridDecoder(nn.Module):
    """(B, code_dim, 8, 8) -> (B, n_colors, 64, 64) logits (recon loss only)."""

    def __init__(self, cfg):
        super().__init__()
        c = cfg.enc_ch
        self.net = nn.Sequential(
            nn.ConvTranspose2d(cfg.code_dim, c, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(),
            nn.ConvTranspose2d(c, c, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(c, cfg.n_colors, 3, padding=1),
        )

    def forward(self, z):
        return self.net(z)


# ---------- EMA vector quantizer ----------

class VectorQuantizer(nn.Module):
    """Straight-through VQ with EMA codebook updates (VQ-VAE-2 style)."""

    def __init__(self, cfg, decay=0.99, eps=1e-5):
        super().__init__()
        self.K, self.D = cfg.codebook, cfg.code_dim
        self.decay, self.eps = decay, eps
        embed = torch.randn(self.K, self.D) * 0.1
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, z):                            # z: (B, D, 8, 8)
        B, D, H, W = z.shape
        flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (BHW, D)
        d = (flat.pow(2).sum(1, keepdim=True)
             - 2 * flat @ self.embed.t()
             + self.embed.pow(2).sum(1))
        idx = d.argmin(1)                            # (BHW,)
        q = self.embed[idx].view(B, H, W, D).permute(0, 3, 1, 2)
        if self.training:
            # EMA codebook update — MUST be outside autograd. Without the
            # no_grad + detach, `flat`'s gradient graph gets entangled with
            # the persistent buffers and every step's activations are
            # retained: observed leak of ~65MB/step until a 140GB H200
            # filled (139.08 GiB "allocated by PyTorch"). detach().float()
            # also keeps EMA statistics in fp32 under bf16 autocast.
            with torch.no_grad():
                flat_d = flat.detach().float()
                onehot = F.one_hot(idx, self.K).float()
                self.cluster_size.mul_(self.decay).add_(onehot.sum(0), alpha=1 - self.decay)
                self.embed_avg.mul_(self.decay).add_(onehot.t() @ flat_d, alpha=1 - self.decay)
                n = self.cluster_size.sum()
                cs = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
                self.embed.copy_(self.embed_avg / cs.unsqueeze(1))
        commit = F.mse_loss(z, q.detach())
        q = z + (q - z).detach()                     # straight-through
        return q, idx.view(B, H, W), commit


class Tokenizer(nn.Module):
    """encoder + VQ + decoder bundle."""

    def __init__(self, cfg):
        super().__init__()
        self.enc = GridEncoder(cfg)
        self.vq = VectorQuantizer(cfg)
        self.dec = GridDecoder(cfg)

    def encode(self, x):
        q, idx, _ = self.vq(self.enc(x))
        return q, idx                                # idx: (B, 8, 8) token ids

    def forward(self, x, target_grid):
        q, idx, commit = self.vq(self.enc(x))
        logits = self.dec(q)                         # (B, n_colors, 64, 64)
        recon = F.cross_entropy(logits, target_grid)
        return recon + 0.25 * commit, recon, idx
