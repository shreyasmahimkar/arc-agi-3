"""v17 TRM — a Tiny Recursive Model (our own, in the spirit of Jolicoeur-
Martineau 2025's TRM and HRM): a single small core applied RECURSIVELY to
refine a latent belief about the current state, then read out a policy prior
and a value. This is the "policy/value" half the BFSLLM note calls for — the
ExIt apprentice that biases search toward expert-like actions.

Architecture (pure numpy, BPTT over T refinement steps):
  x  = object_features(frame)                      (76-d, object-centric)
  z_0 = 0
  for t in 1..T:  z_t = tanh(Wzz z_{t-1} + Wxz x + bz)   # SHARED weights
  policy = softmax(Wp z_T + bp)   over {move1,move2,move3,move4, CLICK}
  value  = sigmoid(Wv z_T + bv)   ~ gamma^(steps-to-goal)

The recursion is the whole point: the same tiny weight set is iterated to
"think" longer about a state without adding parameters — TRM's core idea.
Deep supervision: heads are supervised at every step t (sum of losses), which
stabilises the recursion and is faithful to TRM/HRM training.

Trained on: expert actions (policy, cross-entropy) + cost-to-go (value, MSE)
from data.build_datasets. Used in search as policy_fn (strategy='puct') to
prune/reorder the branch set from ~12 actions to the top-k.
"""
from __future__ import annotations
import numpy as np
from vlog import get_logger

CLASSES = [1, 2, 3, 4, 6]        # 6 == CLICK (any centroid)
GAMMA = 0.9


class TRM:
    def __init__(self, in_dim=76, hidden=48, T=3, seed=1):
        self.T = T
        rng = np.random.RandomState(seed)
        h = hidden
        self.Wxz = (rng.randn(in_dim, h) * np.sqrt(1.0/in_dim)).astype(np.float32)
        self.Wzz = (rng.randn(h, h) * 0.1).astype(np.float32)
        self.bz = np.zeros(h, dtype=np.float32)
        self.Wp = (rng.randn(h, len(CLASSES)) * np.sqrt(1.0/h)).astype(np.float32)
        self.bp = np.zeros(len(CLASSES), dtype=np.float32)
        self.Wv = (rng.randn(h, 1) * np.sqrt(1.0/h)).astype(np.float32)
        self.bv = np.zeros(1, dtype=np.float32)
        self.x_mean = None; self.x_std = None
        self._adam = {}

    def _norm(self, X):
        return X if self.x_mean is None else (X - self.x_mean) / self.x_std

    def _recur(self, X):
        """Return list of z_t (t=1..T). X: (n, in_dim)."""
        n = X.shape[0]
        z = np.zeros((n, self.Wzz.shape[0]), dtype=np.float32)
        zs, pres = [], []
        for t in range(self.T):
            pre = X @ self.Wxz + z @ self.Wzz + self.bz
            z = np.tanh(pre)
            zs.append(z); pres.append(pre)
        return zs, pres

    def forward(self, frame_objs):
        X = self._norm(np.atleast_2d(frame_objs).astype(np.float32))
        zs, _ = self._recur(X)
        zT = zs[-1]
        logits = zT @ self.Wp + self.bp
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        v = 1.0/(1.0+np.exp(-(zT @ self.Wv + self.bv)))
        return p, v[:, 0]

    def policy_fn(self, frame):
        """Adapter for search.candidate_actions: frame -> {action_key: prob}."""
        import engine as E
        obj = E.object_features(frame)
        p, _ = self.forward(obj)
        p = p[0]
        out = {}
        for i, cls in enumerate(CLASSES):
            if cls == 6:
                for (a, d) in E.dynamic_clicks(frame, limit=10):
                    out[(6, d["x"], d["y"])] = float(p[i])
            else:
                out[cls] = float(p[i])
        return out

    def fit(self, X, aclasses, costs, epochs=300, lr=3e-3, bsz=64,
            iter_tag="v17", l2=1e-5):
        lg = get_logger("trm", iter_tag)
        X = np.asarray(X, np.float32)
        self.x_mean = X.mean(0); self.x_std = X.std(0) + 1e-6
        Xn = (X - self.x_mean) / self.x_std
        y_pol = np.asarray(aclasses, np.int64)
        v_tgt = (GAMMA ** np.asarray(costs, np.float32)).astype(np.float32)
        n = len(Xn)
        valid_pol = (y_pol >= 0)
        lg.info(f"[FIT] TRM n={n} T={self.T} hidden={self.Wzz.shape[0]} "
                f"epochs={epochs} pol_labeled={int(valid_pol.sum())}")
        for ep in range(epochs):
            idx = np.random.permutation(n)
            for s in range(0, n, bsz):
                bi = idx[s:s+bsz]
                self._step(Xn[bi], y_pol[bi], v_tgt[bi], lr, l2)
            if ep % max(1, epochs//8) == 0 or ep == epochs-1:
                p, v = self.forward_raw(Xn)
                vm = valid_pol
                acc = float((p[vm].argmax(1) == y_pol[vm]).mean()) if vm.any() else 0.0
                vmse = float(((v - v_tgt)**2).mean())
                lg.info(f"[FIT] ep={ep:3d} policy_acc={acc:.3f} value_mse={vmse:.4f}")
        return self

    def forward_raw(self, Xn):
        zs, _ = self._recur(Xn)
        zT = zs[-1]
        logits = zT @ self.Wp + self.bp
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits); p /= p.sum(1, keepdims=True)
        v = 1.0/(1.0+np.exp(-(zT @ self.Wv + self.bv)))[:, 0]
        return p, v

    def _step(self, Xn, y_pol, v_tgt, lr, l2):
        """BPTT over T steps with deep supervision at each step."""
        n = Xn.shape[0]
        zs, pres = self._recur(Xn)
        gWxz = np.zeros_like(self.Wxz); gWzz = np.zeros_like(self.Wzz)
        gbz = np.zeros_like(self.bz)
        gWp = np.zeros_like(self.Wp); gbp = np.zeros_like(self.bp)
        gWv = np.zeros_like(self.Wv); gbv = np.zeros_like(self.bv)
        dz_next = np.zeros_like(zs[0])
        valid = (y_pol >= 0)
        for t in reversed(range(self.T)):
            zt = zs[t]
            # heads (deep supervision every step)
            logits = zt @ self.Wp + self.bp
            logits -= logits.max(1, keepdims=True)
            p = np.exp(logits); p /= p.sum(1, keepdims=True)
            dlog = p.copy()
            dlog[np.arange(n), np.clip(y_pol, 0, len(CLASSES)-1)] -= 1.0
            dlog[~valid] = 0.0
            denom = max(1, int(valid.sum()))
            dlog /= denom
            gWp += zt.T @ dlog + l2*self.Wp; gbp += dlog.sum(0)
            v = 1.0/(1.0+np.exp(-(zt @ self.Wv + self.bv)))
            dv = (2.0/n) * (v - v_tgt[:, None]) * v * (1-v)
            gWv += zt.T @ dv + l2*self.Wv; gbv += dv.sum(0)
            dz = dlog @ self.Wp.T + dv @ self.Wv.T + dz_next
            dpre = dz * (1 - zt*zt)            # tanh'
            gWxz += Xn.T @ dpre
            gbz += dpre.sum(0)
            zprev = zs[t-1] if t > 0 else np.zeros_like(zt)
            gWzz += zprev.T @ dpre
            dz_next = dpre @ self.Wzz.T
        gWxz += l2*self.Wxz; gWzz += l2*self.Wzz
        self._adam_step(lr, dict(Wxz=gWxz, Wzz=gWzz, bz=gbz, Wp=gWp, bp=gbp, Wv=gWv, bv=gbv))

    def _adam_step(self, lr, grads, b1=0.9, b2=0.999, eps=1e-8):
        for name, g in grads.items():
            p = getattr(self, name)
            st = self._adam.setdefault(name, {"m": np.zeros_like(p), "v": np.zeros_like(p), "t": 0})
            st["t"] += 1
            st["m"] = b1*st["m"] + (1-b1)*g
            st["v"] = b2*st["v"] + (1-b2)*(g*g)
            mh = st["m"]/(1-b1**st["t"]); vh = st["v"]/(1-b2**st["t"])
            setattr(self, name, p - lr*mh/(np.sqrt(vh)+eps))

    def save(self, path):
        np.savez(path, **{k: getattr(self, k) for k in
                          ("Wxz", "Wzz", "bz", "Wp", "bp", "Wv", "bv", "x_mean", "x_std")},
                 T=self.T)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(T=int(d["T"]))
        for k in ("Wxz", "Wzz", "bz", "Wp", "bp", "Wv", "bv", "x_mean", "x_std"):
            setattr(m, k, d[k])
        return m
