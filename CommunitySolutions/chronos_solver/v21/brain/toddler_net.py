# =====================================================================
# v21 brain/toddler_net.py — the NEURAL toddler (Epic C3 / R9+R11).
#
# StochasticGoose-style frame-change predictor: a tiny CNN that, given a frame,
# predicts for each action whether it will CHANGE the frame and whether it WINS.
# That prediction is the intuitive prior that ORDERS actions for Go-Explore /
# search — turning blind exploration into guided exploration (the ls20 L5 lever).
#
# Trains on the Mac's OWN GPU — Apple Silicon via PyTorch MPS (device="mps") or
# CPU fallback. These nets are tiny (a 4-layer conv; ~10^5-10^6 params), so an
# M1/M2/M3 Pro trains them in minutes. NO NVIDIA/CUDA needed. Data = (frame,
# action -> changed/won) tuples harvested from the cadence's own rollouts
# (blackboard/transitions) — StochasticGoose's exact supervised signal.
#
# Pluggable + guarded like llm_backend: if torch is absent the class degrades to
# the frequency/effectiveness prior (blackboard.action_order), so the cascade
# never depends on it. Pure/offline parts (data IO, device pick, fallback order)
# are covered by test_toddler.py; training itself runs on the Mac.
#
# Env: V21_TODDLER_NET=1 to use the net; V21_TODDLER_DEVICE to force cpu/mps.
# Weights persist at brain/toddler/<gid>.pt ; samples at brain/toddler/<gid>.jsonl
# =====================================================================
import os, json, glob, logging

logger = logging.getLogger("v21.toddler")
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.environ.get("V21_TODDLER_DIR", os.path.join(_HERE, "toddler"))
ALL_ACTIONS = [1, 2, 3, 4, 5, 6, 7]
GRID = 64                      # frames are padded/cropped to GRID x GRID
N_COLORS = 16


def pick_device():
    """Apple-Silicon-first device selection. MPS (Mac GPU) > CUDA > CPU."""
    forced = os.environ.get("V21_TODDLER_DEVICE")
    if forced:
        return forced
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def torch_available():
    try:
        import torch  # noqa
        return True
    except Exception:
        return False


# ---- data harvest (pure, offline-testable) -------------------------------------
def append_samples(game, samples):
    """samples: list of {"frame": 2D int list, "action": int, "changed": bool,
    "won": bool}. Appended as jsonl so training data compounds across the loop."""
    if not samples:
        return
    os.makedirs(_DIR, exist_ok=True)
    p = os.path.join(_DIR, f"{str(game).split('-')[0]}.jsonl")
    with open(p, "a") as f:
        for s in samples:
            f.write(json.dumps({"frame": s["frame"], "action": int(s["action"]),
                                "changed": bool(s.get("changed")),
                                "won": bool(s.get("won"))}) + "\n")


def load_samples(game, max_n=50000):
    p = os.path.join(_DIR, f"{str(game).split('-')[0]}.jsonl")
    out = []
    if os.path.exists(p):
        for line in open(p):
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out[-max_n:]


def _encode(frame):
    """2D int frame -> GRID x GRID int8 numpy (cropped/padded, clamped to colors)."""
    import numpy as np
    f = np.asarray(frame)
    if f.ndim != 2:
        f = np.zeros((GRID, GRID), dtype=np.int64)
    f = np.clip(f, 0, N_COLORS - 1).astype(np.int64)
    out = np.zeros((GRID, GRID), dtype=np.int64)
    h, w = min(f.shape[0], GRID), min(f.shape[1], GRID)
    out[:h, :w] = f[:h, :w]
    return out


class ToddlerNet:
    """The neural intuitive prior. `.order_actions(frame, game)` is the fixed
    interface the searches call (same as IntuitionPrior); falls back to a supplied
    frequency order (or canonical) when torch/weights are unavailable."""

    def __init__(self, game, fallback_order=None):
        self.game = str(game).split("-")[0]
        self.fallback = list(fallback_order or ALL_ACTIONS)
        self.device = pick_device()
        self._model = None

    def available(self):
        return torch_available()

    # -- model ---------------------------------------------------------------------
    def _build(self):
        import torch, torch.nn as nn
        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                s.emb = nn.Embedding(N_COLORS, 8)
                s.conv = nn.Sequential(
                    nn.Conv2d(8, 32, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1))
                s.head_change = nn.Linear(64, len(ALL_ACTIONS))
                s.head_win = nn.Linear(64, len(ALL_ACTIONS))
            def forward(s, x):                       # x: (B,H,W) int
                h = s.emb(x).permute(0, 3, 1, 2)      # (B,8,H,W)
                h = s.conv(h).flatten(1)              # (B,64)
                return s.head_change(h), s.head_win(h)
        return Net()

    def _weights_path(self):
        return os.path.join(_DIR, f"{self.game}.pt")

    def load(self):
        if self._model is not None:
            return True
        if not torch_available():
            return False
        import torch
        p = self._weights_path()
        if self._model is None and os.path.exists(p):
            self._model = self._build().to(self.device)
            self._model.load_state_dict(torch.load(p, map_location=self.device))
            self._model.eval()
        return self._model is not None

    # -- train (runs on the Mac GPU) -----------------------------------------------
    def train(self, epochs=8, lr=1e-3, batch=128, min_samples=200):
        """Train the frame-change/win predictor on harvested rollouts. Returns a
        short status string. No-op (returns a reason) if torch or data missing."""
        if not torch_available():
            return "torch unavailable — toddler stays on the frequency prior"
        import torch, torch.nn as nn, numpy as np
        rows = load_samples(self.game)
        if len(rows) < min_samples:
            return f"only {len(rows)} samples (< {min_samples}) — skip train"
        aidx = {a: i for i, a in enumerate(ALL_ACTIONS)}
        X = np.stack([_encode(r["frame"]) for r in rows])
        ai = np.array([aidx.get(int(r["action"]), 0) for r in rows])
        yc = np.array([1.0 if r.get("changed") else 0.0 for r in rows], dtype="float32")
        yw = np.array([1.0 if r.get("won") else 0.0 for r in rows], dtype="float32")
        dev = self.device
        Xt = torch.tensor(X, dtype=torch.long, device=dev)
        model = self._build().to(dev); model.train()
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        bce = nn.BCEWithLogitsLoss()
        n = len(rows)
        for ep in range(epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                pc, pw = model(Xt[idx])
                rows_i = torch.arange(len(idx), device=dev)
                a = torch.tensor(ai, device=dev)[idx]
                lc = bce(pc[rows_i, a], torch.tensor(yc, device=dev)[idx])
                lw = bce(pw[rows_i, a], torch.tensor(yw, device=dev)[idx])
                opt.zero_grad(); (lc + lw).backward(); opt.step()
        os.makedirs(_DIR, exist_ok=True)
        model.eval(); torch.save(model.state_dict(), self._weights_path())
        self._model = model
        return f"trained on {n} samples, device={dev}, epochs={epochs}"

    # -- inference: the intuition the searches consume -----------------------------
    def order_actions(self, frame=None, game=None):
        if frame is None or not self.load():
            return list(self.fallback)
        try:
            import torch, numpy as np
            x = torch.tensor(_encode(frame)[None], dtype=torch.long, device=self.device)
            with torch.no_grad():
                pc, pw = self._model(x)
            score = (2.0 * torch.sigmoid(pw) + torch.sigmoid(pc))[0].cpu().numpy()
            return [ALL_ACTIONS[i] for i in np.argsort(-score)]
        except Exception as e:
            logger.debug("toddler infer fail: %s", e)
            return list(self.fallback)
