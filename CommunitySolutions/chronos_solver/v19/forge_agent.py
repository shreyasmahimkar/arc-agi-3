"""FORGE v19 — black-box agent (StochasticGoose + Blind Squirrel fusion).

This is the v20 architecture (the user's rewrite) adapted to run LOCALLY against
v18's blackbox_env (which exposes exactly the API-only surface the real Kaggle
competition serves). It fuses the two ARC-AGI-3 preview winners:
  * ChangeNet  — a CNN that predicts which action will change the frame
                 (StochasticGoose, 1st place 12.58%): used as a prior to rank
                 untried actions and as the sampling dist when exhausted.
  * Transition graph + frontier exploration (Blind Squirrel, 2nd place 6.71%):
                 keyed by frame hash; prefer untried actions; when a state is
                 exhausted, plan a path through known edges to the nearest state
                 with untried actions. "Never knowingly repeat a transition"
                 (RHAE: wasted actions are the enemy).

I/O adapted: consumes an `Obs` (frame ndarray, state str, levels_completed,
available_actions tuple); emits (action_id, data) for blackbox_env.step.
NO white-box: frame-only, no engine internals, no game-source instantiation.
"""
from __future__ import annotations
import hashlib, random, time
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# self-contained constants (no blackbox_env dep — Kaggle-safe)
MOVES = [1, 2, 3, 4]
RESET = 0
CLICK = 6
GRID = 64
N_SIMPLE = 5                # ACTION1..ACTION5 -> akey 0..4 ; action_id = akey+1
CLICK_BASE = N_SIMPLE       # click akey = CLICK_BASE + y*64 + x ; action_id 6
FEAT_CH = 21
MACRO_CAP = 40              # v17 borrow: a movement repeats until the frame stops
                            # changing (wall) — collapses a corridor into 1 edge.
                            # Capped so a counter/animation can't loop forever.


def featurize(frames: torch.Tensor) -> torch.Tensor:
    """(B,64,64) int64 -> (B,21,64,64). Identical at train and inference."""
    B = frames.shape[0]
    oh = F.one_hot(frames.clamp(0, 15), 16).permute(0, 3, 1, 2).float()
    cnt = oh.sum(dim=[2, 3])
    bg = cnt.argmax(dim=1)
    bg_mask = (frames == bg.view(B, 1, 1)).float().unsqueeze(1)
    mx = cnt.max(dim=1, keepdim=True)[0].clamp(min=1.0)
    rarity = (oh * (1.0 - cnt / mx).view(B, 16, 1, 1)).sum(1, keepdim=True)
    f = frames.unsqueeze(1).float()
    pad = F.pad(f, (1, 1, 1, 1), mode="replicate")
    edge = ((f != pad[:, :, :-2, 1:-1]) | (f != pad[:, :, 2:, 1:-1]) |
            (f != pad[:, :, 1:-1, :-2]) | (f != pad[:, :, 1:-1, 2:])).float()
    rp = torch.linspace(0, 1, GRID, device=frames.device).view(1, 1, GRID, 1).expand(B, 1, GRID, GRID)
    cp = torch.linspace(0, 1, GRID, device=frames.device).view(1, 1, 1, GRID).expand(B, 1, GRID, GRID)
    return torch.cat([oh, bg_mask, rarity, edge, rp, cp], dim=1)


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        h = F.relu(self.c1(x)); h = self.c2(h)
        return F.relu(x + h)


class ChangeNet(nn.Module):
    def __init__(self, in_ch=FEAT_CH):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.res1 = ResBlock(128); self.res2 = ResBlock(128)
        self.a_pool = nn.AdaptiveAvgPool2d(4)
        self.a_fc1 = nn.Linear(128 * 16, 256)
        self.a_fc2 = nn.Linear(256, N_SIMPLE)
        self.drop = nn.Dropout(0.1)
        self.c_dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 1, 1))

    def forward(self, x):
        f = self.res2(self.res1(self.stem(x)))
        a = self.a_pool(f).flatten(1)
        a_logits = self.a_fc2(self.drop(F.relu(self.a_fc1(a))))
        c_logits = self.c_dec(f).squeeze(1)
        return a_logits, c_logits


class Node:
    __slots__ = ("cands", "tried", "edges")

    def __init__(self, cands):
        self.cands = cands; self.tried = set(); self.edges = {}


class ForgeAgent:
    name = "forge"

    def __init__(self, seed: int = 0, device=None, weights=None, **kw):
        self.seed = seed
        self.weights_path = weights        # optional pretrained ChangeNet prior
        if device is not None:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            try:
                mps = torch.backends.mps.is_available()
            except Exception:
                mps = False
            self.device = torch.device("mps" if mps else "cpu")

    # one fresh agent per game (CNN learns this game's dynamics online)
    def reset(self, game_id: str):
        random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed)
        self.net = ChangeNet().to(self.device)
        if self.weights_path:
            try:
                sd = torch.load(self.weights_path, map_location=self.device, weights_only=True)
                self.net.load_state_dict(sd)
            except Exception:
                pass
        self.opt = optim.Adam(self.net.parameters(), lr=3e-4)
        self.buf = deque(maxlen=60000)
        self.buf_seen = set()
        self.bsz = 64; self.train_every = 8; self._acts = 0
        self.level = -1
        self.graph = {}; self.visited = set(); self.plan = deque()
        self.no_change_streak = 0; self.sweeps_cleared = 0
        self.prev_frame = None; self.prev_hash = None; self.prev_akey = None
        self.pending_expected = None; self.pending_src = None; self.pending_akey_done = None
        self._fallback_i = 0
        # macro-action state (movement repeated until the frame stops changing)
        self._macro = False; self._macro_act_id = None
        self._macro_prev_hash = None; self._macro_steps = 0; self._macro_start_levels = 0
        # online transient-pixel mask (counters/timers) — frozen after calibration
        self._mask = None; self._mask_frozen = False
        self._px_changes = np.zeros((GRID, GRID), dtype=np.int32); self._px_total = 0

    # ---- observation helpers (Obs from blackbox_env) ----
    @staticmethod
    def _raw(obs):
        return np.asarray(obs.frame, dtype=np.uint8)

    def _hash(self, frame):
        m = self._mask
        if m is not None:
            frame = frame.copy(); frame[m] = 0
        return hashlib.md5(frame.tobytes()).hexdigest()[:16]

    @staticmethod
    def _norm_avail(avail):
        out = set(int(a) for a in (avail or []))
        return out or {1, 2, 3, 4, 5, 6}

    def _click_candidates(self, frame, cap=48):
        cnt = np.bincount(frame.ravel(), minlength=16)
        bg = int(cnt.argmax()); cands = []
        for c in range(16):
            if c == bg or cnt[c] == 0 or cnt[c] > 3200:
                continue
            ys, xs = np.where(frame == c)
            cy, cx = int(np.median(ys)), int(np.median(xs))
            d = np.abs(ys - cy) + np.abs(xs - cx); j = int(d.argmin())
            cands.append((int(ys[j]), int(xs[j])))
            step = max(1, len(ys) // 6)
            for k in range(0, len(ys), step):
                cands.append((int(ys[k]), int(xs[k])))
        seen, keys = set(), []
        for (y, x) in cands:
            key = CLICK_BASE + y * GRID + x
            if key not in seen:
                seen.add(key); keys.append(key)
            if len(keys) >= cap:
                break
        return keys

    def _candidates(self, frame, avail):
        av = self._norm_avail(avail)
        keys = [i - 1 for i in (1, 2, 3, 4, 5) if i in av]
        if 6 in av:
            keys += self._click_candidates(frame)
        return keys

    @torch.no_grad()
    def _score(self, frame, keys):
        t = torch.from_numpy(frame.astype(np.int64)).unsqueeze(0).to(self.device)
        a_logits, c_logits = self.net(featurize(t))
        a_p = torch.sigmoid(a_logits[0]).cpu().numpy()
        c_p = torch.sigmoid(c_logits[0]).cpu().numpy()
        out = {}
        for k in keys:
            if k < N_SIMPLE:
                out[k] = float(a_p[k])
            else:
                ci = k - CLICK_BASE
                out[k] = float(c_p[ci // GRID, ci % GRID])
        return out

    def _record(self, prev_frame, prev_hash, akey, curr_hash):
        changed = curr_hash != prev_hash
        novel = changed and (curr_hash not in self.visited)
        if changed:
            self.visited.add(curr_hash); self.no_change_streak = 0
        else:
            self.no_change_streak += 1
        target = 1.0 if novel else (0.6 if changed else 0.0)
        dedup = prev_hash + ":" + str(akey)
        if dedup not in self.buf_seen:
            self.buf_seen.add(dedup)
            self.buf.append((prev_frame.copy(), akey, target))
        node = self.graph.get(prev_hash)
        if node is not None:
            node.tried.add(akey); node.edges[akey] = curr_hash

    def _train(self):
        if len(self.buf) < self.bsz:
            return
        idx = np.random.choice(len(self.buf), self.bsz, replace=False)
        frames = np.stack([self.buf[i][0] for i in idx]).astype(np.int64)
        keys = [self.buf[i][1] for i in idx]
        targs = torch.tensor([self.buf[i][2] for i in idx], dtype=torch.float32, device=self.device)
        x = featurize(torch.from_numpy(frames).to(self.device))
        a_logits, c_logits = self.net(x)
        sel = torch.empty(self.bsz, device=self.device)
        for i, k in enumerate(keys):
            if k < N_SIMPLE:
                sel[i] = a_logits[i, k]
            else:
                ci = k - CLICK_BASE; sel[i] = c_logits[i, ci // GRID, ci % GRID]
        loss = F.binary_cross_entropy_with_logits(sel, targs)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def _plan_to_frontier(self, start_hash, limit=2500):
        parents = {start_hash: None}; q = deque([start_hash]); goal = None; n = 0
        while q and n < limit:
            h = q.popleft(); n += 1
            node = self.graph.get(h)
            if node is None:
                continue
            if h != start_hash and any(k not in node.tried for k in node.cands):
                goal = h; break
            for akey, h2 in node.edges.items():
                if h2 == h or h2 in parents:
                    continue
                parents[h2] = (h, akey); q.append(h2)
        if goal is None:
            return None
        path = []; h = goal
        while parents[h] is not None:
            ph, akey = parents[h]; path.append((akey, h)); h = ph
        path.reverse(); return deque(path)

    # ---- action key -> (action_id, data) for blackbox_env ----
    @staticmethod
    def _to_action(akey):
        if akey < N_SIMPLE:
            return akey + 1, None
        ci = akey - CLICK_BASE
        return 6, {"x": int(ci % GRID), "y": int(ci // GRID), "game_id": "forge"}

    def _safe_fallback(self):
        self._fallback_i = (self._fallback_i + 1) % N_SIMPLE
        self.prev_hash = None
        return self._to_action(self._fallback_i)

    def _set_pending(self, frame, h, akey):
        self.prev_frame = frame; self.prev_hash = h; self.prev_akey = akey; self._acts += 1

    def _maybe_train(self):
        if self._acts % self.train_every == 0:
            self._train()

    # ---- the policy ----
    def act(self, obs):
        try:
            return self._choose(obs)
        except Exception:
            self.plan.clear(); self.prev_hash = None
            return self._safe_fallback()

    def _emit(self, akey, frame, h, lvl):
        """Issue an action. Movement (ACTION1-4) becomes a MACRO: the same move
        will be repeated on following turns until the frame stops changing, then
        the whole corridor is recorded as ONE edge."""
        self._set_pending(frame, h, akey)
        if akey < 4:                       # ACTION1-4 -> macro
            self._macro = True
            self._macro_act_id = akey + 1
            self._macro_prev_hash = h
            self._macro_steps = 0
            self._macro_start_levels = lvl
        else:
            self._macro = False
        return self._to_action(akey)

    def _choose(self, obs):
        lvl = obs.levels_completed
        if lvl != self.level:
            self.level = lvl
            self.graph.clear(); self.visited.clear(); self.plan.clear()
            self.prev_frame = self.prev_hash = self.prev_akey = None
            self.pending_expected = self.pending_src = self.pending_akey_done = None
            self.no_change_streak = 0; self.sweeps_cleared = 0
            self._macro = False
            self._mask = None; self._mask_frozen = False
            self._px_changes[:] = 0; self._px_total = 0
            for _ in range(min(10, len(self.buf) // self.bsz)):
                self._train()

        if obs.state in ("NOT_PLAYED", "GAME_OVER"):
            self.plan.clear(); self.prev_hash = None; self.pending_expected = None
            self._macro = False
            return 0, None       # RESET

        frame = self._raw(obs); h = self._hash(frame)

        # ---- macro in progress: repeat the move until the frame stops changing
        if self._macro:
            terminating = (h == self._macro_prev_hash
                           or lvl != self._macro_start_levels
                           or obs.state != "NOT_FINISHED"
                           or self._macro_steps >= MACRO_CAP)
            if not terminating:
                self._macro_prev_hash = h
                self._macro_steps += 1
                return self._macro_act_id, None
            self._macro = False        # terminates here; close the edge below

        # ---- online transient-pixel detection (v17/v18 borrow) ----
        # pixels that change on (almost) every transition are counters/timers;
        # mask them from the hash or every frame looks novel and the graph
        # explodes (which also defeats the macro wall-detection).
        if self.prev_frame is not None and self.prev_akey is not None and not self._mask_frozen:
            self._px_changes += (self.prev_frame != frame)
            self._px_total += 1
            if self._px_total >= 12:
                rate = self._px_changes / self._px_total
                m = rate > 0.85
                self._mask = m if m.any() else None
                self._mask_frozen = True
                self.graph.clear(); self.visited.clear(); self.plan.clear()
                self.prev_hash = None            # boundary transition; don't record
                h = self._hash(frame)            # re-key this state with the mask

        # ---- close the pending transition (single step OR collapsed macro) ----
        if self.prev_hash is not None and self.prev_akey is not None:
            self._record(self.prev_frame, self.prev_hash, self.prev_akey, h)

        node = self.graph.get(h)
        if node is None:
            node = Node(self._candidates(frame, obs.available_actions))
            self.graph[h] = node; self.visited.add(h)

        if self.pending_expected is not None:
            if h != self.pending_expected:
                src = self.graph.get(self.pending_src) if self.pending_src else None
                if src is not None and self.pending_akey_done in src.edges:
                    del src.edges[self.pending_akey_done]
                self.plan.clear()
            self.pending_expected = self.pending_src = self.pending_akey_done = None

        if self.plan:
            akey, expected = self.plan.popleft()
            self.pending_expected = expected; self.pending_src = h; self.pending_akey_done = akey
            return self._emit(akey, frame, h, lvl)

        untried = [k for k in node.cands if k not in node.tried]
        if untried:
            scores = self._score(frame, untried)
            akey = random.choice(untried) if random.random() < 0.05 else max(untried, key=lambda k: scores[k])
            self._maybe_train()
            return self._emit(akey, frame, h, lvl)

        path = self._plan_to_frontier(h)
        if path:
            self.plan = path
            akey, expected = self.plan.popleft()
            self.pending_expected = expected; self.pending_src = h; self.pending_akey_done = akey
            return self._emit(akey, frame, h, lvl)

        if self.no_change_streak > 40 and self.sweeps_cleared < 8:
            self.sweeps_cleared += 1
            for nd in self.graph.values():
                nd.tried.clear()
            self.no_change_streak = 0

        keys = node.cands or list(range(N_SIMPLE))
        scores = self._score(frame, keys)
        ks = list(scores.keys())
        ps = np.array([max(scores[k], 1e-4) for k in ks]); ps = ps / ps.sum()
        akey = int(np.random.choice(ks, p=ps))
        self._maybe_train()
        return self._emit(akey, frame, h, lvl)
