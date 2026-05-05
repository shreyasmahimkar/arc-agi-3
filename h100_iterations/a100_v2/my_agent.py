# =====================================================================
# FORGE v21 — A100 Optimized + Agentic Swarm & Semantic Director
#
# Upgrades over v20:
# [A100 Opts] TF32 Tensor Cores, PyTorch 2.0 compile, AMP.
# [Semantic Director] Converts grid to Scene Graphs. Uses spatial masking 
#                     to prune 4096 click space to localized targets.
# [Agentic Swarm] Runs threaded worker roles (Navigator, Painter, Chaos) 
#                 to brute-force distinct logical paths in parallel.
# [Go-Explore Blackboard] Teleports to semantically novel frontiers if stuck.
# =====================================================================
import copy
import glob
import hashlib
import importlib.util
import logging
import os
import random
import time
import traceback
import concurrent.futures
from collections import deque

import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ==================== A100 OPTIMIZATIONS ====================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== SEMANTIC DIRECTOR & HASHING ====================

class SemanticDirector:
    """Acts as the offline Vision Model proxy to guide spatial attention and pruning."""
    def __init__(self, G=64):
        self.G = G

    def extract_scene_graph(self, frame, bg_color):
        """Translates raw pixels into a structural Scene Graph (objects/topology)."""
        objs = []
        for c in range(16):
            if c == bg_color: continue
            mask = (frame == c)
            if not np.any(mask): continue
            
            labels, num = scipy.ndimage.label(mask)
            for i in range(1, num + 1):
                ys, xs = np.where(labels == i)
                mass = len(ys)
                if mass < 1 or mass > 3500: continue
                objs.append({
                    'c': c, 'n': mass,
                    'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)),
                    'min_x': int(np.min(xs)), 'max_x': int(np.max(xs)),
                    'min_y': int(np.min(ys)), 'max_y': int(np.max(ys))
                })
        return sorted(objs, key=lambda o: (o['c'], -o['n']))

    def semantic_hash(self, objs):
        """Hashes the topology (positions quantized to 4x4 blocks) to ignore visual noise."""
        if not objs: return "empty"
        sig = "|".join([f"c{o['c']}n{o['n']}x{int(o['cx'])//4}y{int(o['cy'])//4}" for o in objs])
        return hashlib.md5(sig.encode()).hexdigest()[:16]

    def get_spotlight_mask(self, objs):
        """Creates a spatial heatmap (mask) to prune background clicks (-inf)."""
        mask = torch.zeros(self.G * self.G, dtype=torch.float32)
        if not objs: return mask 
        
        mask.fill_(float('-inf'))
        for o in objs:
            # Allow clicks directly on and adjacent to objects (padding=2)
            y1 = max(0, o['min_y'] - 2); y2 = min(self.G, o['max_y'] + 3)
            x1 = max(0, o['min_x'] - 2); x2 = min(self.G, o['max_x'] + 3)
            for y in range(y1, y2):
                for x in range(x1, x2):
                    mask[y * self.G + x] = 0.0 
        return mask

# ==================== SWARM ENGINE ====================

def find_game_source_and_class(game_id, arc_env=None):
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha(): cls_name = gid[0].upper() + gid[1:]
    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path; import re
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', candidate.read_text()[:2000])
                    if m: cls_name = m.group(1)
                    break
    if not src:
        import re
        for pattern in [f"/tmp/*/{gid}/*/{gid}.py", f"/kaggle/*/{gid}*/{gid}.py", f"**/game_sources/**/{gid}.py"]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', open(src).read()[:2000])
                if m: cls_name = m.group(1)
                break
    return src, cls_name

def run_worker(game_cls, level_idx, role, bg_color):
    """Isolated, multithread-safe worker for Agentic Swarm rollouts."""
    try:
        game = game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame: return None
        
        director = SemanticDirector()
        history = []
        f0 = np.array(r0.frame[-1])
        visited = {director.semantic_hash(director.extract_scene_graph(f0, bg_color))}
        
        for _ in range(40):
            avail = game._available_actions
            
            # Mixture of Experts routing
            if role == "navigator":
                choices = [a for a in avail if a <= 5]
                if not choices: break
                act = random.choice(choices)
                data = None
            elif role == "painter":
                if 6 not in avail: break
                act = 6
                objs = director.extract_scene_graph(np.array(r0.frame[-1]), bg_color)
                if objs and random.random() < 0.8:
                    o = random.choice(objs)
                    data = {'x': random.randint(o['min_x'], o['max_x']), 'y': random.randint(o['min_y'], o['max_y'])}
                else:
                    data = {'x': random.randint(0, 63), 'y': random.randint(0, 63)}
            else: # chaos
                act = random.choice(avail)
                data = {'x': random.randint(0, 63), 'y': random.randint(0, 63)} if act == 6 else None
                
            ai = ActionInput(id=GameAction.from_id(act), data=data) if data else ActionInput(id=GameAction.from_id(act))
            try: r0 = game.perform_action(ai, raw=True)
            except: break
            
            history.append((act, data))
            if r0.levels_completed > level_idx or game._current_level_index > level_idx:
                return {'status': 'solved', 'history': history}
            if not r0.frame: break
                
            f = np.array(r0.frame[-1])
            sg = director.extract_scene_graph(f, bg_color)
            h = director.semantic_hash(sg)
            
            if h in visited and role != "chaos": continue
            visited.add(h)
            
        return {'status': 'timeout', 'history': history, 'frontier_hash': h}
    except Exception as e:
        return None

# ==================== CNN ARCHITECTURE (A100 Scaled) ====================

class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        self.fc1 = nn.Linear(ch, max(ch//r, 4)); self.fc2 = nn.Linear(max(ch//r, 4), ch)
        self.sp = nn.Conv2d(2, 1, 7, padding=3)
    def forward(self, x):
        B,C,H,W = x.shape
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=[2,3])))))
        x = x * w.view(B,C,1,1)
        a = torch.sigmoid(self.sp(torch.cat([x.max(1,keepdim=True)[0], x.mean(1,keepdim=True)], 1)))
        return x * a

class ActionEffectAttention(nn.Module):
    def __init__(self, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        self.mem_dim = mem_dim
        self.diff_enc = nn.Sequential(nn.Conv2d(1, 8, 8, stride=8), nn.ReLU(),
                                      nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
                                      nn.Flatten(), nn.Linear(16*2*2, mem_dim))
        self.q_proj = nn.Linear(feat_dim, mem_dim)
        self.v_proj = nn.Linear(mem_dim + 1 + n_actions, n_actions)
        self.scale = mem_dim**0.5
    def forward(self, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B, M = mem_actions.shape
        if M == 0: return torch.zeros(B, 5, device=cnn_feat.device)
        keys = self.diff_enc(mem_diffs.reshape(B*M, 1, 64, 64)).reshape(B, M, self.mem_dim)
        q = self.q_proj(cnn_feat).unsqueeze(1)
        attn = F.softmax(torch.bmm(q, keys.transpose(1, 2)) / self.scale, dim=-1)
        act_oh = F.one_hot(mem_actions.clamp(0, 4), 5).float()
        vals = torch.cat([keys, mem_rewards.unsqueeze(-1), act_oh], dim=-1)
        ctx = torch.bmm(attn, vals).squeeze(1)
        return self.v_proj(ctx)

class ForgeNetV21(nn.Module):
    def __init__(self, in_ch=26, g=64):
        super().__init__()
        self.g = g
        # A100 allows for widened channel representations
        self.c1 = nn.Conv2d(in_ch, 64, 3, padding=1); self.c2 = nn.Conv2d(64, 128, 3, padding=1)
        self.c3 = nn.Conv2d(128, 256, 3, padding=1); self.c4 = nn.Conv2d(256, 256, 3, padding=1)
        self.attn = CBAM(256)
        
        self.ar = nn.Conv2d(256, 64, 1); self.ap = nn.MaxPool2d(4, 4)
        self.af = nn.Linear(64*16*16, 256); self.ah = nn.Linear(256, 5)
        self.dr = nn.Dropout(0.15)
        
        self.cc1 = nn.Conv2d(256, 128, 3, padding=1); self.cc2 = nn.Conv2d(128, 64, 3, padding=1)
        self.cc3 = nn.Conv2d(64, 32, 1); self.cc4 = nn.Conv2d(32, 1, 1)
        
        self.gp = nn.AdaptiveAvgPool2d(1); self.gf = nn.Linear(256, 64)
        self.aea = ActionEffectAttention(feat_dim=64, mem_dim=32, n_actions=5)
        
    def forward(self, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x = F.relu(self.c1(x)); x = F.relu(self.c2(x)); x = F.relu(self.c3(x)); f = F.relu(self.c4(x))
        f = self.attn(f)
        af = F.relu(self.ar(f)); af = self.ap(af).reshape(f.size(0), -1)
        al = self.ah(self.dr(F.relu(self.af(af))))
        cf = F.relu(self.cc1(f)); cf = F.relu(self.cc2(cf)); cf = F.relu(self.cc3(cf))
        cl = self.cc4(cf).reshape(f.size(0), -1)
        if mem_diffs is not None and mem_actions is not None:
            gf = self.gf(self.gp(f).reshape(f.size(0), -1))
            al = al + self.aea(gf, mem_diffs, mem_actions, mem_rewards)
        return torch.cat([al, cl], 1)

# ==================== MAIN AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(self.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        
        self.start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # A100 AMP Scaler
        self.scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        self.G = 64; self.IN = 26
        self.net = None; self.opt = None
        self.director = SemanticDirector(G=self.G)
        
        self.buf = deque(maxlen=200000)
        self.buf_h = set()
        self.bsz = 256 # Massive batch size for A100
        self.tfreq = 10
        
        self.pt = None; self.pai = None; self.pr = None; self.ph = None; self.pr_sg = None
        self.cl = -1; self.fhist = deque(maxlen=6); self.la = 0
        self.al = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5]
        self._wd = False; self._bg = 0
        self._aem_diffs = deque(maxlen=256); self._aem_actions = deque(maxlen=256); self._aem_rewards = deque(maxlen=256)
        
        self._eps = 0.15; self._eps_min = 0.03; self._eps_decay = 0.9995
        
        # Go-Explore Swarm & Blackboard States
        self.blackboard = {} 
        self.teleport_queue = deque() 
        self._visited_shashes = set()
        self._unproductive = 0
        
        self._game_cls = None
        self._swarm_solution = None
        self._swarm_step = 0

    def append_frame(self, f):
        self.frames.append(f)
        if len(self.frames) > self._MAX_FRAMES: self.frames = self.frames[-self._MAX_FRAMES:]
        if f.guid: self.guid = f.guid
        if hasattr(self, "recorder") and not self.is_playback:
            import json; self.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(self, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(self, fd): return np.array(fd.frame, dtype=np.int64)[-1]
    def _tensor(self, fd): return self._frame_to_tensor(self._raw(fd))

    def _init_swarm_env(self):
        src, cls = find_game_source_and_class(self.game_id, getattr(self, 'arc_env', None))
        if src:
            try:
                spec = importlib.util.spec_from_file_location('gmod', src)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self._game_cls = getattr(mod, cls)
                logger.info(f"Swarm: Engine Loaded -> {cls}")
            except Exception as e:
                logger.warning(f"Swarm Engine load failed: {e}")

    def _run_swarm(self, level_idx, bg_color):
        if not self._game_cls: return
        logger.info(f"Swarm: Deploying Parallel Agentic Workers for Level {level_idx}...")
        roles = ["navigator", "painter", "navigator", "chaos"]
        
        best_sol = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(run_worker, self._game_cls, level_idx, role, bg_color) for role in roles]
            for future in concurrent.futures.as_completed(futures):
                try:
                    res = future.result()
                    if not res: continue
                    if res['status'] == 'solved':
                        best_sol = res['history']
                        logger.info(f"Swarm: Level SOLVED offline!")
                        break
                    else:
                        h = res['frontier_hash']
                        if h not in self.blackboard:
                            self.blackboard[h] = res['history']
                except: pass
                
        if best_sol:
            self._swarm_solution = best_sol
            self._swarm_step = 0

    def _frame_to_tensor(self, frame, augment=False):
        oh = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)
        cnt = np.bincount(frame.flatten(), minlength=16)
        bg = int(cnt.argmax()); mx = max(cnt.max(), 1)
        
        if augment:
            perm = np.arange(16)
            non_bg = [c for c in range(16) if c != bg]
            np.random.shuffle(non_bg)
            idx = 0
            for c in range(16):
                if c != bg: perm[c] = non_bg[idx]; idx += 1
            oh = oh[perm]

        bg_m = (frame == bg).astype(np.float32)
        rar = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0: rar[frame == c] = 1.0 - cnt[c]/mx
            
        pad = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2,1:-1]) | (frame != pad[2:,1:-1]) | (frame != pad[1:-1,:-2]) | (frame != pad[1:-1,2:])).astype(np.float32)
        rp = np.linspace(0, 1, 64, dtype=np.float32).reshape(64, 1).repeat(64, 1)
        cp = np.linspace(0, 1, 64, dtype=np.float32).reshape(1, 64).repeat(64, 0)
        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))
        
        d1 = torch.zeros(3, 64, 64, dtype=torch.float32)
        for i, prev in enumerate(reversed(list(self.fhist))):
            if i >= 3: break
            d1[i] = torch.from_numpy((frame != prev).astype(np.float32))
            
        d2 = torch.zeros(2, 64, 64, dtype=torch.float32)
        h = list(self.fhist)
        if len(h) >= 2: d2[0] = torch.from_numpy((h[-1] != h[-2]).astype(np.float32))
        if len(h) >= 4: d2[1] = torch.from_numpy((h[-2] != h[-4]).astype(np.float32))
        
        return torch.cat([oh, aug, d1, d2], 0).to(self.device, non_blocking=True)

    def _train(self):
        if len(self.buf) < self.bsz: return
        indices = np.random.choice(len(self.buf), self.bsz, replace=False)
        batch = [self.buf[i] for i in indices]
        
        states = torch.stack([self._frame_to_tensor(e['s'], augment=True) for e in batch])
        next_states = torch.stack([self._frame_to_tensor(e['s_next'], augment=True) for e in batch])
        acts = torch.tensor([e['a'] for e in batch], dtype=torch.long, device=self.device)
        rews = torch.tensor([e['r'] for e in batch], dtype=torch.float32, device=self.device)
        
        self.opt.zero_grad(set_to_none=True)
        
        if self.scaler:
            with torch.amp.autocast('cuda'):
                logits = self.net(states)
                acts_c = acts.clamp(0, logits.size(1) - 1)
                q_vals = logits.gather(1, acts_c.unsqueeze(1)).squeeze(1)
                
                with torch.no_grad():
                    next_logits = self.net(next_states)
                    max_next_q = next_logits.max(1)[0]
                    targets = rews + 0.95 * max_next_q
                    
                loss = F.mse_loss(q_vals, targets) + 1e-4 * logits.pow(2).mean()

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 10.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            logits = self.net(states)
            acts_c = acts.clamp(0, logits.size(1) - 1)
            q_vals = logits.gather(1, acts_c.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_logits = self.net(next_states)
                targets = rews + 0.95 * next_logits.max(1)[0]
            loss = F.mse_loss(q_vals, targets) + 1e-4 * logits.pow(2).mean()
            loss.backward()
            self.opt.step()

    def _get_aem_tensors(self):
        if len(self._aem_diffs) < 2: return None, None, None
        M = len(self._aem_diffs)
        diffs = torch.zeros(1, M, 1, 64, 64, device=self.device)
        acts = torch.zeros(1, M, dtype=torch.long, device=self.device)
        rews = torch.zeros(1, M, device=self.device)
        for i, (d, a, r) in enumerate(zip(self._aem_diffs, self._aem_actions, self._aem_rewards)):
            diffs[0, i, 0] = torch.from_numpy(d.astype(np.float32)); acts[0, i] = min(a, 4); rews[0, i] = r
        return diffs, acts, rews

    def _reward(self, prev_raw, curr_raw, prev_sh, curr_sh, prev_sg, curr_sg):
        r = 0.0
        mask = np.ones((64, 64), dtype=bool); mask[:2] = False; mask[62:] = False
        if np.any((prev_raw != curr_raw) & mask): r += 0.5
        
        if curr_sh != prev_sh:
            if curr_sh not in self._visited_shashes:
                r += 2.0  # Eureka: Semantic State Change Found
            else: r += 0.2
        else: r -= 0.1
            
        if prev_sg and curr_sg and len(prev_sg) != len(curr_sg):
            r += 1.0  # Object dynamically created or destroyed
            
        return r

    def is_done(self, frames, lf):
        try: return lf.state is GameState.WIN or (time.time() - self.start_time) >= 8*3600-300
        except: return True

    def choose_action(self, frames, lf):
        try:
            lvl = self._lvl(lf)

            # ===== LEVEL CHANGE & INIT =====
            if lvl != self.cl:
                self.cl = lvl
                self._init_swarm_env()
                self.buf.clear(); self.buf_h.clear()
                
                # A100 optimized network boot
                base_net = ForgeNetV21(self.IN, self.G).to(self.device)
                if hasattr(torch, 'compile'):
                    try: self.net = torch.compile(base_net)
                    except: self.net = base_net
                else: self.net = base_net
                
                self.opt = optim.Adam(self.net.parameters(), lr=0.0003)
                
                self.pt=None; self.pai=None; self.pr=None; self.ph=None; self.pr_sg=None
                self.fhist.clear(); self.la=0; self._wd=False
                self._aem_diffs.clear(); self._aem_actions.clear(); self._aem_rewards.clear()
                self._unproductive=0; self._visited_shashes = set()
                
                self.blackboard.clear()
                self.teleport_queue.clear()
                self._swarm_solution = None; self._swarm_step = 0
                
                self._bg = int(np.bincount(self._raw(lf).flatten(), minlength=16).argmax())
                
                # Multi-Threaded Sandbox Rollouts
                self._run_swarm(lvl, self._bg)
                
                if not self._swarm_solution:
                    self._eps = 0.15

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.pt=None; self.pai=None; self.pr=None; self.ph=None; self.pr_sg=None
                self.fhist.clear() 
                a = GameAction.RESET; a.reasoning = "reset"; return a

            # ===== SWARM QUEUE EXECUTION =====
            if self._swarm_solution and self._swarm_step < len(self._swarm_solution):
                act_id, data = self._swarm_solution[self._swarm_step]
                self._swarm_step += 1
                sel = GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning = f"swarm:{self._swarm_step}/{len(self._swarm_solution)}"
                self.fhist.append(self._raw(lf).copy())
                self.la += 1
                return sel

            # ===== GO-EXPLORE TELEPORT QUEUE =====
            if self.teleport_queue:
                act_id, data = self.teleport_queue.popleft()
                sel = GameAction.from_id(act_id)
                if data: sel.set_data(data)
                sel.reasoning = f"teleport_replay"
                self.fhist.append(self._raw(lf).copy())
                self.la += 1
                return sel

            # ===== SEMANTIC RL LOOP =====
            raw = self._raw(lf)
            self.fhist.append(raw.copy()) 
            tensor = self._tensor(lf)
            avail = getattr(lf, 'available_actions', None) or []

            sg_curr = self.director.extract_scene_graph(raw, self._bg)
            curr_sh = self.director.semantic_hash(sg_curr)
            is_novel = curr_sh not in self._visited_shashes
            self._visited_shashes.add(curr_sh)

            if self.pt is not None and self.pai is not None:
                mask = np.ones((64, 64), dtype=bool); mask[:2] = False; mask[62:] = False
                changed = np.any((self.pr != raw) & mask)
                eh = hashlib.md5(self.pr.tobytes()[:1000] + str(self.pai).encode()).hexdigest()[:16]
                
                if eh not in self.buf_h:
                    r = self._reward(self.pr, raw, self.ph, curr_sh, self.pr_sg, sg_curr)
                    self.buf.append({'s': self.pr.copy(), 'a': self.pai, 'r': r, 's_next': raw.copy()})
                    self.buf_h.add(eh)
                    if changed:
                        self._aem_diffs.append((self.pr != raw) & mask)
                        self._aem_actions.append(min(self.pai, 4))
                        self._aem_rewards.append(r)
                        
                if r <= 0: self._unproductive += 1
                else: self._unproductive = 0

            # --- Go-Explore Teleport Trigger ---
            if self._unproductive >= 30 and len(self.blackboard) > 0:
                unvisited = [h for h in self.blackboard.keys() if h not in self._visited_shashes]
                if unvisited:
                    target_h = random.choice(unvisited)
                    target_traj = self.blackboard[target_h]
                    logger.info(f"Agent stuck: Teleporting to semantic frontier {target_h}")
                    self._unproductive = 0
                    self.teleport_queue = deque(target_traj)
                    self.pt=None; self.pai=None; self.pr=None; self.ph=None; self.pr_sg=None
                    a = GameAction.RESET; a.reasoning = "teleport"; return a

            if not self._wd:
                if self.la < 10: aidx, coords = 0, None
                else:
                    self._wd = True
                    for _ in range(min(5, len(self.buf) // self.bsz)): self._train()

            if self._wd:
                # Semantic Spotlight Masking (Action Space Pruning)
                spotlight_mask = self.director.get_spotlight_mask(sg_curr).to(self.device)
                
                if random.random() < self._eps:
                    logits = torch.zeros(4101, device=self.device)
                    logits[5:] += spotlight_mask
                    aidx, coords = self._sample(logits, avail, temp=2.0)
                else:
                    with torch.no_grad():
                        if self.scaler:
                            with torch.amp.autocast('cuda'):
                                mem = self._get_aem_tensors()
                                if mem[0] is not None: logits = self.net(tensor.unsqueeze(0), *mem).squeeze(0)
                                else: logits = self.net(tensor.unsqueeze(0)).squeeze(0)
                        else:
                            mem = self._get_aem_tensors()
                            if mem[0] is not None: logits = self.net(tensor.unsqueeze(0), *mem).squeeze(0)
                            else: logits = self.net(tensor.unsqueeze(0)).squeeze(0)
                            
                        # Spotlight constraints directly applied to network outputs
                        logits[5:] += spotlight_mask
                        
                    aidx, coords = self._sample(logits, avail, temp=0.5)
                self._eps = max(self._eps_min, self._eps * self._eps_decay)
            elif self.la >= 10: self._wd = True; aidx, coords = 0, None

            if aidx < 5:
                sel = self.al[aidx]; sel.reasoning = f"cnn:a{aidx+1}"
            else:
                sel = GameAction.ACTION6; y, x = coords
                sel.set_data({"x": int(x), "y": int(y)}); sel.reasoning = f"cnn:c({x},{y})"

            self.pt = tensor; self.pai = aidx if aidx < 5 else (5 + coords[0] * self.G + coords[1])
            self.pr = raw.copy(); self.ph = curr_sh; self.pr_sg = sg_curr; self.la += 1
            
            if self.action_counter % self.tfreq == 0 and self._wd: self._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a = random.choice(self.al); a.reasoning = f"err:{e}"; return a

    def _sample(self, logits, avail=None, temp=1.0):
        al = logits[:5].clone(); cl = logits[5:5+4096].clone()
        if avail is not None and len(avail) > 0:
            mask = torch.full_like(al, float('-inf')); a6 = False
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                if 1 <= aid <= 5: mask[aid-1] = 0.0
                elif aid == 6: a6 = True
            al = al + mask
            if not a6: cl = cl + torch.full_like(cl, float('-inf'))
            
        ap = torch.softmax(al/temp, dim=0); cp = torch.softmax(cl/temp, dim=0) / (self.G * self.G)
        if torch.isnan(ap).any(): ap = torch.zeros_like(ap)
        if torch.isnan(cp).any(): cp = torch.zeros_like(cp)
        
        allp = torch.cat([ap, cp]); sm = allp.sum()
        if sm < 1e-8 or torch.isnan(sm): allp = torch.ones_like(allp) / len(allp)
        else: allp = allp / sm
        
        idx = np.random.choice(len(allp), p=allp.cpu().numpy())
        if idx < 5: return idx, None
        ci = idx - 5; return 5, (ci // self.G, ci % self.G)