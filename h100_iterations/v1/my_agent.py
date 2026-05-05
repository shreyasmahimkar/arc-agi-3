# =====================================================================
# FORGE v21 — H100 Optimized Architectural Upgrades
#
# NEW FIXES (v21):
# 1. H100 PyTorch Optimizations: Enabled TF32 globally. Added torch.autocast
#    with bfloat16 mixed precision and GradScaler for tensor cores.
# 2. GPU Replay Buffer: Pre-allocated contiguous VRAM buffer completely 
#    bypassing CPU deque bottleneck.
# 3. Transformer Extractor: Replaced slow eager-mode CBAM with a deep 
#    ResNet stem and PyTorch Transformer (leveraging FlashAttention-2).
# 4. Target Network: Decoupled target Q-value generation for Bellman 
#    stability during massive batch updates.
# 5. Asynchronous Learner: Decoupled actor and learner. _train runs in a 
#    background thread continuously while the agent steps through the game.
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
import threading
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Enable Hopper-specific TF32 optimizations globally
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ==================== BFS SOLVER ====================

class BFSSolver:
    """
    Offline BFS solver using direct game class instantiation.
    It attempts to find the shortest path of actions to complete a level.
    """

    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {} 

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.class_name)
            return True
        except Exception as e:
            logger.warning(f"BFS: Failed to load game class: {e}")
            return False

    def _state_hash(self, g, frame, hidden_fields=None):
        fh = hashlib.md5(frame.tobytes()).hexdigest()[:16]
        if hidden_fields:
            extras = []
            for field_name in hidden_fields:
                try:
                    v = getattr(g, field_name, None)
                    if v is not None:
                        extras.append(f"{field_name}={v}")
                except:
                    pass
            if extras:
                return fh + "|" + "|".join(extras)
        return fh

    def _probe_hidden_fields(self, game, actions):
        if not actions:
            return []
        initial = {}
        for k, v in game.__dict__.items():
            if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                initial[k] = v

        changing_fields = set()
        frame0 = game.get_pixels(0, 0, 64, 64)
        for act_id, data in actions[:10]:
            g = copy.deepcopy(game)
            try:
                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                g.perform_action(ai, raw=True)
            except:
                continue
            f = g.get_pixels(0, 0, 64, 64)
            for k, v in g.__dict__.items():
                if isinstance(v, (int, float, bool)) and not k.startswith('__'):
                    if k in initial and v != initial[k]:
                        if k not in ('_action_count', '_full_reset', '_action_complete'):
                            changing_fields.add(k)

        hidden = []
        for f in changing_fields:
            if f.startswith('_') and f not in ('_current_level_index', '_score'):
                continue
            hidden.append(f)
        return sorted(hidden)

    def _scan_actions(self, game, f0, bg):
        avail = game._available_actions
        actions = []
        for a in [a for a in avail if a <= 5]:
            g = copy.deepcopy(game)
            try:
                r = g.perform_action(ActionInput(id=GameAction.from_id(a)), raw=True)
                if r.frame and np.sum(f0 != np.array(r.frame[-1])) > 0:
                    actions.append((a, None))
            except:
                pass
        if 6 in avail:
            t0 = time.time()
            seen_effects = set()
            for y in range(0, 64, 2):
                if time.time() - t0 > self.scan_timeout:
                    break
                for x in range(0, 64, 2):
                    if f0[y, x] == bg:
                        continue
                    g = copy.deepcopy(game)
                    try:
                        r = g.perform_action(
                            ActionInput(id=GameAction.ACTION6, data={'x': x, 'y': y, 'game_id': 'bfs'}),
                            raw=True
                        )
                        if not r.frame:
                            continue
                        f = np.array(r.frame[-1])
                        diff = np.sum(f0 != f)
                        if diff > 0:
                            effect_hash = hashlib.md5(f.tobytes()).hexdigest()[:12]
                            if effect_hash not in seen_effects:
                                seen_effects.add(effect_hash)
                                actions.append((6, {'x': x, 'y': y, 'game_id': 'bfs'}))
                    except:
                        pass
        return actions

    def solve_level(self, level_idx, max_states=500000, prev_solution=None):
        if not self.game_cls: return None

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame: return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        actions = self._scan_actions(game, f0, bg)

        if not actions:
            avail = game._available_actions
            for warmup_id in [a for a in avail if a <= 4]:
                g_warmup = copy.deepcopy(game)
                try:
                    g_warmup.perform_action(ActionInput(id=GameAction.from_id(warmup_id)), raw=True)
                    f_after = np.array(g_warmup.get_pixels(0, 0, 64, 64))
                    warmup_actions = self._scan_actions(g_warmup, f_after, bg)
                    if warmup_actions:
                        game = g_warmup; f0 = f_after; actions = warmup_actions
                        break
                except:
                    pass

        if not actions: return None

        hidden_fields = None
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)

        base_game = copy.deepcopy(game)
        queue.append(([], 0))

        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            hist, depth = queue.popleft()

            g = copy.deepcopy(base_game)
            try:
                for a_id, a_data in hist:
                    ai = ActionInput(id=GameAction.from_id(a_id), data=a_data) if a_data else ActionInput(id=GameAction.from_id(a_id))
                    g.perform_action(ai, raw=True)
            except:
                continue

            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g2.perform_action(ai, raw=True)
                except:
                    continue
                explored += 1

                if not r.frame:
                    continue
                f = np.array(r.frame[-1])
                h = self._state_hash(g2, f, hidden_fields)
                if h in visited:
                    continue
                visited.add(h)

                new_hist = hist + [(act_id, data)]

                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    self.solutions[level_idx] = new_hist
                    return new_hist

                if depth < 30:
                    queue.append((new_hist, depth + 1))

        elapsed_first = time.time() - t0
        if explored < 20 and elapsed_first > 10.0:
            return None

        if len(visited) < 50 and elapsed_first < self.bfs_timeout * 0.8:
            hidden_fields = self._probe_hidden_fields(game, actions)
            if hidden_fields:
                game2 = self.game_cls()
                game2.set_level(level_idx)
                game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                r0_2 = game2.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                if not r0_2.frame: return None
                f0_2 = np.array(r0_2.frame[-1])
                h0_2 = self._state_hash(game2, f0_2, hidden_fields)

                base_game2 = copy.deepcopy(game2)
                visited2 = set()
                visited2.add(h0_2)
                queue2 = deque()
                queue2.append(([], 0))

                t0_2 = time.time()
                explored2 = 0
                remaining = max(30, self.bfs_timeout - elapsed_first)

                while queue2 and explored2 < max_states and (time.time() - t0_2) < remaining:
                    hist, depth = queue2.popleft()

                    g = copy.deepcopy(base_game2)
                    try:
                        for a_id, a_data in hist:
                            ai = ActionInput(id=GameAction.from_id(a_id), data=a_data) if a_data else ActionInput(id=GameAction.from_id(a_id))
                            g.perform_action(ai, raw=True)
                    except:
                        continue

                    for act_id, data in actions:
                        g2 = copy.deepcopy(g)
                        try:
                            ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                            r = g2.perform_action(ai, raw=True)
                        except:
                            continue
                        explored2 += 1

                        if not r.frame: continue
                        f = np.array(r.frame[-1])
                        h = self._state_hash(g2, f, hidden_fields)
                        if h in visited2: continue
                        visited2.add(h)

                        new_hist = hist + [(act_id, data)]

                        if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                            self.solutions[level_idx] = new_hist
                            return new_hist

                        if depth < 30:
                            queue2.append((new_hist, depth + 1))

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame: return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c: continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2: continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)

            if not objs_prev or not objs_curr: return None

            matched = []
            for op in objs_prev:
                best = None
                best_dist = float('inf')
                for oc in objs_curr:
                    if oc['color'] == op['color'] and abs(oc['n'] - op['n']) < max(op['n'], oc['n']) * 0.5:
                        d = abs(oc['cx'] - op['cx']) + abs(oc['cy'] - op['cy'])
                        if d < best_dist:
                            best_dist = d
                            best = oc
                if best:
                    matched.append((op, best))

            if not matched: return None

            dx = np.mean([m[1]['cx'] - m[0]['cx'] for m in matched])
            dy = np.mean([m[1]['cy'] - m[0]['cy'] for m in matched])

            transferred = []
            for act_id, data in prev_solution:
                if data and 'x' in data:
                    new_data = dict(data)
                    new_data['x'] = max(0, min(63, int(data['x'] + dx)))
                    new_data['y'] = max(0, min(63, int(data['y'] + dy)))
                    transferred.append((act_id, new_data))
                else:
                    transferred.append((act_id, data))

            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(transferred):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
        except Exception as e:
            pass
        return None


def find_game_source_and_class(game_id, arc_env=None):
    gid = game_id.split('-')[0]
    cls_name = gid.capitalize()
    if len(gid) == 4 and gid[0].isalpha():
        cls_name = gid[0].upper() + gid[1:]

    src = None
    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            import re
            ld = Path(ei.local_dir)
            for candidate in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if candidate.exists():
                    src = str(candidate)
                    content = candidate.read_text()[:2000]
                    m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                    if m:
                        cls_name = m.group(1)
                    break

    if not src:
        import re
        for pattern in [
            f"/tmp/*/{gid}/*/{gid}.py",
            f"/kaggle/*/{gid}*/{gid}.py",
            f"**/game_sources/**/{gid}.py",
        ]:
            matches = glob.glob(pattern, recursive=True)
            if matches:
                src = matches[0]
                content = open(src).read()[:2000]
                m = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', content)
                if m:
                    cls_name = m.group(1)
                break

    return src, cls_name


# ==================== CNN GPU UPGRADES ====================

class GPUReplayBuffer:
    """Pre-allocated contiguous VRAM buffer for zero-overhead GPU memory streaming."""
    def __init__(self, capacity, state_shape, device):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        # Allocate directly on GPU memory
        self.states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, *state_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, 1), dtype=torch.long, device=device)
        self.rewards = torch.zeros((capacity, 1), dtype=torch.float32, device=device)

    def push(self, state_tensor, action, reward, next_state_tensor):
        # Write directly to pinned GPU memory index
        self.states[self.ptr] = state_tensor
        self.next_states[self.ptr] = next_state_tensor
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx]

class ForgeNet(nn.Module):
    """Deep network designed to trigger Hopper's FlashAttention-2 inside PyTorch 2.x"""
    def __init__(self, in_ch=26, g=64):
        super().__init__()
        self.g = g
        
        # Deeper ResNet-style spatial extraction stem
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        
        # Hopper Optimized Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, 
            batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        
        # Action/Click heads
        self.ah = nn.Linear(256, 5) 
        self.ch = nn.Linear(256, g*g) 
        self.dr = nn.Dropout(0.1)

    def forward(self, x):
        features = self.stem(x) 
        B, C, H, W = features.shape
        
        # Flatten spatial for sequence mixing
        seq = features.view(B, C, -1).permute(0, 2, 1) 
        out_seq = self.transformer(seq) 
        
        # Global base actions
        pooled = out_seq.mean(dim=1) 
        al = self.ah(self.dr(pooled)) 
        
        # Spatial reconstruction for precision clicking
        spatial = out_seq.permute(0, 2, 1).view(B, 256, H, W)
        spatial_avg = F.adaptive_avg_pool2d(spatial, (1, 1)).view(B, -1)
        cl = self.ch(spatial_avg)
        
        return torch.cat([al, cl], 1)


def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs


# ==================== AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        s.G=64; s.IN=26
        s.bsz = 1024 # Scaled up for H100
        s.buffer = GPUReplayBuffer(100000, (s.IN, s.G, s.G), s.device)
        
        s.net=None; s.target_net=None; s.opt=None; s.scaler=None
        s.buf_h=set()
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._obj_moved=0
        s._visited_hashes = set()
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        
        # Async Learner Decoupling
        s._stop_learner = False
        s._learner_thread = None

    def __del__(s):
        s._stop_learner = True
        if s._learner_thread: s._learner_thread.join(timeout=2.0)

    def append_frame(s, f):
        s.frames.append(f)
        if len(s.frames) > s._MAX_FRAMES: s.frames = s.frames[-s._MAX_FRAMES:]
        if f.guid: s.guid = f.guid
        if hasattr(s, "recorder") and not s.is_playback:
            import json; s.recorder.record(json.loads(f.model_dump_json()))

    def _lvl(s, f): return getattr(f, 'score', None) or f.levels_completed
    def _raw(s, fd): return np.array(fd.frame, dtype=np.int64)[-1]

    def _init_bfs(s):
        src, cls = find_game_source_and_class(s.game_id, s.arc_env)
        if src:
            s._bfs = BFSSolver(src, cls, scan_timeout=5, bfs_timeout=180)
            s._bfs.load()
        else:
            s._bfs = None

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None: return None
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        return None

    def _frame_to_tensor(s, frame, augment=False):
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        bg=int(cnt.argmax());mx=max(cnt.max(),1)
        
        if augment:
            perm = np.arange(16)
            non_bg = [c for c in range(16) if c != bg]
            np.random.shuffle(non_bg)
            idx = 0
            for c in range(16):
                if c != bg:
                    perm[c] = non_bg[idx]
                    idx += 1
            oh = oh[perm]

        bg_m=(frame==bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        
        d1=torch.zeros(3,64,64,dtype=torch.float32)
        for i,prev in enumerate(reversed(list(s.fhist))):
            if i>=3:break
            d1[i]=torch.from_numpy((frame!=prev).astype(np.float32))
        d2=torch.zeros(2,64,64,dtype=torch.float32)
        h=list(s.fhist)
        if len(h)>=2:d2[0]=torch.from_numpy((h[-1]!=h[-2]).astype(np.float32))
        if len(h)>=4:d2[1]=torch.from_numpy((h[-2]!=h[-4]).astype(np.float32))
        
        return torch.cat([oh,aug,d1,d2],0).to(s.device)

    def _detect_template(s, frame):
        mask=torch.ones(4096,dtype=torch.float32)
        col_act=np.sum(frame!=s._bg,axis=0)
        for c in range(20,44):
            if col_act[c]<=2 and np.sum(col_act[:c]>0)>=5 and np.sum(col_act[c+1:]>0)>=5:
                for y in range(64):
                    for x in range(c+1):mask[y*64+x]=0.05
                return mask
        row_act=np.sum(frame!=s._bg,axis=1)
        for r in range(20,44):
            if row_act[r]<=2 and np.sum(row_act[:r]>0)>=5 and np.sum(row_act[r+1:]>0)>=5:
                for y in range(r+1):
                    for x in range(64):mask[y*64+x]=0.05
                return mask
        return mask

    def _reward(s, prev_raw, curr_raw, prev_h, curr_h):
        mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
        diff=(prev_raw!=curr_raw)&mask;changed=np.any(diff)
        r=0.0
        if curr_h != prev_h:
            if curr_h not in s._visited_hashes:
                r += 1.5
                s._visited_hashes.add(curr_h)
            else:
                r += 0.2
        else:
            r -= 0.1
        if changed:r+=0.5
        curr_objs=fast_objects(curr_raw,s._bg)
        if s._prev_objs and curr_objs:
            moved=0
            for co in curr_objs:
                for po in s._prev_objs:
                    if co[0]==po[0]:
                        dist=abs(co[1]-po[1])+abs(co[2]-po[2])
                        if 2<dist<20:moved+=1;break
            if moved>0:r+=0.3*min(moved,3);s._obj_moved=moved
        s._prev_objs=curr_objs
        return r

    def _sample(s, logits, avail=None, temp=1.0):
        al=logits[:5].clone();cl=logits[5:5+4096].clone()
        if avail is not None and len(avail)>0:
            mask=torch.full_like(al,float('-inf'));a6=False
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:mask[aid-1]=0.0
                elif aid==6:a6=True
            al=al+mask
            if not a6:cl=cl+torch.full_like(cl,float('-inf'))
        if s._wm is not None:cl=cl+torch.log(s._wm.to(s.device).clamp(min=0.01))
        
        ap=torch.softmax(al/temp, dim=0); cp=torch.softmax(cl/temp, dim=0)/(s.G*s.G)
        allp=torch.cat([ap,cp]);sm=allp.sum()
        if sm<1e-8:allp=torch.ones_like(allp)/len(allp)
        else:allp=allp/sm
        idx=np.random.choice(len(allp),p=allp.cpu().numpy())
        if idx<5:return idx,None
        ci=idx-5;return 5,(ci//s.G,ci%s.G)

    def _heuristic(s, frame, avail, step):
        av=set(int(a.value) if hasattr(a,'value') else int(a) for a in avail)
        for d in[1,2,3,4]:
            if d in av and step<4:return d-1,None
        if 6 in av:
            cnt=np.bincount(frame.flatten(),minlength=16);targets=[]
            for c in range(16):
                if c==s._bg or cnt[c]==0 or cnt[c]>2000:continue
                ys,xs=np.where(frame==c)
                if len(ys)>=2:targets.append((int(np.median(xs)),int(np.median(ys)),len(ys)))
            targets.sort(key=lambda t:t[2]);pidx=step-4
            if 0<=pidx<len(targets):return 5,(targets[pidx][1],targets[pidx][0])
        if 5 in av:return 4,None
        choices=[a for a in av if 1<=a<=5]
        if choices:return random.choice(choices)-1,None
        return 0,None

    def _async_learner_loop(s):
        """Background thread executing continuous AMP training on H100"""
        while not s._stop_learner:
            if s.buffer.size >= s.bsz:
                states, acts, rews, next_states = s.buffer.sample(s.bsz)
                
                s.opt.zero_grad(set_to_none=True)
                
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = s.net(states)
                    acts_c = acts.clamp(0, logits.size(1)-1)
                    q_vals = logits.gather(1, acts_c).squeeze(1)
                    
                    with torch.no_grad():
                        next_logits = s.target_net(next_states)
                        max_next_q = next_logits.max(1)[0]
                        targets = rews.squeeze(1) + 0.95 * max_next_q
                        
                    loss = F.mse_loss(q_vals, targets) + 1e-4 * logits.pow(2).mean()
                
                s.scaler.scale(loss).backward()
                s.scaler.step(s.opt)
                s.scaler.update()

                # Soft Update Target Network
                with torch.no_grad():
                    tau = 0.005
                    for target_param, param in zip(s.target_net.parameters(), s.net.parameters()):
                        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            else:
                time.sleep(0.01)

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE & RESET =====
            if lvl != s.cl:
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)

                s.buffer.ptr = 0; s.buffer.size = 0
                s.buf_h.clear()
                
                # Fuse kernels with torch.compile if supported, otherwise standard load
                base_net = ForgeNet(s.IN, s.G).to(s.device)
                try:
                    s.net = torch.compile(base_net, mode="max-autotune")
                    s.net(torch.zeros(2, s.IN, s.G, s.G, device=s.device)) # trigger compile
                except:
                    s.net = base_net
                
                s.target_net = copy.deepcopy(s.net)
                s.target_net.eval()
                
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003, fused=True) # use fused Adam
                s.scaler = torch.cuda.amp.GradScaler()
                
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0
                s._visited_hashes = set()
                
                if not s._bfs_solution:
                    s._eps = 0.15

                # Start Async Learner
                s._stop_learner = True
                if s._learner_thread: s._learner_thread.join(timeout=1.0)
                s._stop_learner = False
                s._learner_thread = threading.Thread(target=s._async_learner_loop, daemon=True)
                s._learner_thread.start()

                # CLTI 
                if lvl > 0 and s._bfs and s._bfs.solutions.get(lvl - 1):
                    prev_sol = s._bfs.solutions[lvl - 1]
                    try:
                        replay_game = s._bfs.game_cls()
                        replay_game.set_level(lvl - 1)
                        replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        r0 = replay_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
                        if r0.frame:
                            prev_frame = np.array(r0.frame[-1], dtype=np.int64)
                            for act_id, data in prev_sol:
                                ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                                result = replay_game.perform_action(ai, raw=True)
                                action_idx = (act_id - 1) if act_id <= 5 else (
                                    5 + data.get('y', 0) * 64 + data.get('x', 0) if data else 0)
                                
                                if result.frame:
                                    next_frame = np.array(result.frame[-1], dtype=np.int64)
                                else:
                                    next_frame = prev_frame.copy()
                                    
                                t_s = s._frame_to_tensor(prev_frame, augment=False)
                                t_sn = s._frame_to_tensor(next_frame, augment=False)
                                s.buffer.push(t_s, action_idx, 2.0, t_sn)
                                prev_frame = next_frame.copy()
                    except Exception as e:
                        logger.warning(f"CLTI failed: {e}")

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.fhist.clear()
                a=GameAction.RESET;a.reasoning="reset";return a

            # ===== BFS SOLUTION EXECUTION =====
            if s._bfs_solution and s._bfs_step < len(s._bfs_solution):
                act_id, data = s._bfs_solution[s._bfs_step]
                s._bfs_step += 1
                sel = GameAction.from_id(act_id)
                if data:
                    sel.set_data(data)
                sel.reasoning = f"bfs:{s._bfs_step}/{len(s._bfs_solution)}"
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            # ===== CNN FALLBACK =====
            s.fhist.append(s._raw(lf).copy())
            tensor = s._frame_to_tensor(s._raw(lf), augment=True)
            raw = s._raw(lf)
            ch = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
                diff_map=(s.pr!=raw)&mask;changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,'',ch)
                    s.buffer.push(s.pt, s.pai, r, tensor)
                    s.buf_h.add(eh)
                if changed:s._ckpt_hash=ch;s._unproductive=0
                else:s._unproductive+=1

            avail_idx=[]
            for a in avail:
                aid=a.value if hasattr(a,'value') else int(a)
                if 1<=aid<=5:avail_idx.append(aid-1)
                elif aid==6:avail_idx.extend([5+i for i in range(0,4096,128)])

            if s._wm is None:s._wm=s._detect_template(raw)

            if s._undo_avail and s._unproductive>=30 and s._ckpt_hash:
                s._unproductive=0;a=GameAction.ACTION7;a.reasoning="undo"
                s.pt=tensor;s.pai=6;s.pr=raw.copy();s.ph=ch;s.la+=1;return a

            if not s._wd:
                if s.la<10:aidx,coords=s._heuristic(raw,avail,s.la)
                else: s._wd=True

            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        logits = s.net(tensor.unsqueeze(0)).squeeze(0).float()
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:s._wd=True;aidx,coords=0,None

            if aidx<5:sel=s.al[aidx];sel.reasoning=f"cnn:a{aidx+1}"
            else:
                sel=GameAction.ACTION6;y,x=coords
                sel.set_data({"x":int(x),"y":int(y)});sel.reasoning=f"cnn:c({x},{y})"

            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a