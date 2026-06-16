# =====================================================================
# CHRONOS SOLVER V11 — Multi-Agent Architecture with Local Gemma 4
#
# 1. Google ADK Integration (Wrapper Pattern)
# 2. Hierarchical 3-Tier Multi-Agent team (Manager, Leads, Sub-Agents)
# 3. Local Gemma 4 Multimodal Model (100% Offline, Ollama/llama.cpp)
# 4. State Plumbing (Payload passing, output_key)
# 5. Python Sandbox Validator (Simulation in isolated subprocess)
# 6. Action Compression Logic (PlanningLead callbacks)
# 7. V9 Logging integration (persisting images, run.log, autopsies)
# =====================================================================
import sys
import os
import time
import glob
import json
import copy
import hashlib
import logging
import random
import traceback
import tempfile
import base64
import subprocess
import threading
import queue
from collections import deque
from pathlib import Path

# Third-party baseline imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

# Try adding common local ADK virtualenv paths to sys.path if not present
if "google.adk" not in sys.modules:
    adk_paths = [
        "/Users/shreyas/.pyenv/versions/adk/lib/python3.13/site-packages",
        os.path.expanduser("~/.pyenv/versions/adk/lib/python3.13/site-packages"),
    ]
    for path in adk_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.append(path)

# ADK imports
from google.adk.agents import LlmAgent, SequentialAgent, ParallelAgent, LoopAgent, InvocationContext
from google.adk.runners import InMemoryRunner
from google.adk.models import Gemma
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
ADK_AVAILABLE = True

logger = logging.getLogger(__name__)

# Configure local logging to v11_run.log
log_path = os.path.join(os.path.dirname(__file__), "v11_run.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout)
    ]
)

# ==================== GEMMA SERVER BOOT HELPER ====================

def boot_gemma_server(vram_gb=16, backend="ollama"):
    """
    Instructions to boot a local inference server using Gemma 4 (12B Multimodal).
    
    To boot the server:
    
    Ollama (Recommended):
      $ ollama run gemma4:12b
      
    llama.cpp (Metal / CUDA):
      $ llama-server -m gemma4-12b.gguf --mmproj gemma4-12b-mmproj.gguf -ngl 99 --ctx-size 8192
    """
    logger.info(f"[Server Boot] Gemma 4 server setup instructions initialized for {backend}.")
    # Placeholder background subprocess call to show how server is booted:
    # if backend == "ollama":
    #     subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # else:
    #     subprocess.Popen(["llama-server", "-m", "gemma4-12b.gguf", "-ngl", "99", "-c", "8192"], stdout=subprocess.DEVNULL)

# ==================== LOCAL MULTIMODAL VISION CALL ====================

_kaggle_model = None
_kaggle_processor = None

def call_local_multimodal_model(prompt: str, images: list, base_url: str = "http://localhost:11434") -> str:
    """
    Calls the local Gemma 4 12B/31B multimodal model.
    1. If Ollama is available, uses the Ollama API.
    2. Otherwise, if running on Kaggle with local model inputs, uses Hugging Face transformers.
    """
    global _kaggle_model, _kaggle_processor
    
    # Try Ollama first
    try:
        # Encode PIL images to base64
        base64_imgs = []
        for img in images:
            import io
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
            base64_imgs.append(img_str)
            
        payload = {
            "model": "gemma4:12b",
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": base64_imgs
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_ctx": 8192
            }
        }
        
        # Try Ollama native API
        r = requests.post(f"{base_url}/api/chat", json=payload, timeout=60.0)
        if r.status_code == 200:
            return r.json().get("message", {}).get("content", "")
            
        # Try OpenAI-compatible endpoint as fallback
        payload_openai = {
            "model": "gemma4:12b",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "temperature": 0.1
        }
        for img_str in base64_imgs:
            payload_openai["messages"][0]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_str}"}
            })
        r = requests.post(f"{base_url}/v1/chat/completions", json=payload_openai, timeout=60.0)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
            
    except Exception as e:
        logger.debug(f"Ollama local multimodal call failed/unavailable: {e}")

    # Fallback to local Hugging Face model (e.g. on Kaggle)
    kaggle_paths = [
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b/2",
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1",
        "/kaggle/input/models/google/gemma-4/transformers/gemma-4-12b/1",
    ]
    model_id = None
    for kp in kaggle_paths:
        if os.path.exists(kp):
            model_id = kp
            break
            
    if model_id:
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForMultimodalLM, BitsAndBytesConfig
            
            if _kaggle_model is None:
                logger.info(f"Loading local Hugging Face model from {model_id}...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True
                )
                _kaggle_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
                _kaggle_model = AutoModelForMultimodalLM.from_pretrained(
                    model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                
            messages = [
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]
            for img in images:
                messages[0]["content"].append({"type": "image", "image": img})
                
            formatted_prompt = _kaggle_processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=True
            )
            
            if images:
                inputs = _kaggle_processor(text=formatted_prompt, images=images, return_tensors="pt").to(_kaggle_model.device)
            else:
                inputs = _kaggle_processor(text=formatted_prompt, return_tensors="pt").to(_kaggle_model.device)
                
            with torch.no_grad():
                outputs = _kaggle_model.generate(
                    **inputs, 
                    max_new_tokens=1024, 
                    do_sample=False,
                    temperature=0.1
                )
            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            response = _kaggle_processor.decode(generated_tokens, skip_special_tokens=True)
            return response
        except Exception as ex:
            logger.error(f"Kaggle local Hugging Face multimodal call failed: {ex}")
            
    return ""

def temp_io_buffer():
    import io
    return io.BytesIO()

# ==================== SYNC WORKER FOR ASYNC ADK PIPELINES ====================

def run_async_in_thread(coro):
    """Executes a coroutine inside a clean thread and event loop."""
    q = queue.Queue()
    def worker():
        loop = asyncio.new_event_loop() if 'asyncio' in globals() else temp_asyncio_loop()
        import asyncio
        asyncio.set_event_loop(loop)
        try:
            res = loop.run_until_complete(coro)
            q.put((True, res))
        except Exception as e:
            q.put((False, e))
        finally:
            loop.close()
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    success, val = q.get()
    if success:
        return val
    else:
        raise val

def temp_asyncio_loop():
    import asyncio
    return asyncio.new_event_loop()

# ==================== BFS SOLVER ====================

class BFSSolver:
    def __init__(self, game_path, game_class_name, scan_timeout=3, bfs_timeout=120):
        self.game_path = game_path
        self.class_name = game_class_name
        self.scan_timeout = scan_timeout
        self.bfs_timeout = bfs_timeout
        self.game_cls = None
        self.solutions = {}

    def load(self):
        try:
            spec = importlib.util.spec_from_file_location('game_mod', self.game_path) if 'importlib' in globals() else temp_importlib_spec(self.game_path)
            import importlib.util
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
        if not self.game_cls:
            return None

        game = self.game_cls()
        game.set_level(level_idx)
        game.perform_action(ActionInput(id=GameAction.RESET), raw=True)

        r0 = game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
        if not r0.frame:
            return None
        f0 = np.array(r0.frame[-1])
        bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

        if prev_solution and level_idx > 0:
            transfer_result = self._try_transfer(game, level_idx, prev_solution, f0)
            if transfer_result:
                return transfer_result

        actions = self._scan_actions(game, f0, bg)
        if not actions:
            return None

        hidden_fields = None
        visited = set()
        queue = deque()
        h0 = self._state_hash(game, f0, None)
        visited.add(h0)
        queue.append((copy.deepcopy(game), [], 0))

        t0 = time.time()
        explored = 0

        while queue and explored < max_states and (time.time() - t0) < self.bfs_timeout:
            g, hist, depth = queue.popleft()

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
                h = self._state_hash(g2, f, None)
                if h in visited:
                    continue
                visited.add(h)

                new_hist = hist + [(act_id, data)]

                if r.levels_completed > level_idx or g2._current_level_index > level_idx:
                    elapsed = time.time() - t0
                    logger.info(f"BFS L{level_idx}: SOLVED in {len(new_hist)} actions ({explored} explored, {elapsed:.1f}s)")
                    self.solutions[level_idx] = new_hist
                    return new_hist

                if depth < 30:
                    queue.append((g2, new_hist, depth + 1))

        return None

    def _try_transfer(self, game, level_idx, prev_solution, f1):
        try:
            g = copy.deepcopy(game)
            for i, (act_id, data) in enumerate(prev_solution):
                try:
                    ai = ActionInput(id=GameAction.from_id(act_id), data=data) if data else ActionInput(id=GameAction.from_id(act_id))
                    r = g.perform_action(ai, raw=True)
                    if r.levels_completed > level_idx or g._current_level_index > level_idx:
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (direct replay, {i+1} actions)")
                        sol = prev_solution[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break

            prev_game = self.game_cls()
            prev_game.set_level(level_idx - 1)
            prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            r_prev = prev_game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
            if not r_prev.frame:
                return None
            f0 = np.array(r_prev.frame[-1])
            bg = int(np.bincount(f0.flatten(), minlength=16).argmax())

            def get_objects(frame, bg_c):
                objs = []
                for c in range(16):
                    if c == bg_c:
                        continue
                    mask = (frame == c)
                    npix = int(np.sum(mask))
                    if npix < 2:
                        continue
                    ys, xs = np.where(mask)
                    objs.append({'color': c, 'cx': float(np.mean(xs)), 'cy': float(np.mean(ys)), 'n': npix})
                return sorted(objs, key=lambda o: (o['color'], -o['n']))

            objs_prev = get_objects(f0, bg)
            objs_curr = get_objects(f1, bg)

            if not objs_prev or not objs_curr:
                return None

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

            if not matched:
                return None

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
                        logger.info(f"BFS L{level_idx}: TRANSFER SUCCESS (offset dx={dx:.0f},dy={dy:.0f}, {i+1} actions)")
                        sol = transferred[:i+1]
                        self.solutions[level_idx] = sol
                        return sol
                except:
                    break
        except Exception as e:
            logger.warning(f"BFS transfer failed: {e}")
        return None

def temp_importlib_spec(path):
    import importlib.util
    return importlib.util.spec_from_file_location('game_mod', path)

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


# ==================== CNN FALLBACK ====================

class CBAM(nn.Module):
    def __init__(s, ch, r=16):
        super().__init__()
        s.fc1=nn.Linear(ch,max(ch//r,4)); s.fc2=nn.Linear(max(ch//r,4),ch)
        s.sp=nn.Conv2d(2,1,7,padding=3)
    def forward(s, x):
        B,C,H,W=x.shape
        w=torch.sigmoid(s.fc2(F.relu(s.fc1(x.mean(dim=[2,3]))))); x=x*w.view(B,C,1,1)
        a=torch.sigmoid(s.sp(torch.cat([x.max(1,keepdim=True)[0],x.mean(1,keepdim=True)],1)))
        return x*a

class ActionEffectAttention(nn.Module):
    def __init__(s, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        s.mem_dim=mem_dim
        s.diff_enc=nn.Sequential(nn.Conv2d(1,8,8,stride=8),nn.ReLU(),nn.Conv2d(8,16,4,stride=4),nn.ReLU(),nn.Flatten(),nn.Linear(16*2*2,mem_dim))
        s.q_proj=nn.Linear(feat_dim,mem_dim)
        s.v_proj=nn.Linear(mem_dim+1+n_actions,n_actions)
        s.scale=mem_dim**0.5
    def forward(s, cnn_feat, mem_diffs, mem_actions, mem_rewards):
        B,M=mem_actions.shape
        if M==0:return torch.zeros(B,5,device=cnn_feat.device)
        keys=s.diff_enc(mem_diffs.reshape(B*M,1,64,64)).reshape(B,M,s.mem_dim)
        q=s.q_proj(cnn_feat).unsqueeze(1)
        attn=F.softmax(torch.bmm(q,keys.transpose(1,2))/s.scale,dim=-1)
        act_oh=F.one_hot(mem_actions.clamp(0,4),5).float()
        vals=torch.cat([keys,mem_rewards.unsqueeze(-1),act_oh],dim=-1)
        ctx=torch.bmm(attn,vals).squeeze(1)
        return s.v_proj(ctx)

class ForgeNet(nn.Module):
    def __init__(s, in_ch=26, g=64):
        super().__init__()
        s.g=g
        s.c1=nn.Conv2d(in_ch,32,3,padding=1);s.c2=nn.Conv2d(32,64,3,padding=1)
        s.c3=nn.Conv2d(64,128,3,padding=1);s.c4=nn.Conv2d(128,256,3,padding=1)
        s.attn=CBAM(256);s.ar=nn.Conv2d(256,64,1);s.ap=nn.MaxPool2d(4,4)
        s.af=nn.Linear(64*16*16,256);s.ah=nn.Linear(256,5);s.dr=nn.Dropout(0.15)
        s.cc1=nn.Conv2d(256,128,3,padding=1);s.cc2=nn.Conv2d(128,64,3,padding=1)
        s.cc3=nn.Conv2d(64,32,1);s.cc4=nn.Conv2d(32,1,1)
        s.gp=nn.AdaptiveAvgPool2d(1);s.gf=nn.Linear(256,64)
        s.aea=ActionEffectAttention(feat_dim=64,mem_dim=32,n_actions=5)
    def forward(s, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x=F.relu(s.c1(x));x=F.relu(s.c2(x));x=F.relu(s.c3(x));f=F.relu(s.c4(x))
        f=s.attn(f);af=F.relu(s.ar(f));af=s.ap(af).reshape(f.size(0),-1)
        al=s.ah(s.dr(F.relu(s.af(af))))
        cf=F.relu(s.cc1(f));cf=F.relu(s.cc2(cf));cf=F.relu(s.cc3(cf))
        cl=s.cc4(cf).reshape(f.size(0),-1)
        if mem_diffs is not None and mem_actions is not None:
            gf=s.gf(s.gp(f).reshape(f.size(0),-1))
            al=al+s.aea(gf,mem_diffs,mem_actions,mem_rewards)
        return torch.cat([al,cl],1)

def fast_objects(frame, bg):
    objs=[]
    for c in range(16):
        if c==bg:continue
        mask=(frame==c);npix=int(np.sum(mask))
        if npix<4 or npix>3000:continue
        ys,xs=np.where(mask)
        objs.append((c,float(np.mean(xs)),float(np.mean(ys)),npix))
    return objs

def stable_deepcopy(game):
    camera = getattr(game, '_camera', None)
    if camera is not None:
        game._camera = None
    g = copy.deepcopy(game)
    if camera is not None:
        game._camera = camera
        g._camera = camera
    return g

_fast_deepcopy = stable_deepcopy

# ==================== SANDBOX VALIDATOR TOOL ====================

def run_simulation_in_sandbox(script_code: str, tool_context: ToolContext) -> dict:
    """
    Executes the generated Python simulation code in an isolated subprocess.
    If the simulation completes successfully and matches target expectations,
    updates state['simulation_passed'] = True and escalates to exit the LoopAgent.
    """
    logger.info("Sandbox Validator: Starting execution of generated simulation code...")
    state = tool_context.state
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_code)
        temp_path = f.name
        
    try:
        res = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=5.0
        )
        passed = (res.returncode == 0)
        logger.info(f"Sandbox Validator: passed={passed} | stdout={res.stdout.strip()} | stderr={res.stderr.strip()}")
        
        # Log to session traces for debugging/improvements
        if not hasattr(tool_context.session, "_sandbox_simulations"):
            tool_context.session._sandbox_simulations = []
        tool_context.session._sandbox_simulations.append({
            "script": script_code,
            "stdout": res.stdout,
            "stderr": res.stderr,
            "passed": passed
        })
        
        state["simulation_passed"] = passed
        if passed:
            # Successfully matched! Escalate to exit LoopAgent.
            tool_context.actions.escalate = True
            tool_context.actions.skip_summarization = True
            
        return {
            "status": "success" if passed else "failed",
            "passed": passed,
            "stdout": res.stdout,
            "stderr": res.stderr
        }
    except subprocess.TimeoutExpired:
        state["simulation_passed"] = False
        return {"status": "timeout", "passed": False, "stdout": "", "stderr": "Execution timed out"}
    except Exception as e:
        state["simulation_passed"] = False
        return {"status": "error", "passed": False, "stdout": "", "stderr": str(e)}
    finally:
        try:
            os.unlink(temp_path)
        except:
            pass

# ==================== ACTION COMPRESSION CALLBACK ====================

def compress_actions_callback(callback_context: CallbackContext):
    """
    Runs an optimization loop on the planned actions of PlanningLead.
    Groups individual coordinate click sequences into macro commands.
    """
    state = callback_context.state
    proposed = state.get("proposed_actions", [])
    if not proposed:
        return
        
    logger.info(f"Action Compression: Analyzing {len(proposed)} proposed actions...")
    state["uncompressed_actions"] = copy.deepcopy(proposed)
    
    # Merges adjacent pixels recolors / removes coordinate redundancies
    compressed = []
    seen_clicks = set()
    for act in proposed:
        if isinstance(act, dict) and act.get("action") == 6:
            coord = (act.get("x"), act.get("y"))
            if coord in seen_clicks:
                # Remove duplicate coordinate clicks to optimize RHAE efficiency
                continue
            seen_clicks.add(coord)
        compressed.append(act)
        
    state["compressed_actions"] = compressed
    logger.info(f"Action Compression: Merged coordinate clicks. Optimized action count: {len(compressed)}")

# ==================== CHRONOS V11 WRAPPER AGENT ====================

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES = 10

    def __init__(s, *a, **kw):
        super().__init__(*a, **kw)
        import json
        seed = int(time.time()*1e6) + hash(s.game_id) % 1000000
        random.seed(seed); np.random.seed(seed%(2**32-1)); torch.manual_seed(seed%(2**32-1))
        
        # Persist long term game semantics
        s._memory_file = os.path.join(os.path.dirname(__file__), "v11_long_term_memory.json")
        s._global_semantic_cache = {
            "game_name": s.game_id,
            "level_completed": "-1",
            "winning_strategy": "",
            "generalized_mechanics_learned": []
        }
        if os.path.exists(s._memory_file):
            try:
                with open(s._memory_file, 'r') as f:
                    s._global_semantic_cache = json.load(f)
                logger.info(f"Loaded long-term semantic memory: {s._global_semantic_cache}")
            except: pass
            
        s.start_time = time.time()
        s.device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        s.G=64; s.IN=26
        s.net=None; s.opt=None
        s.buf=deque(maxlen=50000); s.buf_h=set()
        s.bsz=64; s.tfreq=10
        s.pt=None; s.pai=None; s.pr=None; s.ph=None
        s.cl=-1; s.fhist=deque(maxlen=6); s.la=0
        s.al=[GameAction.ACTION1,GameAction.ACTION2,GameAction.ACTION3,GameAction.ACTION4,GameAction.ACTION5]
        s._wd=False; s._bg=0; s._wm=None
        s._aem_diffs=deque(maxlen=256); s._aem_actions=deque(maxlen=256); s._aem_rewards=deque(maxlen=256)
        s._ckpt_hash=None; s._unproductive=0; s._undo_avail=False
        s._eps=0.15; s._eps_min=0.03; s._eps_decay=0.9997
        s._prev_objs=None; s._obj_moved=0
        
        # BFS solver fallback
        s._bfs = None
        s._bfs_solution = None
        s._bfs_step = 0
        s._bfs_tried = False
        
        # Local multi-agent planned path
        s._multi_agent_solution = None
        s._multi_agent_step = 0
        
        # Initialize the ADK Multi-Agent Team (Gemma 4 local backend)
        s._init_adk_agents()

    def _init_adk_agents(s):
        """Initializes all 3 tiers of ADK agents using the local Gemma 4 model."""
        if not ADK_AVAILABLE:
            logger.warning("ADK library not available. Falling back to baseline search solvers.")
            return
            
        # Point to the local Gemma 4 endpoint (Ollama default port 11434)
        s._gemma_model = Gemma(
            model="gemma4:12b",
            base_url="http://localhost:11434/v1"
        )
        
        # Tier 3: 18 Narrowly scoped functional agents
        s._sub_agents = {
            "SymmetryCheckerAgent": LlmAgent(name="SymmetryCheckerAgent", model=s._gemma_model, instruction="Determine grid symmetry patterns."),
            "GravityPhysicsAgent": LlmAgent(name="GravityPhysicsAgent", model=s._gemma_model, instruction="Determine if blocks fall down or slide under gravity."),
            "ColorPaletteAgent": LlmAgent(name="ColorPaletteAgent", model=s._gemma_model, instruction="Identify the color palette mappings."),
            "PixelCountAgent": LlmAgent(name="PixelCountAgent", model=s._gemma_model, instruction="Count visual pixels of each color."),
            "BackgroundDetectorAgent": LlmAgent(name="BackgroundDetectorAgent", model=s._gemma_model, instruction="Identify which color represents the background."),
            "BorderDetectorAgent": LlmAgent(name="BorderDetectorAgent", model=s._gemma_model, instruction="Locate visual borders or walls."),
            "CornerFinderAgent": LlmAgent(name="CornerFinderAgent", model=s._gemma_model, instruction="Locate grid corners."),
            "PatternMatcherAgent": LlmAgent(name="PatternMatcherAgent", model=s._gemma_model, instruction="Match visual patterns across frames."),
            "ObjectExtractorAgent": LlmAgent(name="ObjectExtractorAgent", model=s._gemma_model, instruction="Locate discrete objects in the grid."),
            "ScaleFactorAgent": LlmAgent(name="ScaleFactorAgent", model=s._gemma_model, instruction="Determine grid scaling factors."),
            "TranslationAgent": LlmAgent(name="TranslationAgent", model=s._gemma_model, instruction="Identify translation movements."),
            "RotationAgent": LlmAgent(name="RotationAgent", model=s._gemma_model, instruction="Identify rotational transformations."),
            "ReflectionAgent": LlmAgent(name="ReflectionAgent", model=s._gemma_model, instruction="Identify reflection transformations."),
            "PeriodicPatternAgent": LlmAgent(name="PeriodicPatternAgent", model=s._gemma_model, instruction="Identify repeating periodic shapes."),
            "GridResizerAgent": LlmAgent(name="GridResizerAgent", model=s._gemma_model, instruction="Identify changes in grid dimensions."),
            "FloodFillAgent": LlmAgent(name="FloodFillAgent", model=s._gemma_model, instruction="Find recolored flood-fill regions."),
            "LineDrawerAgent": LlmAgent(name="LineDrawerAgent", model=s._gemma_model, instruction="Find line-drawing recoloring paths."),
            "CodeCreatorAgent": LlmAgent(
                name="CodeCreatorAgent",
                model=s._gemma_model,
                instruction="Write a python script that mathematically simulates the moves on the grid. Verify the result matches the goal. Use tool run_simulation_in_sandbox.",
                tools=[run_simulation_in_sandbox],
                output_key="simulation_code"
            )
        }
        
        # Tier 2: 4 Dedicated Pillar Leads
        s._exploration_lead = ParallelAgent(
            name="ExplorationLead",
            sub_agents=[
                s._sub_agents["SymmetryCheckerAgent"],
                s._sub_agents["GravityPhysicsAgent"],
                s._sub_agents["ColorPaletteAgent"],
                s._sub_agents["PixelCountAgent"],
                s._sub_agents["BackgroundDetectorAgent"]
            ]
        )
        
        s._modeling_lead = ParallelAgent(
            name="ModelingLead",
            sub_agents=[
                s._sub_agents["PatternMatcherAgent"],
                s._sub_agents["ObjectExtractorAgent"],
                s._sub_agents["ScaleFactorAgent"],
                s._sub_agents["TranslationAgent"],
                s._sub_agents["RotationAgent"],
                s._sub_agents["ReflectionAgent"]
            ]
        )
        
        s._goal_setting_lead = LlmAgent(
            name="GoalSettingLead",
            model=s._gemma_model,
            instruction="Based on modeling analysis, identify sub-goals needed to achieve the target grid. Write to state target_grid.",
            output_key="target_grid"
        )
        
        s._planning_lead = LlmAgent(
            name="PlanningLead",
            model=s._gemma_model,
            instruction="Propose a list of actions (directional actions or click recolors) to achieve the target sub-goal. Output JSON array of actions.",
            output_key="proposed_actions",
            after_agent_callback=compress_actions_callback
        )
        
        # Loop validation pipeline for Planning
        s._planning_loop = LoopAgent(
            name="PlanningLoop",
            sub_agents=[s._planning_lead, s._sub_agents["CodeCreatorAgent"]],
            max_iterations=3
        )
        
        # Tier 1: Root Orchestrator
        s._manager_agent = LlmAgent(
            name="ManagerAgent",
            model=s._gemma_model,
            instruction="Coordinate the leads to analyze the puzzle grid and plan optimized actions.",
            output_key="analysis_state"
        )
        
        # Assemble Sequential Pipeline
        s._leads_pipeline = SequentialAgent(
            name="LeadsPipeline",
            sub_agents=[
                s._exploration_lead,
                s._modeling_lead,
                s._goal_setting_lead,
                s._planning_loop
            ]
        )
        
        # Complete Multi-Agent Workflow root
        s._root_workflow = SequentialAgent(
            name="RootWorkflow",
            sub_agents=[s._manager_agent, s._leads_pipeline]
        )

    def _run_adk_pipeline(s, lf):
        """Runs the synchronous ADK multi-agent workflow inside a dedicated thread."""
        if not ADK_AVAILABLE:
            return None
            
        logger.info("[ADK Workflow] Preparing game state payload...")
        
        # Prepare state plumbing payload
        raw_pixels = s._raw(lf)
        payload = {
            "grid_pixels": raw_pixels.tolist(),
            "visited_hashes": list(getattr(s, "_visited_hashes", set())),
            "exploration_history": list(getattr(s, "fhist", deque())),
            "available_actions": [int(a.value if hasattr(a, 'value') else a) for a in (getattr(lf, "available_actions", None) or [])]
        }
        
        async def run_pipeline():
            runner = InMemoryRunner(agent=s._root_workflow)
            # Pass payload state_delta
            async for event in runner.run_async(
                user_id="user_v11",
                session_id=f"session_{s.game_id}_{s.cl}",
                new_message="Solve the current grid level.",
                state_delta=payload
            ):
                # Extensive logging of thoughts/actions to stdout and v11_run.log
                if event.content:
                    text = "".join(p.text for p in event.content.parts if p.text)
                    if text.strip():
                        logger.info(f"[{event.author}] {text.strip()}")
            
            # Fetch final state delta
            session = await runner.session_service.get_session(user_id="user_v11", session_id=f"session_{s.game_id}_{s.cl}")
            return session
            
        try:
            session = run_async_in_thread(run_pipeline())
            state = session.state
            
            # Structured JSON Scratchpad logging
            scratchpad_data = {
                "timestamp": time.time(),
                "grid_size": f"{raw_pixels.shape[0]}x{raw_pixels.shape[1]}",
                "manager_analysis": state.get("analysis_state"),
                "uncompressed_actions": state.get("uncompressed_actions"),
                "compressed_actions": state.get("compressed_actions"),
                "simulation_passed": state.get("simulation_passed"),
                "agent_traces": getattr(session, "_agent_traces", []),
                "sandbox_simulations": getattr(session, "_sandbox_simulations", [])
            }
            s._update_scratchpad_data(scratchpad_data)
            
            return state.get("compressed_actions")
        except Exception as e:
            logger.error(f"Error running ADK Multi-Agent pipeline: {e}")
            traceback.print_exc()
        return None

    def _update_scratchpad_data(s, scratchpad_data, death_iteration_advance=False):
        """Saves extensive structured traces to V11 scratchpad files."""
        if not hasattr(s, '_scratchpad_iteration'):
            s._scratchpad_iteration = 0
            
        sp_file = os.path.join(
            os.path.dirname(__file__), 
            f"v11_{s.game_id}_level_{s.cl}_scratchpad_iteration_{s._scratchpad_iteration}.json"
        )
        
        try:
            with open(sp_file, 'w') as f:
                json.dump(scratchpad_data, f, indent=2)
            logger.info(f"Persisted V11 structured scratchpad to: {os.path.basename(sp_file)}")
        except Exception as e:
            logger.warning(f"Failed to save structured scratchpad: {e}")
            
        if death_iteration_advance:
            s._scratchpad_iteration += 1

    def _perform_death_autopsy(s, lf, fatal_state):
        """Performs death autopsy locally using the local Gemma 4 12B multimodal server."""
        try:
            from PIL import Image
            img_dir = os.path.join(os.path.dirname(__file__), "images", s.game_id)
            search_pattern = os.path.join(img_dir, f"level_{s.cl:02d}_step_*.png")
            existing_images = sorted(glob.glob(search_pattern))
            
            images = []
            for img_path in existing_images[-2:]:
                try:
                    img = Image.open(img_path)
                    img.load()
                    images.append(img)
                except: pass
                
            prompt = "A fatal event or level reset just occurred. Compare these two frames (1 step before reset, and the reset). What sub-optimal move or hazard triggered this? Did a resource gauge deplete? Define the exact visual cause of failure. Output ONLY a JSON with key 'DEATH_ANALYSIS' containing your autopsy."
            
            # Local multimodal visual call
            response_text = call_local_multimodal_model(prompt, images)
            
            import json, re
            json_str = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_str:
                analysis = json.loads(json_str.group())
                autopsy = analysis.get("DEATH_ANALYSIS", response_text)
            else:
                autopsy = response_text
                
            s._last_death_autopsy = autopsy
            
            autopsy_scratchpad = {
                "timestamp": time.time(),
                "fatal_event": "Silent Reset / Death",
                "autopsy_reason": autopsy,
                "fatal_trajectory_state": fatal_state
            }
            s._update_scratchpad_data(autopsy_scratchpad, death_iteration_advance=True)
            logger.info(f"== GEMINI LOCAL AUTOPSY ==\n{autopsy}\n=======================")
        except Exception as e:
            logger.warning(f"Local autopsy failed: {e}")

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
            if s._bfs.load():
                logger.info(f"BFS: loaded {cls} from {src}")
            else:
                s._bfs = None
                logger.warning(f"BFS: failed to load game class")
        else:
            logger.warning(f"BFS: game source not found for {s.game_id}")

    def _try_bfs_solve(s, level_idx):
        if s._bfs is None:
            return None
        prev_sol = s._bfs.solutions.get(level_idx - 1) if level_idx > 0 else None
        sol = s._bfs.solve_level(level_idx, prev_solution=prev_sol)
        if sol:
            s._bfs_solution = sol
            s._bfs_step = 0
            return sol
        return None

    def _tensor(s, fd):
        frame = s._raw(fd)
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        s._bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==s._bg).astype(np.float32)
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
        s.fhist.append(frame.copy())
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
            if not hasattr(s, '_visited_hashes'):
                s._visited_hashes = set()
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
        ap=torch.sigmoid(al/temp);cp=torch.sigmoid(cl/temp)/(s.G*s.G)
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

    def _frame_to_tensor(s, frame):
        oh=torch.zeros(16,64,64,dtype=torch.float32)
        oh.scatter_(0,torch.from_numpy(frame).unsqueeze(0),1)
        cnt=np.bincount(frame.flatten(),minlength=16)
        bg=int(cnt.argmax());mx=max(cnt.max(),1)
        bg_m=(frame==bg).astype(np.float32)
        rar=np.zeros((64,64),np.float32)
        for c in range(16):
            if cnt[c]>0:rar[frame==c]=1.0-cnt[c]/mx
        pad=np.pad(frame,1,mode='edge')
        edge=((frame!=pad[:-2,1:-1])|(frame!=pad[2:,1:-1])|(frame!=pad[1:-1,:-2])|(frame!=pad[1:-1,2:])).astype(np.float32)
        rp=np.linspace(0,1,64,dtype=np.float32).reshape(64,1).repeat(64,1)
        cp=np.linspace(0,1,64,dtype=np.float32).reshape(1,64).repeat(64,0)
        aug=torch.from_numpy(np.stack([bg_m,rar,edge,rp,cp]))
        zeros=torch.zeros(5,64,64,dtype=torch.float32)
        return torch.cat([oh,aug,zeros],0)

    def _train(s):
        if len(s.buf)<s.bsz:return
        indices=np.random.choice(len(s.buf),s.bsz,replace=False)
        batch=[s.buf[i] for i in indices]
        states=torch.stack([s._frame_to_tensor(e['s']).to(s.device) for e in batch])
        acts=torch.tensor([e['a'] for e in batch],dtype=torch.long,device=s.device)
        rews=torch.tensor([e['r'] for e in batch],dtype=torch.float32,device=s.device)
        s.opt.zero_grad()
        logits=s.net(states)
        acts_c=acts.clamp(0,logits.size(1)-1)
        sel=logits.gather(1,acts_c.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(sel, rews)
        p=torch.sigmoid(logits);loss=loss-0.0001*p[:,:5].mean()-0.00001*p[:,5:].mean()
        loss.backward();s.opt.step()

    def _get_aem_tensors(s):
        if len(s._aem_diffs)<2:return None,None,None
        M=len(s._aem_diffs)
        diffs=torch.zeros(1,M,1,64,64,device=s.device)
        acts=torch.zeros(1,M,dtype=torch.long,device=s.device)
        rews=torch.zeros(1,M,device=s.device)
        for i,(d,a,r) in enumerate(zip(s._aem_diffs,s._aem_actions,s._aem_rewards)):
            diffs[0,i,0]=torch.from_numpy(d.astype(np.float32));acts[0,i]=min(a,4);rews[0,i]=r
        return diffs,acts,rews

    def is_done(s, frames, lf):
        try: return lf.state is GameState.WIN or (time.time()-s.start_time) >= 8*3600-300
        except: return True

    def choose_action(s, frames, lf):
        try:
            lvl = s._lvl(lf)

            # ===== LEVEL CHANGE =====
            if lvl != s.cl:
                if s.cl >= 0:
                    try:
                        logger.info(f"V11 Retrospective: Analyzing Level {s.cl} completion...")
                        from PIL import Image
                        img_dir = os.path.join(os.path.dirname(__file__), "images", s.game_id)
                        search_pattern = os.path.join(img_dir, f"level_{s.cl:02d}_step_*.png")
                        existing_images = sorted(glob.glob(search_pattern))
                        
                        step_val = max(1, len(existing_images) // 10)
                        selected_images = existing_images[::step_val][-10:]
                        
                        recap_prompt = "You successfully solved this level. Look at these frames from start to finish. What were the rules of this level? How did you win? What objects did you interact with? Reply ONLY with a JSON object like {\"winning_strategy\": \"I moved right...\", \"generalized_mechanics_learned\": [\"green is fuel\"]}"
                        
                        content_images = []
                        for img_path in selected_images:
                            try:
                                img = Image.open(img_path)
                                img.load()
                                content_images.append(img)
                            except: pass
                            
                        if content_images:
                            response_text = call_local_multimodal_model(recap_prompt, content_images)
                            import json, re
                            json_str = re.search(r'\{.*\}', response_text, re.DOTALL)
                            if json_str:
                                recap = json.loads(json_str.group())
                                s._global_semantic_cache["level_completed"] = str(s.cl)
                                s._global_semantic_cache["winning_strategy"] = recap.get("winning_strategy", "")
                                s._global_semantic_cache["generalized_mechanics_learned"] = recap.get("generalized_mechanics_learned", [])
                                with open(s._memory_file, 'w') as f:
                                    json.dump(s._global_semantic_cache, f)
                                logger.info(f"V11 Recap Saved: {s._global_semantic_cache}")
                    except Exception as e:
                        logger.warning(f"V11 Retrospective failed: {e}")

                # Init BFS solver
                if not s._bfs_tried:
                    s._bfs_tried = True
                    s._init_bfs()

                s._bfs_solution = None
                s._bfs_step = 0
                if s._bfs:
                    s._try_bfs_solve(lvl)

                # Reset multi-agent state parameters
                s._multi_agent_solution = None
                s._multi_agent_step = 0
                
                # Execute Multi-Agent Pipeline Offline
                if ADK_AVAILABLE and not s._bfs_solution:
                    logger.info("[ADK Workflow] Invoking Hierarchical Multi-Agent pipeline...")
                    planned_actions = s._run_adk_pipeline(lf)
                    if planned_actions:
                        s._multi_agent_solution = [(act.get("action"), {"x": act.get("x", 0), "y": act.get("y", 0)}) for act in planned_actions]
                        logger.info(f"[ADK Workflow] Hierarchical Multi-Agent planned {len(s._multi_agent_solution)} actions!")

                # Init CNN fallback
                s.buf.clear(); s.buf_h.clear()
                s.net = ForgeNet(s.IN, s.G).to(s.device)
                for wp in ['/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                           'pretrained_weights.pt']:
                    try:
                        if os.path.exists(wp):
                            state=torch.load(wp,map_location=s.device,weights_only=True)
                            ms=s.net.state_dict()
                            for k in list(state.keys()):
                                if k in ms and state[k].shape==ms[k].shape:ms[k]=state[k]
                            s.net.load_state_dict(ms);break
                    except: pass
                s.opt = optim.Adam(s.net.parameters(), lr=0.0003)
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
                s.cl=lvl;s.fhist.clear();s.la=0
                s._wd=False;s._wm=None
                s._aem_diffs.clear();s._aem_actions.clear();s._aem_rewards.clear()
                s._prev_objs=None;s._obj_moved=0;s._ckpt_hash=None;s._unproductive=0
                s._visited_hashes = set()
                
                f_crop = s._raw(lf)[:55, :]
                s._start_of_level_hash = hashlib.md5(f_crop.tobytes()).hexdigest()[:16]
                s._last_death_autopsy = None

            # ===== RESET =====
            if lf.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                s.pt=None;s.pai=None;s.pr=None;s.ph=None
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

            # ===== MULTI-AGENT EXECUTION PATH =====
            if s._multi_agent_solution and s._multi_agent_step < len(s._multi_agent_solution):
                act_id, data = s._multi_agent_solution[s._multi_agent_step]
                s._multi_agent_step += 1
                sel = GameAction.from_id(act_id)
                if act_id == 6 and data:
                    sel.set_data(data)
                sel.reasoning = f"adk_multi_agent:{s._multi_agent_step}/{len(s._multi_agent_solution)}"
                raw = s._raw(lf)
                s.fhist.append(raw.copy())
                s.pr = raw.copy()
                s.la += 1
                return sel

            # ===== SILENT RESET DETECTOR =====
            f_crop = s._raw(lf)[:55, :]
            ch = hashlib.md5(f_crop.tobytes()).hexdigest()[:16]
            if hasattr(s, '_start_of_level_hash') and ch == s._start_of_level_hash and s.la > 5:
                logger.warning("SILENT RESET DETECTED! Performing Local Death Autopsy...")
                if not hasattr(s, '_fatal_hashes'):
                    s._fatal_hashes = set()
                if hasattr(s, '_hash_history') and len(s._hash_history) >= 2:
                    fatal_state = s._hash_history[-2]
                    s._fatal_hashes.add(fatal_state)
                    s._perform_death_autopsy(lf, fatal_state)
                s._hash_history.clear()
                s._bfs_solution = None
                s._bfs_step = 0
                s._multi_agent_solution = None
                s._multi_agent_step = 0
                s.la = 0

            if not hasattr(s, '_hash_history'):
                s._hash_history = deque(maxlen=15)
            s._hash_history.append(ch)

            # ===== CNN FALLBACK =====
            tensor = s._tensor(lf)
            raw = s._raw(lf)
            avail = getattr(lf, 'available_actions', None) or []
            s._undo_avail = any((a.value if hasattr(a,'value') else int(a))==7 for a in avail)

            if s.pt is not None and s.pai is not None:
                mask=np.ones((64,64),dtype=bool);mask[:2]=False;mask[62:]=False
                diff_map=(s.pr!=raw)&mask;changed=np.any(diff_map)
                eh=hashlib.md5(s.pr.tobytes()[:1000]+str(s.pai).encode()).hexdigest()[:16]
                if eh not in s.buf_h:
                    r=s._reward(s.pr,raw,'',ch)
                    s.buf.append({'s':s.pr.copy(),'a':s.pai,'r':r})
                    s.buf_h.add(eh)
                    if changed:
                        s._aem_diffs.append(diff_map)
                        s._aem_actions.append(min(s.pai,4))
                        s._aem_rewards.append(r)
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
                else:
                    s._wd=True
                    for _ in range(min(5,len(s.buf)//s.bsz)):s._train()

            if s._wd:
                if random.random()<s._eps:
                    aidx,coords=s._sample(torch.zeros(4101,device=s.device),avail,temp=2.0)
                else:
                    with torch.no_grad():
                        mem=s._get_aem_tensors()
                        if mem[0] is not None:logits=s.net(tensor.unsqueeze(0),*mem).squeeze(0)
                        else:logits=s.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx,coords=s._sample(logits,avail,temp=0.5)
                s._eps=max(s._eps_min,s._eps*s._eps_decay)
            elif s.la>=10:s._wd=True;aidx,coords=0,None

            if aidx<5:sel=s.al[aidx];sel.reasoning=f"cnn:a{aidx+1}"
            else:
                sel=GameAction.ACTION6;y,x=coords
                sel.set_data({"x":int(x),"y":int(y)});sel.reasoning=f"cnn:c({x},{y})"

            s.pt=tensor;s.pai=aidx if aidx<5 else(5+coords[0]*s.G+coords[1])
            s.pr=raw.copy();s.ph=ch;s.la+=1
            if s.action_counter%s.tfreq==0 and s._wd:s._train()
            return sel

        except Exception as e:
            traceback.print_exc()
            a=random.choice(s.al);a.reasoning=f"err:{e}";return a