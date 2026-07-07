# =====================================================================
# v21 llm_backend — one pluggable LLM interface for BOTH code-writers:
#   • runtime writer (offline, Kaggle T4)  -> HFBackend (local Qwen2.5-Coder)
#   • evolve writer   (cadence box)        -> HFBackend big / OpenAIBackend
#   • tests / no-GPU                        -> MockBackend (deterministic, offline)
#
# Selection (env V21_LLM_BACKEND): auto | hf | openai | mock
#   auto = hf if transformers+model load, else mock. OpenAIBackend is NEVER used
#   in the Kaggle/offline path (it needs network; the offline guard blocks it).
#
# Model (env V21_LLM_MODEL): default Qwen/Qwen2.5-Coder-7B-Instruct  (~6GB @4bit,
#   fits a 16GB T4). Set to Qwen/Qwen2.5-Coder-1.5B-Instruct for tight memory,
#   or Qwen/Qwen3-Coder-Next on the cadence box.
# =====================================================================
import os, json, logging, textwrap

logger = logging.getLogger("v21.llm")
DEFAULT_MODEL = os.environ.get("V21_LLM_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")


class LLMBackend:
    name = "base"
    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.2, stop=None):
        raise NotImplementedError
    def available(self):
        return True


class HFBackend(LLMBackend):
    """Local transformers model. 4-bit if bitsandbytes present. Fully offline once
    the weights are cached/bundled (set HF_HUB_OFFLINE=1 on Kaggle)."""
    name = "hf"
    def __init__(self, model_id=DEFAULT_MODEL, four_bit=True):
        self.model_id, self.four_bit = model_id, four_bit
        self._tok = self._model = None
    def available(self):
        try:
            import torch, transformers  # noqa
            return True
        except Exception:
            return False
    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        kw = {"torch_dtype": "auto", "device_map": "auto"}
        if self.four_bit:
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4")
            except Exception as e:
                logger.warning("4-bit unavailable (%s); loading full precision", e)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kw)
        logger.info("HFBackend loaded %s", self.model_id)
    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.2, stop=None):
        self._load()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        text = self._tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self._tok(text, return_tensors="pt").to(self._model.device)
        out = self._model.generate(**inp, max_new_tokens=max_tokens,
                                    do_sample=temperature > 0, temperature=max(temperature, 1e-3),
                                    pad_token_id=self._tok.eos_token_id)
        gen = self._tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if stop:
            for s in stop:
                gen = gen.split(s)[0]
        return gen


class OpenAIBackend(LLMBackend):
    """Cadence-box only. Needs network + OPENAI_API_KEY. Never in the offline path."""
    name = "openai"
    def __init__(self, model=os.environ.get("V21_OPENAI_MODEL", "gpt-5.5")):
        self.model = model
    def available(self):
        return bool(os.environ.get("OPENAI_API_KEY")) and _importable("openai")
    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.2, stop=None):
        from openai import OpenAI
        cli = OpenAI()
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        r = cli.chat.completions.create(model=self.model, messages=msgs,
                                        max_tokens=max_tokens, temperature=temperature, stop=stop)
        return r.choices[0].message.content


class OllamaBackend(LLMBackend):
    """MacBook / any-machine local backend via Ollama (http://localhost:11434).
    Stdlib-only (urllib) — no extra pip deps. Runs Qwen2.5-Coder on Apple Silicon
    via Metal. This is the recommended cadence-box backend on a Mac.
        brew install ollama && ollama serve
        ollama pull qwen2.5-coder:7b
        export V21_LLM_BACKEND=ollama V21_OLLAMA_MODEL=qwen2.5-coder:7b
    """
    name = "ollama"
    def __init__(self, model=None, host=None):
        self.model = model or os.environ.get("V21_OLLAMA_MODEL", "qwen2.5-coder:7b")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
    def _tags(self):
        import urllib.request
        with urllib.request.urlopen(self.host + "/api/tags", timeout=3) as r:
            return [m.get("name", "") for m in json.loads(r.read()).get("models", [])]
    def available(self):
        try:
            tags = self._tags()
        except Exception:
            return False
        # server up AND the requested model actually pulled (":latest" tolerant)
        return any(t == self.model or t.split(":")[0] == self.model.split(":")[0] for t in tags)
    @staticmethod
    def _deadline():
        """Hard wall-clock cap (seconds) for ONE completion. A stalled/swapping
        Ollama (e.g. a 7B model thrashing on a 16GB Mac) can hold an HTTP
        connection open indefinitely — urllib's `timeout` is a per-socket-op
        inactivity timeout, NOT a total deadline, so it never trips while the
        server dribbles keepalives. This cap is enforced in a watchdog thread so
        the runtime coder ALWAYS returns control to the cadence (which then falls
        back to safety-net plans and moves on to the next game). Env-tunable."""
        try:
            return max(5.0, float(os.environ.get("V21_OLLAMA_DEADLINE", "180")))
        except Exception:
            return 180.0

    def _complete_raw(self, prompt, system, max_tokens, temperature, stop, sock_timeout):
        import urllib.request, urllib.error
        msgs = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens}}
        if stop:
            body["options"]["stop"] = stop
        req = urllib.request.Request(self.host + "/api/chat",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=sock_timeout) as r:
                return json.loads(r.read())["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                try:
                    have = self._tags()
                except Exception:
                    have = []
                raise RuntimeError(
                    f"Ollama 404 for model '{self.model}'. Pulled models: {have or 'none'}. "
                    f"Run:  ollama pull {self.model}") from e
            raise

    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.2, stop=None):
        """Run the Ollama request under a hard wall-clock deadline in a watchdog
        thread. On timeout we RAISE (never hang) so the caller's except-branch
        degrades gracefully; the abandoned request finishes/dies in its daemon
        thread without blocking the cadence."""
        import threading
        deadline = self._deadline()
        result, error = [], []

        def _work():
            try:
                result.append(self._complete_raw(
                    prompt, system, max_tokens, temperature, stop, sock_timeout=deadline))
            except BaseException as e:  # noqa: BLE001 — surfaced to the main thread
                error.append(e)

        t = threading.Thread(target=_work, name="ollama-complete", daemon=True)
        t.start()
        t.join(deadline)
        if t.is_alive():
            raise RuntimeError(
                f"Ollama completion exceeded hard deadline {deadline:.0f}s "
                f"(model '{self.model}' likely OOM/swapping). Skipping this coder "
                f"attempt; set V21_OLLAMA_DEADLINE to tune or use a smaller model.")
        if error:
            raise error[0]
        return result[0] if result else ""


class MockBackend(LLMBackend):
    """Deterministic, offline. Emits a RUNNABLE executable world model that encodes
    the blind/repeat/click strategies the research proved solve the public set — so
    the whole harness (synthesize->verify->plan) is testable with no GPU/model.
    A real HFBackend replaces this on the cadence box / Kaggle GPU."""
    name = "mock"
    def complete(self, prompt, system=None, max_tokens=1024, temperature=0.2, stop=None):
        # crude intent routing off the prompt so tests exercise real code paths
        if "world model" in prompt.lower() or "WorldModel" in prompt:
            return _MOCK_WORLD_MODEL
        if "propose" in prompt.lower() or "challenger" in prompt.lower():
            return json.dumps({"blitz_K": 200, "action_order": [2, 1, 6, 3, 4, 5, 7],
                               "note": "mock: raise repeat budget, try ACTION2 first (ls20)"})
        return "{}"


_MOCK_WORLD_MODEL = textwrap.dedent('''
    # executable world model (mock): tries the research-proven trivial strategies.
    import numpy as np
    class WorldModel:
        """Encodes candidate hypotheses; plan() returns an action sequence to try."""
        def __init__(self, observations):
            self.obs = observations
        def candidate_plans(self, max_len=200):
            # depth-1 blind (each action once), repeat-one-action, then clicks
            plans = [[(a, None)] for a in (6, 1, 2, 3, 4, 5, 7)]
            for a in (1, 2, 6):
                plans.append([(a, None)] * max_len)
            return plans
        def predict(self, state, action):
            return state  # mock: identity (real model learns transitions)
''').strip()


def _importable(mod):
    try:
        __import__(mod); return True
    except Exception:
        return False


def get_backend(name=None):
    name = (name or os.environ.get("V21_LLM_BACKEND", "auto")).lower()
    if name == "ollama":
        return OllamaBackend()
    if name == "hf":
        return HFBackend()
    if name == "openai":
        return OpenAIBackend()
    if name == "mock":
        return MockBackend()
    # auto: prefer a running Ollama (Mac-friendly) -> local transformers -> mock
    olla = OllamaBackend()
    if olla.available():
        logger.info("llm auto -> OllamaBackend(%s)", olla.model); return olla
    hf = HFBackend()
    if hf.available():
        logger.info("llm auto -> HFBackend(%s)", hf.model_id); return hf
    logger.info("llm auto -> MockBackend (no ollama/transformers)"); return MockBackend()
