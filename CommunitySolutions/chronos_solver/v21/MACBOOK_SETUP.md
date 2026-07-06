# Running the v21 code-writer on your MacBook

Your Mac is the **cadence box** — it runs the 4-hour SOLVE+EVOLVE loop and the LLM
code-writer. Apple Silicon has no CUDA, so **ignore the bitsandbytes/4-bit path** (that's
only for Kaggle's NVIDIA T4). On a Mac use **Ollama** (easiest) or **MLX** (fastest).

## Option A — Ollama (recommended, 3 commands)

```bash
brew install ollama
ollama serve                       # leave running (or: `brew services start ollama`)
ollama pull qwen2.5-coder:7b       # ~4.7 GB; pick by RAM (table below)
```

Then point the cadence at it and run:

```bash
cd /Users/shreyas/gitrepos/OpenSource/kaggle/arc3/CommunitySolutions/chronos_solver/v21
export V21_LLM_BACKEND=ollama
export V21_OLLAMA_MODEL=qwen2.5-coder:7b
/Users/shreyas/gitrepos/OpenSource/kaggle/arc3/.venv312/bin/python \
    cadence_runner.py --bfs-timeout 180 --evolve --allow-network
```

`--allow-network` is needed so the runner can reach `localhost:11434` (the offline guard
blocks all sockets otherwise). This is the cadence box, so that's fine — the Kaggle
submission never uses this path.

Quick sanity check that the backend is wired:

```bash
V21_LLM_BACKEND=ollama .../python -c \
 "import llm_backend as lb; b=lb.get_backend(); print(b.name, b.available()); \
  print(b.complete('Say only: ok'))"
```

### Pick the model by your Mac's RAM
| Mac unified memory | Model (`ollama pull ...`) | Notes |
|---|---|---|
| 8 GB | `qwen2.5-coder:1.5b` | fits; weakest writer |
| 16 GB | `qwen2.5-coder:7b` | **default**, good balance |
| 24–32 GB | `qwen2.5-coder:14b` or `qwen2.5-coder:32b`(q4) | stronger challenger code |
| 48 GB+ | `qwen3-coder:30b` / `qwen3-coder-next` | top local coder (2026) |

Set `V21_OLLAMA_MODEL` to whichever you pulled. Bigger = better solver-evolution, slower.

## Option B — MLX (Apple-native, fastest on M-series)

```bash
pip install mlx-lm --break-system-packages
mlx_lm.server --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit --port 11434
```
`mlx_lm.server` speaks the OpenAI API. Use it via the `openai` backend pointed at localhost:
```bash
export V21_LLM_BACKEND=openai OPENAI_BASE_URL=http://localhost:11434/v1 OPENAI_API_KEY=x
```
(Ollama also runs MLX/Metal under the hood, so Option A already gets you GPU acceleration —
MLX is only worth it if you want to squeeze max tokens/sec.)

## What the LLM actually does here
1. **Runtime writer** (`runtime_coder.py`) — writes an executable `WorldModel` per level,
   sandbox-execs it, verifies vs frames, plans, commits the shortest win.
2. **Evolve writer** (`evolve.py`) — proposes challenger configs/heuristics for the wall
   levels each cycle; promoted only if it beats the champion on held-out.

No Ollama running? The runner auto-falls back to the **mock** backend: the intuition distill
and BFS/blitz stages still run; only the LLM code-writing is skipped. So it never breaks.

## Note on Kaggle (the other box)
Kaggle is offline — you can't call Ollama there. For the *submission*, bundle
`Qwen/Qwen2.5-Coder-7B-Instruct` as a Kaggle **dataset**, set `HF_HUB_OFFLINE=1`, and the
`hf` backend loads it 4-bit on the T4. The Mac (Ollama) and Kaggle (bundled HF) are two
different backends behind the same `llm_backend` interface.
