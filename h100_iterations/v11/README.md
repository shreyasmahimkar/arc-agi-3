# ARC-AGI-3 Agent — v11: `PRISM` (Policy-Reasoned Interactive Solver via Multimodal Actions)

> **Vision-Language-Action architecture powered by a Gemma 3 backbone, a Pi0-style flow-matching action expert, and Ray for distributed rollout + RL post-training.**

---

## What Changed From v10 → v11

| Dimension | v10 (FORGE v21 lineage) | v11 `PRISM` |
|---|---|---|
| **Backbone** | PaliGemma 3B (frozen) | Gemma 3 27B (interleaved attention, 128K ctx) |
| **Action decoder** | Single linear projection head | Pi0-style flow-matching action expert (300M param transformer) |
| **Training** | Offline SFT on recorded trajectories | 3-stage: Pretraining → SFT → RL Interactive Post-Training (RIPT) |
| **Compute orchestration** | Single-GPU, sequential rollouts | Ray cluster: decoupled CPU data workers + GPU trainer actors |
| **Reward signal** | None (no RL) | Sparse binary: 1.0 on exact grid match, 0.0 otherwise |
| **Action space** | Discrete (argmax over pixels) | Continuous trajectory chunk → quantized grid coordinates |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         PRISM Model                             │
│                                                                 │
│   ┌─────────────────────────────────┐                           │
│   │        Gemma 3 27B Backbone     │  ← Vision (SigLIP) +      │
│   │  (interleaved local/global attn)│    Language instruction    │
│   └───────────────┬─────────────────┘                           │
│                   │  semantic token sequence                     │
│                   ▼                                             │
│   ┌─────────────────────────────────┐                           │
│   │     Pi0-style Action Expert     │  ← flow-matching decoder  │
│   │        (300M transformer)       │    conditioned on tokens   │
│   └────────┬───────────────┬────────┘                           │
│            │               │                                    │
│            ▼               ▼                                    │
│   ┌───────────────┐ ┌──────────────┐                            │
│   │ Coordinate    │ │ Action-Type  │                            │
│   │ Head [B,H,2]  │ │ Head [B,H,4] │                            │
│   │  (Sigmoid)    │ │  (Softmax)   │                            │
│   └───────────────┘ └──────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
         │                   │
         ▼                   ▼
   grid_x, grid_y      action_type
   (denormalized)       [Move|Click|Fill|Clear]
```

---

## Why Each Technology

### 1. Gemma 3 as the Backbone

The original `ArcPi0Wrapper` uses **PaliGemma** — a vision-language model built around a Gemma 1/2 decoder. v11 upgrades the backbone to **Gemma 3 27B** for the following reasons:

- **Native multimodal support**: Gemma 3 processes images and text in a single unified forward pass using its SigLIP encoder + interleaved attention. No separate VLM wrapper is needed.
- **128K context window**: ARC-AGI levels involve showing multiple input/output *example* pairs before the test grid. Gemma 3's long context allows the model to keep all few-shot demonstrations in a single pass without truncation.
- **Interleaved local/global attention (5:1)**: Most of the transformer layers use sliding-window local attention (efficient), with one global attention layer every 5. This massively reduces KV-cache size, making long-context ARC inference tractable on a single H100 without quantization.
- **Pan&Scan image processing**: Gemma 3's variable-resolution image patching allows the full ARC grid to be rendered at high resolution and attended to patch-by-patch — critical for perceiving fine-grained pixel colors.

```python
# v11 backbone swap (replaces PI0Policy.from_pretrained)
from transformers import Gemma3ForConditionalGeneration, AutoProcessor

backbone = Gemma3ForConditionalGeneration.from_pretrained(
    "google/gemma-3-27b-it",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("google/gemma-3-27b-it")
```

---

### 2. Pi0-Style Flow-Matching Action Expert

The Pi0 wrapper you wrote projects a *continuous hidden state* from the VLM into grid coordinates and action types. v11 keeps this design but formalizes it into a proper **flow-matching action expert** — the key innovation of Physical Intelligence's π₀ architecture:

- **What flow matching does**: Instead of predicting coordinates directly, the action expert learns a vector field that "flows" a noise sample toward a trajectory distribution. This produces smooth, multi-step action *chunks* (e.g., 10 grid moves) in a single forward pass, avoiding the accumulation errors of autoregressive step-by-step prediction.
- **Why this matters for ARC**: An ARC-AGI solution is often a *sequence of operations* (e.g., "select cell (2,3), fill blue, move to (4,1), fill blue..."). Predicting a chunk at once is far more coherent than predicting one action and then re-attending.
- **Blockwise causal attention**: The Gemma 3 tokens and action expert tokens attend to each other via cross-attention, while action tokens are causally masked relative to each other. This preserves temporal order in the chunk.

```python
class ArcActionExpert(nn.Module):
    """
    Pi0-style flow-matching action expert for ARC grid operations.
    Conditions on Gemma 3 semantic tokens and denoises a trajectory from Gaussian noise.
    """
    def __init__(self, hidden_dim: int, horizon: int = 10, n_layers: int = 12, n_heads: int = 8):
        super().__init__()
        self.horizon = horizon
        # Noise embedding for the flow-matching timestep t ∈ [0, 1]
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)
        )
        # Cross-attention from action tokens → Gemma 3 semantic tokens
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=n_heads, batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        # Project to action dimensionality (x, y, action_type_logits=4)
        self.out_proj = nn.Linear(hidden_dim, 2 + 4)

    def forward(self, semantic_tokens: torch.Tensor, noisy_actions: torch.Tensor, t: torch.Tensor):
        """
        Args:
            semantic_tokens: [B, S, D]  ← Gemma 3 output
            noisy_actions:   [B, H, D]  ← noisy trajectory at timestep t
            t:               [B, 1]     ← flow-matching time in [0, 1]
        Returns:
            velocity field:  [B, H, 6]  ← predicted (dx, dy, action_logits)
        """
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1).expand(-1, self.horizon, -1)
        action_tokens = noisy_actions + t_emb
        out = self.transformer(tgt=action_tokens, memory=semantic_tokens)
        return self.out_proj(out)

    @torch.no_grad()
    def sample(self, semantic_tokens: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """ODE integration: flow noise → trajectory using 'n_steps' Euler steps."""
        B, H, D = semantic_tokens.size(0), self.horizon, semantic_tokens.size(-1)
        x = torch.randn(B, H, D, device=semantic_tokens.device)
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((B, 1), i * dt, device=x.device)
            v = self.forward(semantic_tokens, x, t)
            x = x + dt * v[..., :D]  # Euler step on hidden space
        return self.out_proj(x)      # Final projection → [B, H, 6]
```

---

### 3. Ray for Distributed Rollout and RL Post-Training

Previous versions ran a single agent sequentially through levels. v11 introduces **Ray** as the distributed compute backbone across three roles:

#### 3a. Ray Data — Streaming Episode Preprocessing

ARC episode data (grid images + action logs from previous versions) is stored as multi-file shards. Naive parallel loading is 15–135× less efficient than using Ray's file-group partitioning.

```python
import ray.data as rd

dataset = rd.read_parquet(
    "s3://arc-agi-v11/episodes/",
    override_num_blocks=256,        # One block per episode shard
).map_batches(
    preprocess_arc_episode,         # Decode grid PNG → tensor, tokenize instruction
    batch_size=32,
    num_cpus=4,                     # CPU-bound: image decode + tokenization
).filter(lambda row: row["solved"] == True)  # Train only on winning trajectories
```

#### 3b. Ray Train — Distributed SFT + RIPT

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Ray Cluster (v11)                             │
│                                                                      │
│  ┌──────────────┐   stream   ┌──────────────────────────────────┐   │
│  │  CPU Workers  │ ────────► │     GPU Trainer Workers (DDP)    │   │
│  │  (Ray Data)   │           │  ┌────────┐  ┌────────┐          │   │
│  │  • Grid decode│           │  │ GPU 0  │  │ GPU 1  │  ...     │   │
│  │  • Tokenize   │           │  │Gemma 3 │  │Gemma 3 │          │   │
│  │  • Augment    │           │  │+Expert │  │+Expert │          │   │
│  └──────────────┘           │  └────────┘  └────────┘          │   │
│                              └──────────────────────────────────┘   │
│                                            ▲                         │
│  ┌──────────────────────────┐              │ rollout rewards          │
│  │  Ray EnvRunner Actors    │  ────────────┘                        │
│  │  (Parallel ARC Env)      │   RIPT loop: PPO / GRPO               │
│  │  • Simulate grid ops     │                                        │
│  │  • Score: exact match?   │                                        │
│  └──────────────────────────┘                                        │
└──────────────────────────────────────────────────────────────────────┘
```

```python
from ray.train.torch import TorchTrainer
from ray.train import ScalingConfig

trainer = TorchTrainer(
    train_loop_per_worker=prism_train_loop,   # see Training section
    scaling_config=ScalingConfig(
        num_workers=8,
        use_gpu=True,
        resources_per_worker={"GPU": 1, "CPU": 4},
    ),
    datasets={"train": dataset},
)
result = trainer.fit()
```

#### 3c. Ray Serve — Online Inference Endpoint

After training, the `PRISM` policy is served as a stateless Ray Serve deployment for both Kaggle submission and local debugging:

```python
from ray import serve

@serve.deployment(num_replicas=2, ray_actor_options={"num_gpus": 1})
class PrismSolverDeployment:
    def __init__(self):
        self.model = PrismPolicy.from_checkpoint("s3://arc-agi-v11/checkpoints/best/")

    async def __call__(self, request: dict) -> list[dict]:
        grid = torch.tensor(request["grid"])
        actions = self.model.predict_action_chunk(grid, request["width"], request["height"])
        return actions.tolist()
```

---

## Three-Stage Training Pipeline

```
Stage 1: Pretraining (offline)
  └─ Dataset: LeRobot v3.0 format from all prior v1–v10 winning episodes
  └─ Loss: Flow-matching regression (MSE on velocity field)
  └─ Backbone: Gemma 3 frozen; Action Expert fully trainable

Stage 2: Supervised Fine-Tuning (SFT)
  └─ Dataset: Human-verified ARC-AGI-3 winning trajectories
  └─ Loss: Flow-matching + cross-entropy on action_type logits
  └─ Backbone: LoRA-finetuned (rank=16 on Q, V projections); Expert fully trainable

Stage 3: Reinforcement Interactive Post-Training (RIPT)
  └─ Environment: Parallel ARC simulator (Ray EnvRunner actors)
  └─ Reward: +1.0 exact output grid match, 0.0 otherwise
  └─ Algorithm: GRPO (Group Relative Policy Optimization)
  └─ Backbone + Expert: Both updated with constrained KL penalty vs. SFT checkpoint
```

> **Why GRPO over PPO?** GRPO computes relative rewards within a group of sampled trajectories for the same puzzle, avoiding the need for a separate critic (value network). This cuts memory by ~30% — critical when the backbone is a 27B model.

---

## Model: `ArcPrismPolicy` (v11 core class)

```python
class ArcPrismPolicy(nn.Module):
    """
    v11 PRISM: Gemma 3 backbone + Pi0-style flow-matching action expert.
    Replaces the PaliGemma + linear-head design from the original ArcPi0Wrapper.
    """
    def __init__(
        self,
        backbone_id: str = "google/gemma-3-27b-it",
        expert_hidden_dim: int = 1024,
        action_horizon: int = 10,
    ):
        super().__init__()
        self.horizon = action_horizon

        # Gemma 3 backbone (bfloat16, FlashAttention-2)
        self.backbone = Gemma3ForConditionalGeneration.from_pretrained(
            backbone_id, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(backbone_id)

        # LoRA adapter for SFT/RIPT fine-tuning (backbone stays mostly frozen)
        self.lora_config = LoraConfig(r=16, target_modules=["q_proj", "v_proj"])

        # Pi0-style flow-matching action expert
        self.action_expert = ArcActionExpert(
            hidden_dim=expert_hidden_dim,
            horizon=action_horizon,
        )

        # Project Gemma 3 hidden states → action expert dimension
        self.semantic_proj = nn.Linear(
            self.backbone.config.hidden_size, expert_hidden_dim
        )

        # Final coordinate and action type heads
        self.coord_head = nn.Sequential(
            nn.Linear(expert_hidden_dim, 256), nn.ReLU(),
            nn.Linear(256, 2), nn.Sigmoid(),   # [x, y] ∈ [0, 1]
        )
        self.action_type_head = nn.Linear(expert_hidden_dim, 4)

    def encode(self, images, instructions) -> torch.Tensor:
        """Gemma 3 forward pass → semantic token sequence."""
        inputs = self.processor(
            text=instructions, images=images,
            return_tensors="pt", padding=True
        ).to(next(self.backbone.parameters()).device)
        out = self.backbone(**inputs, output_hidden_states=True)
        # Pool last hidden layer tokens
        return out.hidden_states[-1]  # [B, S, D_gemma]

    def forward(self, images, instructions):
        semantic = self.encode(images, instructions)          # [B, S, D_gemma]
        semantic_proj = self.semantic_proj(semantic)          # [B, S, D_expert]

        # Sample trajectory from the flow-matching action expert
        traj = self.action_expert.sample(semantic_proj)       # [B, H, 6]

        # Split into coord features and action type logits
        coord_features = traj[..., :self.action_expert.horizon]
        action_logits  = traj[..., 2:]                        # [B, H, 4]

        # Project to final outputs
        coords = self.coord_head(semantic_proj.mean(dim=1).unsqueeze(1).expand(
            -1, self.horizon, -1))                            # [B, H, 2]
        return coords, action_logits

    @torch.no_grad()
    def predict_action_chunk(self, images, instructions, grid_width, grid_height):
        self.eval()
        coords, action_logits = self.forward(images, instructions)
        grid_x = (coords[..., 0] * (grid_width  - 1)).round().long()
        grid_y = (coords[..., 1] * (grid_height - 1)).round().long()
        action_types = torch.argmax(action_logits, dim=-1)
        return torch.stack([grid_x[0], grid_y[0], action_types[0]], dim=-1)
```

---

## Installation

```bash
# 1. Create environment
python3.12 -m venv .venv312 && source .venv312/bin/activate

# 2. Core dependencies
pip install torch>=2.3.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers>=4.49.0 accelerate peft
pip install lerobot>=0.3.0

# 3. Ray ecosystem
pip install "ray[train,serve,data]>=2.30.0"
pip install vllm>=0.4.0          # Optional: for fast Gemma 3 inference serving

# 4. ARC-specific
pip install -r environment_files/requirements_v11.txt
```

---

## Running Locally

### Single-GPU inference (Kaggle T4 / local RTX 3090+)

```bash
python play_game.py \
  --model-checkpoint checkpoints/prism_v11_sft.pt \
  --backbone google/gemma-3-27b-it \
  --horizon 10 \
  --n-flow-steps 10
```

### Distributed training on Ray cluster

```bash
# Start Ray head node
ray start --head --num-gpus=8

# Launch RIPT training job
python train_prism.py \
  --stage ript \
  --backbone google/gemma-3-27b-it \
  --lora-rank 16 \
  --horizon 10 \
  --rollout-workers 32 \
  --reward-fn exact_grid_match \
  --algorithm grpo \
  --num-episodes 50000
```

### Ray Serve deployment (local debug)

```bash
serve run serve_config.yaml
curl -X POST http://localhost:8000/solve \
  -H "Content-Type: application/json" \
  -d '{"grid": [[0,1],[1,0]], "width": 2, "height": 2}'
```

---

## File Structure (v11)

```
h100_iterations/v11/
├── README.md                        ← this file
├── prism_policy.py                  ← ArcPrismPolicy class (Gemma 3 + Action Expert)
├── action_expert.py                 ← ArcActionExpert (flow-matching transformer)
├── train_prism.py                   ← 3-stage training entrypoint (Ray Train)
├── serve_config.yaml                ← Ray Serve deployment config
├── play_game.py                     ← Local single-GPU inference harness
├── arc_env.py                       ← ARC simulator for RIPT rollouts (Ray EnvRunner)
├── reward_fns.py                    ← exact_grid_match + partial credit rewards
├── colab_runner.ipynb               ← Colab notebook (pulls repo, runs play_game.py)
├── shreyas-h100-arc-agi-3-v11.ipynb ← Kaggle submission notebook
└── requirements_v11.txt             ← Pinned dependencies
```

---

## Key Design Decisions & Trade-offs

| Decision | Rationale | Trade-off |
|---|---|---|
| Gemma 3 27B (not 4B) | Better spatial/visual reasoning on ARC grids | ~4× more VRAM; requires bfloat16 + FA2 |
| Flow matching (not diffusion) | Single forward pass per denoising step; exact ODE | Training is more sensitive to batch size |
| GRPO (not PPO) | No value network → ~30% memory saving | Reward variance can be high early in RIPT |
| LoRA rank=16 | Fine-tune backbone cheaply; preserve general vision | Lower expressivity than full fine-tune |
| Horizon=10 | 10-step action chunks reduce re-planning overhead | Longer chunk = harder credit assignment |
| Ray Serve for inference | Stateless, auto-scaling, decoupled from training | Adds ~150ms cold-start latency |

---

## References

- **π₀ Architecture**: Black et al., *"π₀: A Vision-Language-Action Flow Model for General Robot Control"*, Physical Intelligence (2024). [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
- **Gemma 3**: Google DeepMind, *"Gemma 3 Technical Report"* (March 2025). [ai.google.dev/gemma](https://ai.google.dev/gemma)
- **RIPT-VLA**: *"Reinforcement Interactive Post-Training for Vision-Language-Action Models"* (2025). [openreview.net](https://openreview.net)
- **Ray Distributed Training**: Anyscale, *"Scalable VLA Training with Ray Data + Ray Train"* (2025). [ray.io](https://ray.io)
- **LeRobot**: Hugging Face LeRobot v3.0. [github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)
- **ARC-AGI-3**: Kaggle Competition. [kaggle.com/competitions/arc-prize-2026](https://www.kaggle.com/competitions/arc-prize-2026)
