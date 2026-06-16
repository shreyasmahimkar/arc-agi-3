# V10 Kaggle Deployment (Offline Gemma 4 Execution)

This document outlines the plan for deploying the V10 Gemma-powered Chronos Solver to a Kaggle Notebook capable of running **completely offline**.

## 1. Architecture Map
The final Kaggle submission requires merging the multi-file architecture of V9 (`my_agent.py`, `play_game.py`) into a single monolithic Jupyter Notebook, similar to the `Ash_solver/v3/shreyas-m-arc-agi-agent-v3.ipynb` format.

## 2. Kaggle Offline Constraints
Kaggle evaluation environments **do not have internet access**. All models, tokenizers, and dependencies must be mounted as Kaggle Datasets.

### Dataset Mounting
1.  **Gemma Model**: Mount the official Kaggle Gemma model to `/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1`.
2.  **Pre-trained CNN Weights**: Mount your `ForgeNet` fallback weights to `/kaggle/input/forge-pretrained-weights/pretrained_weights.pt`.
3.  **Dependencies**: Any library not pre-installed on Kaggle (e.g., specific versions of `bitsandbytes` or `accelerate`) must be uploaded as a `.whl` dataset and installed via `!pip install --no-index --find-links /kaggle/input/my-wheels my-package`.

## 3. GPU Configuration
The Kaggle environment provides 2x T4 GPUs. The 31B parameter model requires 4-bit quantization to fit. 

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# 1. Define Model Path
model_id = "/kaggle/input/models/google/gemma-4/transformers/gemma-4-31b-it/1"

# 2. 4-bit Quantization (Required for Kaggle GPUs)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 3. Load Model with device_map="auto" to split across the 2x T4s
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
```

## 4. Merging the Solver Loop
In the notebook, define the classes sequentially:
1.  `GameAction` & Enums
2.  `ForgeNet` & `ActionEffectAttention`
3.  `BFSSolver`
4.  `MyAgent` (with the Gemma model passed into it as a reference to prevent reloading it per level).

### Global Model Loading
**CRITICAL**: Do *not* load the Gemma model inside `MyAgent.__init__` if `MyAgent` gets re-instantiated. Load the model once globally at the top of the notebook, and pass the `model` and `processor` references into the agent. 

## 5. Submission Execution Loop
At the bottom of the notebook, invoke the standard ARC-AGI test suite loop that initializes the environment, creates `MyAgent`, and iterates through the test set.

```python
agent = MyAgent(model=model, processor=processor)
# Iterate test tasks
for task in test_tasks:
    # Run the _get_multimodal_goal logic using the injected global model
    # Execute Swarm / BFS
    pass
```
