# V10 Local Testing with Hugging Face (Gemma 4)

This document outlines the plan for integrating the Gemma 4 (31B) model into the `chronos_solver` architecture for local testing, replacing the Gemini API.

## 1. Hardware Considerations
The Gemma 4 31B model is massive. Even with 4-bit quantization (NF4), loading the model requires approximately **18-22 GB of VRAM**. 
*   **Linux/Nvidia Setup**: You can use the standard `transformers` and `bitsandbytes` library with `BitsAndBytesConfig`.
*   **Mac/Apple Silicon (MPS)**: `bitsandbytes` 4-bit (NF4) quantization is heavily optimized for CUDA and does not natively support Apple Metal (MPS). To run this locally on a Mac, you will need to pivot to **MLX** (`mlx-vlm`) or **llama.cpp** to handle the quantization natively on Apple Unified Memory.

## 2. Option A: Local Python Inference (Transformers)
If running on a capable machine with a 24GB+ GPU (like an RTX 3090 or 4090):

### Dependencies
```bash
pip install transformers accelerate bitsandbytes torch torchvision
```

### Agent Modification (`my_agent.py`)
Instead of initializing `genai.Client`, initialize the local model inside `MyAgent.__init__` so it persists across levels:

```python
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

# Inside MyAgent.__init__:
model_id = "google/gemma-4-31b-it" 
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
self.model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
```

Inside `_get_multimodal_goal`, swap the API call for the local `.generate()`:
```python
messages = [
    {
        "role": "user", 
        "content": [
            {"type": "image"}, # Pass images iteratively
            {"type": "text", "text": prompt + " Explain your reasoning in detail inside a thinking block."}
        ]
    }
]

formatted_prompt = self.processor.apply_chat_template(
    messages, 
    tokenize=False, 
    add_generation_prompt=True,
    enable_thinking=True
)

# Convert PIL images and text to tensors
inputs = self.processor(text=formatted_prompt, images=images, return_tensors="pt").to(self.model.device)

outputs = self.model.generate(
    **inputs, 
    max_new_tokens=2048, 
    do_sample=True,
    temperature=0.7
)

response_text = self.processor.decode(outputs[0], skip_special_tokens=True)
```

## 3. Option B: Local API Server (vLLM / TGI)
If you want to keep `my_agent.py` clean without loading massive weights directly into the simulation loop, host Gemma locally using an OpenAI-compatible server:

1.  Run **vLLM** or **Hugging Face Text Generation Inference (TGI)** in a separate terminal.
2.  In `my_agent.py`, use the standard `openai` python client pointing to `http://localhost:8000/v1` to submit the images and prompt. This keeps the solver lightweight while offloading the heavy inference to a dedicated local endpoint.
