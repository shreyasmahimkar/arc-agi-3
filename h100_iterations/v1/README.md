# H100 Iterations - v1

This directory contains the modernized `FORGE v21` agent, optimized with Hopper-specific (H100) architectural upgrades including FlashAttention-2, mixed precision (bfloat16), asynchronous learning buffers, and TF32 operations.

## Contents
- `my_agent.py`: The newly optimized agent script.
- `play_game.py`: The local execution harness to run `my_agent.py` interactively.
- `colab_runner.ipynb`: A Colab notebook designed to automatically pull this repo from GitHub and run the agent using Colab's compute.
- `shreyas-h100-arc-agi-3-agent-v1.ipynb`: The final Kaggle submission notebook with the `my_agent.py` code embedded into the `%%writefile` block.

## How to use Google Colab (T4 / L4 Compute)

Because this agent uses optimizations tailored for modern GPUs, running it on Colab is highly recommended if you do not have a local Hopper/Ada GPU.

### Step 1: Push changes to GitHub
Ensure all your files are pushed to the `main` branch of your GitHub repository:
```bash
git add .
git commit -m "Update H100 agent files"
git push origin main
```

### Step 2: Open the Colab Runner
Click the magic link below to open `colab_runner.ipynb` directly from your repository in Google Colab:
👉 **[Open colab_runner.ipynb in Google Colab](https://colab.research.google.com/github/shreyasmahimkar/arc-agi-3/blob/main/h100_iterations/v1/colab_runner.ipynb)**

### Step 3: Connect to GPU and Execute
1. In Colab, go to **Runtime > Change runtime type**.
2. Select **T4 GPU** (or **L4 GPU** if you have Colab Pro/Pro+, which will truly unlock the H100 optimizations like bfloat16 and FlashAttention-2).
3. Click **Save**.
4. Finally, click **Runtime > Run all**.

The notebook will automatically download the repo, install the dependencies, and execute `play_game.py`.

## Kaggle Submission
You can upload the `shreyas-h100-arc-agi-3-agent-v1.ipynb` notebook directly to Kaggle. It is already properly formatted with the `%%writefile` and dummy submission fallbacks required by the competition environment.
