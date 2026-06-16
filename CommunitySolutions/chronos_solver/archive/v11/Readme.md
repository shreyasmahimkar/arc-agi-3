# V11 Multi-Agent Solver with Local Gemma 4

This directory contains the V11 Chronos Solver upgrade: a hierarchical multi-agent team integrated with the Google ADK Framework and local multimodal Gemma 4 inference backend, running 100% offline.

## V11 Multi-Agent Architecture

The agent logic is partitioned into a specialized 3-tier team:
- **Tier 1 (Root Orchestrator):** `ManagerAgent` - responsible for ingestion, short-term monologue history, and overall orchestration.
- **Tier 2 (Pillar Leads):**
  - `ExplorationLead` - Coordinates parallel exploration checker agents.
  - `ModelingLead` - Coordinates parallel transformation matcher agents.
  - `GoalSettingLead` - Identifies intermediate sub-goal target grids.
  - `PlanningLead` - Generates sequential moves and coordinates action compression.
- **Tier 3 (Sub-Agents):** 18 narrowly-scoped functional agents (e.g. symmetry detection, line drawing, object extraction, and python code simulation).

## Execution Environments

### 1. Running Locally on macOS (Metal Acceleration)

#### Step A: Setup Gemma 4 Multimodal 12B Model
Install Ollama from [ollama.com](https://ollama.com) and pull the gemma4 model:
```bash
ollama run gemma4:12b
```

#### Step B: Execute the Game Player
Run the player harness locally in the virtual environment:
```bash
source .venv312/bin/activate
python CommunitySolutions/chronos_solver/v11/play_game.py --game ls20
```

---

### 2. Running on Google Colab (A100 GPU Cloud)

Open and run the cells in [colab_runner.ipynb](file:///Users/shreyas/gitrepos/OpenSource/kaggle/arc3/CommunitySolutions/chronos_solver/v11/colab_runner.ipynb). It automatically:
1. Clones the repository.
2. Installs CUDA-enabled PyTorch, Google ADK, and other dependencies via `requirements_colab.txt`.
3. Installs Ollama, pulls the `gemma4:12b` model, and runs the service in the background on the cloud VM.
4. Executes the evaluation loop:
   ```bash
   !python CommunitySolutions/chronos_solver/v11/play_game.py --game ls20
   ```

---

## Logging & Tracing Framework

V11 features highly-detailed logging for offline inspection and feeding data to subsequent versions:
- **`v11_run.log`:** Main process log containing agent reasoning logs, step timings, and environment responses.
- **`images/{game_id}/`:** Contains step-by-step screenshots with an overlay coordinate grid for multimodal analysis.
- **`v11_long_term_memory.json`:** Persistent long-term semantic memory storing winning strategies and generalized mechanics learned across levels.
- **`v11_{game_id}_level_{level}_scratchpad_iteration_{iter}.json`:** Structured JSON scratchpads containing:
  - Visual grid dimensions
  - Manager analysis text
  - Sandbox validation traces (python simulation scripts, stdout/stderr, and outcomes)
  - Uncompressed proposed action paths
  - Compressed optimized action paths (merging redundant click coordinate sequences)
  - Multimodal death autopsy reports in case of silent resets.
