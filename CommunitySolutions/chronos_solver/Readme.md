# Chronos Solver

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreyasmahimkar/arc-agi-3/blob/main/CommunitySolutions/chronos_solver/colab_runner.ipynb)

This folder contains the `chronos_solver` solution, representing FORGE v10 (Dynamic State Probing + Adaptive Search). It dynamically discovers hidden scalar fields per game and uses smart state hashing for a 5x speedup over standard pickle approaches.

## Contents
- `my_agent_from_kaggle.py`: The original code downloaded from the Kaggle competition notebook.
- `my_agent.py`: A copy of the Kaggle code with added explainable comments and no logic changes.
- `play_game.py`: A local execution harness that loads the `MyAgent` class and plays an ARC-AGI environment game (e.g., `ls20`) step-by-step so a human can observe the agent's progress.

## How to Run

### In Google Colab
You can run this agent directly in Google Colab without any local setup:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shreyasmahimkar/arc-agi-3/blob/main/CommunitySolutions/chronos_solver/colab_runner.ipynb)

### Local Execution

1. Ensure your virtual environment is activated and the ARC-AGI-3-Agents dependencies are installed.
2. Run the `play_game.py` script:

```bash
cd /Users/shreyas/gitrepos/OpenSource/kaggle/arc3
.venv312/bin/python CommunitySolutions/chronos_solver/play_game.py
```

3. The console will print the environment loading process and print out each `STEP` as the agent decides on actions. Wait for the game loop to finish or press `Ctrl+C` to stop.
