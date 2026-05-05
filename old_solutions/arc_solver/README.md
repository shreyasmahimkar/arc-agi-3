# ARC-AGI-3 Solver

A multi-agent reasoning system using LLMs for ARC-AGI-3.

## Execution Guide

The system uses a strategy pattern to swap out the underlying LLM provider based on the `ENV` environment variable.

### 1. Cloud Mode (API Key Required)
Run the orchestrator using the official Gemini Cloud API. This utilizes the massive reasoning capabilities of Gemini to genuinely write and test the `LocalSimulator.py` Numba code. You **must** provide a valid API key for this to work.

```bash
# Run from the root workspace directory
GEMINI_API_KEY="YOUR_KEY" ENV=DEV PYTHONPATH=arc_solver .venv312/bin/python arc_solver/arc_solver/kaggle_orchestrator.py
```

### 2. Local/Kaggle Mode (No API Key Required)
In KAGGLE mode, the orchestrator connects to a local open-source vLLM server running on `http://localhost:8000`. If the server is not running, it automatically returns a mocked `LocalSimulator.py`.

```bash
# Run from the root workspace directory
ENV=KAGGLE PYTHONPATH=arc_solver .venv312/bin/python arc_solver/arc_solver/kaggle_orchestrator.py
```
