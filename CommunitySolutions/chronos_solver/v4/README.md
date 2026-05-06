# Chronos Solver - v4: Multimodal Episodic Memory & Semantic Search

This version (v4) of the Chronos solver introduces cognitive persistence and semantic state extraction. Instead of acting as an amnesiac pathfinder that falls into the same traps upon a level reset, the v4 agent mimics human learning by explicitly tracking UI states and maintaining cross-episode memory.

## Improvements on v4 compared to v3

In v3, the agent acted as a stateless, amnesiac pathfinder. It utilized Vision solely for `(x, y)` goal coordinates, lacking the ability to track game-specific semantics like health, fuel, or deadly traps, which led to identical fatal mistakes upon a level reset. 

Here are the key improvements introduced in **v4**:

### 1. Episodic Memory Buffer (EMB)
- **v3 Flaw**: When the game reset (due to running out of lives or hitting a fatal wall/enemy), the A* planner forgot the exploration tree and repeated the exact same fatal trajectory.
- **v4 Improvement**: Added an episodic memory buffer (`_fatal_hashes`) that records the exact state hash right before a `GAME_OVER`. During the A* Swarm phase, any branch encountering a known fatal state is immediately discarded (masked out).

### 2. Multimodal Semantic Extraction (Vision LLM)
- **v3 Flaw**: The Gemini API prompt only requested the `(x, y)` coordinate of the ultimate objective. It ignored vital HUD concepts like fuel and lives.
- **v4 Improvement**: Upgraded the `gemini-3.1-pro-preview` prompt to actively extract a structured JSON schema containing:
  - `Lives Remaining`
  - `Fuel Level`
  - `Target Shape/Color Mapping`
  This transforms the agent from doing blind pathfinding to informed semantic navigation.

### 3. Cross-Level Concept Storage
- **v3 Flaw**: The agent completely flushed its cache between levels, having to re-learn basic concepts (like what object is a goal) from scratch.
- **v4 Improvement**: Introduced `_global_semantic_cache`. Global truths (like a parsed `target_shape`) are cached and explicitly passed back into the Gemini Vision prompt in subsequent levels as few-shot context, enabling true human-like continuous learning.

### 4. Hardcoded Level Step Limits
- **v3 Flaw**: The execution loop allowed unbounded exploration, leading to thousands of steps and timeouts on harder levels.
- **v4 Improvement**: Hardcoded a strict maximum of **200 steps per level**. The `play_game.py` execution loop tracks `level_step_count` and will forcibly terminate the run if the agent enters an unproductive infinite loop exceeding this threshold.

## How to Run
Ensure you have the virtual environment activated, and your `GEMINI_API_KEY` set in the `.env` file within this directory.

```bash
# Ensure you are using the venv where google-generativeai is installed
python CommunitySolutions/chronos_solver/v4/play_game.py
```
