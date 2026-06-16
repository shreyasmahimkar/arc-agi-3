# Chronos Solver - v5: Reactive Vision & Silent Reset Detection

Based on the performance of v4, we identified key areas where the agent's cognitive loops fell out of sync with the actual game engine mechanics. v5 aims to solve these discrepancies by introducing real-time continuous vision and robust death-detection mechanisms.

## Flaws Identified in v4

1. **Infrequent Vision Feedback**: v4 only consulted the Gemini Vision model at the very beginning of a level or when completely stuck. It missed mid-game dynamic changes and couldn't be "coached" through a puzzle.
2. **Silent Resets (Uncaught Deaths)**: The Episodic Memory Buffer (EMB) in v4 assumed that losing a life would trigger a `GameState.GAME_OVER`. However, the environment handles traps with "silent resets"—the agent loses a visual life icon and is instantly warped back to the spawn point, but the API state remains `NOT_FINISHED`. Because the API state didn't change to `GAME_OVER`, v4 never realized it died, failed to record the fatal trajectory in `_fatal_hashes`, and blindly repeated the exact same path.
3. **Lack of Continuous Monologue**: The agent lacked a running context window of its recent gameplay. It couldn't ask Gemini "Am I making progress?" because there was no active session tracking the step-by-step logic.

## Implementation Plan for v5

### 1. The "Silent Reset" Detector
- **Problem**: The agent teleports back to spawn upon hitting a trap, but the game state doesn't flag it as a death.
- **Solution**: We will implement a coordinate-distance check. The agent will track its `(x, y)` position at every step. If the distance between `Step N` and `Step N-1` is mathematically impossible for a single move (e.g., jumping 20+ tiles instantly), the agent will explicitly flag this as a **Silent Reset / Death**. 
- **Action**: When a Silent Reset is detected, the agent will instantly trigger the EMB logic, grab the `_hash_history` from *right before* the teleport, log it to `_fatal_hashes`, and flush the current A* swarm queue to force a reroute.

### 2. Continuous Vision Polling (The "Coaching" Session)
- **Problem**: Gemini is underutilized, acting only as an initial GPS.
- **Solution**: Implement a periodic polling interval. Every `N` steps (e.g., every 5 or 10 steps), or immediately following a "Silent Reset", the agent will capture the current frame and send it to Gemini.
- **Action**: The prompt will be structured as a continuing session: *"You previously told me to head to (12, 38). I am currently at (15, 40). Did I lose a life? Has the environment changed? Where should I go next?"* This allows the LLM to actively coach the A* planner in near-real-time.

### 3. Active Puzzle Memory (State Monologue)
- **Problem**: The agent struggles to piece together complex sequences (e.g., hit switch A to open door B).
- **Solution**: Maintain a `_puzzle_monologue` list. Every time Gemini is polled, its semantic response is appended to this list. This running log is injected into the A* heuristics, ensuring the agent retains a short-term memory of recent structural changes in the puzzle.

### 4. Long-Term Game Intuition (Memory Persistence)
- **Problem**: In `v4`, cross-level learning relies on the `_global_semantic_cache` to store truths like goal shapes. However, this is implemented as an in-memory instance variable (`self._global_semantic_cache`). When the agent terminates or receives a completely new game, this cache is wiped. It does not learn "generally" across different game sessions because the intuition is never written to disk.
- **Solution (Proposed)**: To avoid starting from scratch on new games while preventing overfitting, the framework needs to externalize this long-term memory (e.g., saving successful heuristics or semantic mappings to a persistent JSON/Vector store). Currently, no persistent file-based long-term memory is implemented.



```bash
source .venv312/bin/activate
python CommunitySolutions/chronos_solver/v5/play_game.py
```
