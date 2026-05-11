# Chronos Solver - v7: Post-Level Retrospectives & Cross-Level Context

While v6 successfully implemented multi-frame tracking and spatial chain-of-thought, an architectural flaw remains regarding how knowledge is transferred between levels. v7 focuses on solving the "Silent Solver" problem, ensuring that every victory is analyzed and documented for future use.

## Flaws Identified in v6

1. **Cross-Level Context Amnesia (The "Silent Solver" Problem)**:
   - **Observation**: If a level (like Level 0) is solved entirely by the programmatic BFS Search, the Gemini Vision model is never invoked. 
   - **The Flaw**: Because Gemini is never called during a purely programmatic victory, it never gets to "see" how the level was beaten. When Level 1 starts, the AI is completely blind to the mechanics, rules, or visual transformations that were required to beat Level 0. There is no holistic, cross-level context transfer because no visual hypotheses were ever evaluated or written to long-term memory.

2. **Coordinate Hallucination (Pixel vs. Grid Mismatch)**:
   - **Observation**: Gemini is outputting coordinates like `{"x": 170, "y": 80}`, despite the game grid being strictly 64x64.
   - **The Flaw**: Because Gemini analyzes a rendered `matplotlib` image (which is 400x400+ pixels), it hallucinates coordinates based on the image's raw pixel resolution rather than the underlying 64x64 grid index. This completely breaks the A* Swarm Planner's heuristic. When A* calculates distances to an out-of-bounds coordinate like (170, 80), the mathematical delta explodes, neutralizing the curiosity bonuses and causing the agent to get stuck in corners.

## Implementation Plan for v7

### 1. Post-Level Retrospective (The "Level Recap")
- **Solution**: Force Gemini to perform a visual "post-mortem" analysis every time a level is successfully completed, regardless of whether it was solved by the LLM or BFS.
- **Action**: 
  - When the framework detects a level transition (e.g., advancing from Level 0 to Level 1), the agent will programmatically scan the `images/` directory to gather a sequential batch of screenshots saved during the *winning trajectory* of the previous level (e.g., start state, key interaction state, and the final winning state).
  - It will upload these specific historical image files to Gemini alongside a dedicated "Retrospective Prompt": *"You successfully solved this level. Look at these frames from start to finish. What were the rules of this level? How did you win? What objects did you interact with?"*

### 2. Deep Context Injection
- **Solution**: Store this retrospective narrative in the long-term memory and inject it directly into the next level's initial prompt.
- **Action**: 
  - The framework will write or update `v7_long_term_memory.json`. Instead of a flat string, the memory file will enforce a highly detailed schema:
    ```json
    {
      "game_name": "EscapeMaze",
      "level_completed": "0",
      "winning_strategy": "The agent navigated around the red walls, collected the green cross to replenish fuel, and reached the dark red pattern.",
      "generalized_mechanics_learned": [
        "Red walls are fatal traps.",
        "Green crosses increase fuel.",
        "The objective is always the dark red pattern."
      ]
    }
    ```
  - When the next level begins, this comprehensive "Game Wiki" is injected directly into Gemini's initial prompt. This allows Gemini to instantly understand the environment constraints and utilize previous strategies to solve the current layout.

### 3. Grid-Overlay Coordinate Calibration
- **Solution**: Force Gemini to map its visual understanding to the strict 0-63 matrix constraints, and protect the A* planner from mathematical explosions.
- **Action**: 
  - Instead of passing raw images, the framework will render a subtle coordinate grid overlay or boundary markers onto the images.
  - The prompt will be explicitly fortified: *"CRITICAL: Output coordinates strictly as grid cell indices between 0 and 63. Do NOT output pixel coordinates. If an item is on the far right, its X is 63, not 400."*
  - Implement a safety clamp in the Swarm Planner: `goal_x = min(63, max(0, int(goal_x)))`. This ensures that even if Gemini hallucinates an out-of-bounds pixel coordinate, the heuristic distance logic stays within the 64x64 environment.

By implementing these retrospectives, the agent essentially "writes a wiki" about the game as it plays, ensuring that an early-game BFS victory isn't wasted, but instead leveraged as vital context for complex late-game levels.


```bash
source .venv312/bin/activate
python CommunitySolutions/chronos_solver/v7/play_game.py
```
