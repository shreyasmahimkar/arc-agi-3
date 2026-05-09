# Chronos Solver - v6: Spatial Chain-of-Thought & Intrinsic Curiosity

Based on the performance of v5, we successfully built a continuous feedback loop and caught silent resets, but the *quality* of the cognitive loop degraded. The agent fell into repetitive thought patterns and lacked the intuitive "curiosity" required to solve complex mechanics. v6 aims to implement modern LLM spatial reasoning (Chain-of-Thought) and Intrinsic Motivation algorithms.

## Flaws Identified in v5

1. **Stale/Static LLM Output (Lack of Reasoning)**:
   - **Observation**: The Gemini Vision output became stuck in a loop, continually outputting the exact same JSON `{"x": 16, "y": 42, "lives": 3, "fuel": 17, "target_shape": "dark red pattern"}`.
   - **The Flaw**: By constraining the LLM to *only* output final state variables, we choked off its ability to "think." It had the context (the `_puzzle_monologue`), but no "scratchpad" to evaluate that context. Without generating a Chain-of-Thought (CoT), the model defaults to the most prominent visual feature rather than realizing "Wait, I've output (16, 42) three times and we haven't made progress."
   
2. **Lack of Intrinsic Curiosity**:
   - **Observation**: The agent does not intuitively explore. It ignores new items (like a `+` sign) and doesn't test hypotheses about the environment.
   - **The Flaw**: The A* Swarm planner is purely exploitative—it only heads toward the exact `(x, y)` coordinate the LLM gives it. It has no intrinsic motivation to interact with unknown pixels to learn the game's hidden rules.
   
3. **Shallow Memory Persistence**:
   - **Observation**: The `v5_long_term_memory.json` file only contains a single literal string: `{"target_shape": "dark red pattern"}`.
   - **The Flaw**: The persistence mechanism in v5 was essentially just a variable dump. It lacked the structural detail necessary to store game mechanics. Knowing the target shape is "dark red pattern" doesn't help the agent remember *how* to interact with the environment (e.g., that touching a green cross replenishes fuel).

4. **Single-Frame Blindness & Debugging Opacity**:
   - **Observation**: The agent only sends a single, un-auditable frame to Gemini during its polling loop.
   - **The Flaw**: Without sending a sequence of recent images (e.g., the last 5 frames), Gemini cannot visually perceive motion, the agent's trajectory, or immediate cause-and-effect. Additionally, because the images are generated in memory and not saved to an `images/` folder or explicitly logged by filename, it is impossible to debug what Gemini is actually "seeing."

## Implementation Plan for v6

### 1. Spatial Chain-of-Thought (CoT) Prompting
- **Solution**: We will restructure the JSON schema requirement to force the LLM to "think out loud" before committing to coordinates. Recent research shows that forcing an LLM to generate a symbolic/spatial analysis prior to generating final coordinates significantly boosts spatial intelligence.
- **Action**: Modify the prompt to demand a new key: `"spatial_analysis"`. 
- **Example Expected Output**: 
  ```json
  {
    "spatial_analysis": "Looking at the history, my previous goal of (16, 42) resulted in a lost life because of a trap. I also see a green '+' sign at (20, 15) which might replenish fuel. Let's redirect the agent to test the '+' sign.",
    "x": 20, 
    "y": 15,
    "lives": 2,
    "fuel": 17,
    "target_shape": "dark red pattern"
  }
  ```

### 2. Intrinsic Curiosity & Object Identification
- **Solution**: Transform the agent from an "objective follower" into a "hypothesis tester."
- **Action**: Add an `"unknown_objects"` array to the JSON schema. The LLM will flag any unrecognized sprites (e.g., `"green cross"`, `"yellow lock"`). 
- **The Heuristic Shift**: If the agent's primary A* path is blocked or looping, the Swarm Planner will temporarily override the ultimate goal and apply a massive heuristic bonus to path toward the closest `"unknown_object"`.

### 3. Automated Hypothesis Validation 
- **Solution**: When the agent's curiosity drives it to touch an unknown object, it must mathematically validate what that object does.
- **Action**: The framework compares the game state (lives, fuel) right before and right after touching an unknown object to calculate the exact delta effect.

### 4. Structured Rulebook Memory (Upgraded Persistence)
- **Solution**: Overhaul `v5_long_term_memory.json` from a shallow dictionary into a structured **Action-Effect Rulebook**.
- **Action**: Once a hypothesis is validated (e.g., touching a `+` sign increases fuel by 25), the agent writes a structured rule to the memory file:
  ```json
  {
    "mechanics": {
      "green '+' sign": {"effect": "fuel_replenish", "value": "+25"},
      "red skull": {"effect": "fatal_trap"}
    },
    "current_goal": "dark red pattern"
  }
  ```
  On the next level, the LLM parses this rulebook in its prompt, transforming it from an amnesiac explorer into an experienced player who already knows the game's core mechanics.

### 5. Multi-Frame Vision Context & Image Auditing
- **Solution**: Upgrade the Gemini vision payload from a single image to a multi-frame sequence, and implement explicit image logging to disk.
- **Action**: 
  - Instead of sending just the current frame, the agent will extract the most recent 5 frames from the game state.
  - It will save these frames to an `images/` directory using a strict naming convention (e.g., `level_01_step_0125_frame_1.png`).
  - The agent will log these exact filenames to `v6_run.log` and pass the entire batch of 5 images to Gemini. This grants Gemini temporal awareness (motion tracking) and gives developers a perfect audit trail to debug visual hallucinations.



```bash
source .venv312/bin/activate
python CommunitySolutions/chronos_solver/v6/play_game.py
```
