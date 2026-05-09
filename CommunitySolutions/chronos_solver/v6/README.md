# Chronos Solver - v6: Spatial Chain-of-Thought & Intrinsic Curiosity

Based on the performance of v5, we successfully built a continuous feedback loop and caught silent resets, but the *quality* of the cognitive loop degraded. The agent fell into repetitive thought patterns and lacked the intuitive "curiosity" required to solve complex mechanics. v6 aims to implement modern LLM spatial reasoning (Chain-of-Thought) and Intrinsic Motivation algorithms.

## Flaws Identified in v5

1. **Stale/Static LLM Output (Lack of Reasoning)**:
   - **Observation**: The Gemini Vision output became stuck in a loop, continually outputting the exact same JSON `{"x": 16, "y": 42, "lives": 3, "fuel": 17, "target_shape": "dark red pattern"}`.
   - **The Flaw**: By constraining the LLM to *only* output final state variables, we choked off its ability to "think." It had the context (the `_puzzle_monologue`), but no "scratchpad" to evaluate that context. Without generating a Chain-of-Thought (CoT), the model defaults to the most prominent visual feature rather than realizing "Wait, I've output (16, 42) three times and we haven't made progress."
   
2. **Lack of Intrinsic Curiosity**:
   - **Observation**: The agent does not intuitively explore. It ignores new items (like a `+` sign) and doesn't test hypotheses about the environment.
   - **The Flaw**: The A* Swarm planner is purely exploitative—it only heads toward the exact `(x, y)` coordinate the LLM gives it. It has no intrinsic motivation to interact with unknown pixels to learn the game's hidden rules.

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

### 3. Automated Hypothesis Validation (Memory Write-Back)
- **Solution**: When the agent's curiosity drives it to touch an unknown object, it must learn what that object does.
- **Action**: If the agent steps on a `+` sign and the `fuel` variable jumps from 17 to 42 in the next frame, the framework mathematically validates the interaction. It will then write `"green cross" = "fuel +25"` into the `v6_long_term_memory.json` cache. On the next level, the LLM won't flag it as an unknown object; it will recognize it as a resource.
