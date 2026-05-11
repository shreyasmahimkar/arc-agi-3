# Chronos Solver - v8: Pre-Game Planning & Dynamic Sub-Goal Traversal

While v7 successfully improved precision tracking and hazard avoidance, it exposed a deeper cognitive limitation in the agent's planning capabilities. v8 focuses on shifting the agent from a reactive A* pathfinder into a proactive, hypothesis-driven planner.

## Flaws Identified in v7 (The "Tunnel Vision" Problem)

1. **Failure to Recognize Mandatory Exploration (Sub-Goals)**:
   - **The Flaw**: The solver treats the maze simply as a spatial routing problem to a single `(x, y)` coordinate. It fails to comprehend puzzle mechanics where interacting with "unknown" blocks (like switches, keys, or levers) is a strict prerequisite to unlocking the final goal. It either avoids them (as hazards) or visits them purely out of "curiosity", rather than recognizing them as mandatory sequence steps.

2. **Lack of Pre-Game "Thinking Loop" and Session Memory**:
   - **The Flaw**: The agent immediately starts moving and only reflects on its strategy *after* winning (saving to long-term memory). It does not pause to formulate a plan *before* the maze begins. Furthermore, if the player dies, the agent resets without a persistent "Session Memory" to remember what it learned during that specific death. It repeats the same conceptual mistakes because it lacks a working memory for the active level attempt.

3. **Static Targeting vs. Dynamic State Modification**:
   - **The Flaw**: The agent does not traverse between multiple shapes or actively monitor how its actions modify the end state. It lacks the ability to say: "I stepped on X, the map changed, now my goal is Y." Goal modifications and state understandings are not documented or tracked chronologically during the level solving process.

---

## Implementation Plan for v8

To resolve these issues, v8 will introduce the following architectural upgrades:

### 1. The Pre-Game "Thinking Loop" (Powered by Deep Think)
Before executing any actions on a new level, the agent will pause and invoke Gemini to formulate a comprehensive execution plan. 
- To achieve maximum reasoning depth, the agent will force the Gemini model into a high-level thinking mode using the `ThinkingConfig` (`include_thoughts=True, thinking_level="HIGH"`).
- It will analyze the initial frame and break the puzzle down into sequential steps.
- The raw internal thoughts extracted from the `part.thought` attribute of the response will be explicitly captured and written to the `v8_level_solving_log.json` so we can audit the agent's exact logical deductions.
- **Example Thought Process**: *"The final goal is the red box at (50, 50). However, the path is blocked by a yellow wall. There is a blue switch at (10, 10). I must first navigate to the blue switch, observe the state change, and then proceed to the red box."*

### 2. Dynamic Sub-Goal Traversal (Chained Graph)
Instead of feeding A* a single coordinate, Gemini will generate a queue of `sub_goals`.
- The A* planner will target `sub_goals[0]`. 
- Once `sub_goals[0]` is reached (or if the environment state drastically changes), the agent will pause, capture a new frame, and ask Gemini to verify if the state has changed and if the next sub-goal should be engaged.

### 3. Persistent Session Memory (`v8_session_memory.json`)
We will introduce a distinct layer of memory: **Session Memory**.
- Unlike the *Long-Term Memory* (which only persists successful generalized rules *after* a win), the *Session Memory* persists *throughout* the level's execution and survives player deaths.
- It will log the agent's active hypotheses and failures. 
- **Example Log**: `"Attempt 1: Tried going straight to goal, died to trap at (20, 20). Attempt 2: Updating plan to route around (20, 20)."`

### 4. Level Solving JSON Logging
Every dynamic goal shift and shape traversal will be continuously written to a `v8_level_solving_log.json`. This provides a transparent, chronological ledger of the agent's evolving understanding of the maze logic in real-time. 
- **Structured by Game and Level**: To ensure data is distinguishable across full evaluation runs, the JSON schema will be strictly hierarchical: `{"[game_name]": {"[level_index]": [ { "timestamp": "...", "event": "Goal Update", "details": "..." } ] } }`. This guarantees there is no log collision or amnesia when the solver switches between entirely different games and levels.
- **Thought Logging**: The `event` details will explicitly capture the raw internal thoughts (`part.thought`) from Gemini's Deep Think, providing a clear window into the exact reasoning process that preceded any sub-goal selection or action.


```bash
source .venv312/bin/activate
python CommunitySolutions/chronos_solver/v8/play_game.py
```
