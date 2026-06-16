# Chronos Solver - v3 (Multimodal A* Swarm)

## Analysis of v2 Performance
While `v2` successfully introduced loop detection and chunking, observing the execution logs and images (`level_01_step_0014` to `0079`) revealed critical limitations:

1. **Wasting Moves on Walls:** When the Critic Agent detected a loop, it forced a random exploration move. Because it was "blind" to the environment structure, it frequently chose moves that resulted in running directly into walls, causing zero progress and wasting steps.
2. **Lack of Holistic Vision:** The agent only "saw" local integers in a 2D array. It had no spatial understanding of the green maze structure or the red objective markers, making its pathfinding purely trial-and-error.
3. **BFS Scaling Limitations:** BFS explores uniformly in all directions ($O(b^d)$). Increasing the timeout beyond 180 seconds is a bad idea because the state space grows exponentially. If a maze solution is 50 steps deep, BFS would theoretically need to explore $4^{50}$ states. It fundamentally **does not scale** for deep interactive puzzles.

## Key Improvements for v3

To solve these issues, `v3` will upgrade the swarm from an "uninformed" search to an **"informed" multimodal search**.

### 1. Multimodal LLM Vision Integration
The agent will now have direct access to the PNG images it generates! 
*   **How it works:** At the start of a level (or when stuck), the agent will pass the current visual frame directly to the Gemini Multimodal Vision API. 
*   **The Prompt:** We will ask the LLM: *"Analyze this maze. The blue dot is the player. Identify the (x, y) coordinates of the ultimate objective (e.g., the red square or the end of the green path)."*
*   **The Result:** The LLM acts as the "Global Commander," giving the swarm an exact destination coordinate to target.

### 2. A* Heuristic Pathfinding (Replacing BFS)
Since BFS does not scale, we will replace the `Swarm Planner`'s BFS queue with a **Priority Queue (A* Search)**.
*   Instead of exploring blindly, A* uses a heuristic equation: `f(n) = g(n) + h(n)`.
*   `h(n)` will be the Manhattan distance from the current state to the `(x, y)` target coordinates provided by the Vision LLM.
*   **Pros:** A* is drastically faster. It ignores dead-end branches that move away from the goal, allowing the agent to solve 100-step mazes in seconds instead of timing out at 180s.

### 3. Wall-Aware Collision Masking -- Not all games might have walls!, so be careful while implementing it.
The `Critic Agent` will be upgraded. When it forces the agent to break out of a loop, it will internally simulate the 4 directional actions. If an action results in no pixel changes (meaning it hit a wall), that action is **masked out**. The agent will only pick random exploration moves that actually result in physical movement.

## Implementation Plan

1. **Integrate Google GenAI API:** Add the API call logic to pass `matplotlib` rendered frames to Gemini to extract target coordinates.
2. **Refactor Planner to A*:** Change `deque` to `heapq` in the planner. Add the Manhattan distance heuristic logic.
3. **Add Action Masking:** Update the `CriticAgent` to pre-validate exploration moves before taking them.
4. **Testing:** Run `v3` on `ls20` Level 1 to verify that A* navigates the maze efficiently without hitting walls or timing out.

## How to Run Manually

To test the v3 implementation:

1. Ensure your terminal is at the root of the workspace.
2. Export your API key for the Vision LLM: `export GEMINI_API_KEY="your_api_key"`
3. Activate your virtual environment.
4. Execute the script:

```bash
python CommunitySolutions/chronos_solver/v3/play_game.py
```
