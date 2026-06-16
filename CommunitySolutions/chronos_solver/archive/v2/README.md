# Chronos Solver - v2 (Agentic Swarm Implementation)

## Overview
This directory contains the `v2` iteration of the Chronos solver. 
Based on the failure analysis of `v1` (where the BFS solver timed out on Level 1 and the CNN fallback resulted in an infinite oscillation loop of UP/DOWN/LEFT/RIGHT), `v2` introduces an **Agentic Swarm Architecture** to overcome these limitations.

## Key Improvements Over v1

1. **Agentic Swarm Routing & Delegation**
   Instead of a monolithic BFS and a fragile CNN, `v2` splits the reasoning into a committee of specialized agents:
   - **`VisionScout Agent`**: Analyzes the raw grid to identify the player, walls, and goal coordinates.
   - **`Planner Agent` (Sub-Goal Chunking)**: Breaks the deep search problem into smaller waypoints. Instead of a 180s BFS timeout trying to find the end goal, it delegates short-range BFS tasks to reach intermediate waypoints (e.g., the next corner).
   - **`Critic Agent` (Anti-Oscillation)**: Actively monitors the state hash history. If it detects the agent is stuck in a loop (the "round and round" issue from v1), it forcefully vetos the cyclic actions and enforces exploration.

2. **Parallelized Search (Swarm Pathfinding)**
   When the heuristic fails, multiple lightweight pathfinding agents (swarms) are deployed in parallel to explore different branches of the search tree simultaneously, reporting back the most promising routes to the `Critic Agent`.

3. **Multimodal LLM Fallback Mechanism**
   If the algorithmic swarm fails to find a path, the agent will format the current state grid and explicitly query the Gemini Multimodal LLM to provide a high-level strategic hint to unblock the swarm.

## Implementation Plan
1. **Refactor `my_agent.py`**: Implement the `SwarmCoordinator` class that orchestrates the sub-agents.
2. **Implement Loop Detection**: Add a strict state-hash queue that drastically penalizes previously visited states within a rolling window.
3. **Integrate Waypoint Logic**: Extract the green boundaries and calculate the skeleton of the path to act as sub-goals for the BFS engine.
4. **Iterative Testing**: Run `v2` against `ls20` Level 1 to ensure the swarm successfully bypasses the step 14-79 oscillation and reaches the goal.

## Swarm Collaboration Deep Dive

In `v2`, the "all-or-nothing" monolithic solver and the blind CNN are replaced with a localized swarm architecture that interacts continuously:

### 1. The Critic Agent (The Watchdog)
Before any planning happens, the **Critic Agent** runs a health check:
*   **Memory Integration:** It maintains a rolling queue of the last 15 visual frame hashes.
*   **Oscillation Detection:** It analyzes that memory. If it notices that out of the last 15 steps, there are **4 or fewer unique states**, it realizes the agent is trapped in a loop (exactly what happened from step 14 to 79 in v1).
*   **Veto Power:** When a loop is detected, the Critic forcefully intervenes. It wipes the current plan and injects a high-entropy random action to force the agent to explore a new path, breaking the cycle.

### 2. The Swarm Planner (Rolling Horizon Chunking)
If the Critic approves the current state, it defers to the **Swarm Planner**. Instead of trying to solve the entire maze at once, it breaks the journey into chunks:
*   **Parallel Exploration:** It spawns hundreds of lightweight BFS "pathfinders" from the current position.
*   **Strict Budgeting:** It limits these pathfinders to a maximum depth of 12 steps and a hard timeout of 15 seconds. 
*   **Heuristic Scoring:** When the 15 seconds are up, the pathfinders report back. The Planner scores each branch using `score = np.sum(f0 != f) + depth * 0.5`. This identifies which short path changed the grid the most and moved the furthest.
*   **Waypoint Commitment:** The Planner takes the highest-scoring path. To avoid over-committing to a dead end, it only returns the **first 3 actions** of that path as the immediate waypoint. 

### The Feedback Loop
Together, they form a robust **Model Predictive Control (MPC)** loop:
1. The **Planner** looks 12 steps ahead, finds the most promising direction, and feeds the agent 3 steps to execute.
2. The agent takes those 3 steps.
3. The **Critic** watches the results. As long as the agent isn't looping, the **Planner** is called again to spawn a new 15-second swarm from the new location to find the *next* 3 steps.

## How to Run Manually

To run the v2 Swarm Solver and generate the logs/images:

1. Ensure your terminal is at the root of the workspace.
2. Activate your virtual environment (if you are using one, e.g., `source .venv312/bin/activate`).
3. Execute the `play_game.py` script specifically from the `v2` folder:

```bash
python CommunitySolutions/chronos_solver/v2/play_game.py
```

This will automatically create the `v2_run.log` and populate the `CommunitySolutions/chronos_solver/v2/images/` directory with the step-by-step frame captures.
