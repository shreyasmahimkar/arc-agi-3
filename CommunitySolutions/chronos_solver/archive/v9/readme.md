# V9 Architecture Upgrade: Autonomous Semantic Discovery & Death Autopsies

This implementation plan outlines the upgrades required for the V9 Chronos Solver to dynamically deduce game mechanics (like players, fuel, and deaths) without overfitting to specific games like `ls20`.

## Phase 1: Autonomous Semantic Discovery (The "What Am I?" Initialization)
Instead of assuming a game has a "player" or "lives", the agent will establish a **Semantic Dictionary** at the start of a new game.

*   **Prompt Modification (`_get_multimodal_goal`)**: 
    Before outputting a path, Gemini will be tasked with identifying the abstract roles of visual elements via contrastive frame analysis.
    *   *Prompt Injection*: "Analyze the deltas across these frames. Identify the 'active agent' (the entity changing position/being manipulated). Identify 'gauges/resources' (persistent UI elements that deplete). Identify 'hazards' (what destroys the agent). Note: Some games do not have avatars; identify the core manipulation metric instead."
*   **JSON Schema Update**: 
    Add a `"semantic_definitions"` key to the expected JSON output. Gemini must explicitly define what the player, fuel, and level reset look like visually (e.g., `"semantic_definitions": "Player = dark red block. Fuel = green UI bar. Reset = teleportation to top left."`)
*   **Memory Persistence**: Save this dictionary to `v9_long_term_memory.json` so the agent understands the visual language of the game for all future levels.

## Phase 2: The Death Autopsy
Currently, the `Silent Reset Detector` mathematically flags a death (using `_fatal_hashes`) and silently prunes the path. We will give this system a voice.

*   **Autopsy API Trigger**: When the mathematical hash detects a sudden teleportation/reset, execution pauses and triggers a dedicated `_perform_death_autopsy()` Gemini call.
*   **Frame Injection**: Feed Gemini the screenshot exactly *1 action before death* alongside the *reset frame*.
*   **Autopsy Prompt**: "A fatal event or level reset just occurred. Compare these two frames. What sub-optimal move or hazard triggered this? Did a resource gauge deplete? Define the exact visual cause of failure."
*   **Log Injection**: Write this highly detailed autopsy directly into the active `v9_gamename_level_X_scratchpad_iteration_Y.json` under a loud `"DEATH_ANALYSIS"` tag.

## Phase 3: Iteration Tracking & Sub-optimal Route Correction
The agent must learn from the autopsy in its subsequent attempts.

*   **Iteration Increment**: Upon a detected death, the agent closes the current scratchpad (`iteration_0`) and initializes `iteration_1`. 
*   **Context Passing**: The `DEATH_ANALYSIS` from `iteration_0` is injected into the very first prompt of `iteration_1`.
*   **Feedback Loop**: Gemini is instructed: *"In the previous iteration, your plan was sub-optimal and resulted in death via [Autopsy Reason]. Adjust your `gameplan` to explicitly route around this hazard or manage the depleted resource better."*

## Execution Checklist
1. [ ] Update `v9_long_term_memory.json` schema to accept `"semantic_definitions"`.
2. [ ] Modify the main Gemini prompt in `_get_multimodal_goal` to require the explicit identification of abstract components (player, fuel, UI deltas).
3. [ ] Build the `_perform_death_autopsy()` function to trigger upon `Silent Reset` detection.
4. [ ] Update the `_update_scratchpad()` utility to finalize the current iteration file and spawn the `iteration_X+1` file upon death.
5. [ ] Pass the previous iteration's autopsy results into the active prompt's context window.


```bash
source .venv312/bin/activate
python CommunitySolutions/chronos_solver/v9/play_game.py --game ls20
python CommunitySolutions/chronos_solver/v9/play_game.py --game ar25
python CommunitySolutions/chronos_solver/v9/play_game.py --game bp35
```


