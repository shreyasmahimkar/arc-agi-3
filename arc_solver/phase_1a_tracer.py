import time
import arc_agi
from arcengine import GameAction

# Import our three core modules
from arc_solver.explorer import EpisodicMemory, Explorer
from arc_solver.world_modeler import WorldModeler, SafeSandbox
from arc_solver.planner import MCTSPlanner

def map_action_to_enum(action_str: str) -> GameAction:
    """
    Maps string actions from the planner to arcengine.GameAction enums.
    Attempts to match specific names, otherwise falls back to ACTION1.
    """
    action_upper = action_str.upper()
    if hasattr(GameAction, action_upper):
        return getattr(GameAction, action_upper)
    elif hasattr(GameAction, f"MOVE_{action_upper}"):
        return getattr(GameAction, f"MOVE_{action_upper}")
    else:
        # Generic fallback for ARC games if standard directional names aren't used
        return getattr(GameAction, "ACTION1", list(GameAction)[0])

def execute_action(env, action_dict: dict):
    """Executes a dictionary action against the arcengine environment."""
    action_enum = map_action_to_enum(action_dict["action"])
    try:
        if "x" in action_dict and "y" in action_dict:
            return env.step(action_enum, x=action_dict["x"], y=action_dict["y"])
        else:
            return env.step(action_enum)
    except TypeError:
        # Fallback if step doesn't accept kwargs properly
        return env.step(action_enum)

def main():
    print("=== Phase 1a: End-to-End Tracer Bullet ===")
    
    # Initialize core modules
    memory = EpisodicMemory()
    memory.clear() # Ensure we start fresh
    explorer = Explorer(memory)
    sandbox = SafeSandbox(memory.filepath)
    world_modeler = WorldModeler(sandbox)
    planner = MCTSPlanner(num_simulations=20, num_cores=4)
    
    # 1. Instantiate the official arc_agi API
    print("[1/5] Instantiating arc_agi environment (human render mode)...")
    arc = arc_agi.Arcade()
    env = arc.make('ls20', render_mode='human')
    
    # We assume env has a reset method to initialize state
    try:
        initial_state = env.reset()
    except Exception:
        initial_state = {"grid": [], "available_actions": ["ACTION1"]}
    
    # 2. Gather initial episodic memory (max 5 actions)
    print("[2/5] Gathering episodic memory via strict Explorer...")
    for i in range(5):
        # We perform strictly constrained actions based on available_actions
        action_dict = explorer.sample_action(initial_state)
        
        try:
            step_output = execute_action(env, action_dict)
            # Generally returns (observation, reward, done, truncated, info) or similar
            next_state = step_output[0] if isinstance(step_output, tuple) else step_output
            if next_state is None:
                next_state = {"grid": []}
        except Exception as e:
            print(f"Error stepping environment: {e}")
            break
            
        # Log transition to memory
        memory.log(state=initial_state, action=action_dict, next_state=next_state)
        initial_state = next_state
        time.sleep(0.1)
        
    memory.save()
    print(f"-> Saved {len(memory.memory)} transitions to {memory.filepath}")
    
    # 3. Synthesize LocalSimulator.py using Gemini
    print("[3/5] World Modeler: Synthesizing LocalSimulator.py via LLM...")
    simulator_code = world_modeler.generate_simulator()
    if not simulator_code:
        print("Failed to generate simulator. Exiting tracer.")
        return
        
    # 4. Offline Planner generates array of actions
    print("[4/5] Planner: Running MCTS to generate an array of actions...")
    planned_actions = []
    
    # We'll generate 4 actions to execute in sequence
    for _ in range(4):
        best_action_array = planner.plan(initial_state=initial_state)
        planned_actions.extend(best_action_array)
        
    print(f"-> Planned Actions: {planned_actions}")
    
    # 5. Execute sequence in env and get scorecard
    print("[5/5] Executing planned sequence...")
    for action_dict in planned_actions:
        action_str = action_dict["action"]
        action_enum = map_action_to_enum(action_str)
        print(f"Applying action: {action_dict} -> {action_enum}")
        
        try:
            execute_action(env, action_dict)
        except Exception as e:
            print(f"Failed to execute {action_enum}: {e}")
            
        # Add 0.5s sleep to visually watch the moves render
        time.sleep(0.5)
        
    print("\n=== Execution Complete ===")
    print("Scorecard Efficiency:")
    print(arc.get_scorecard())

if __name__ == "__main__":
    main()
