import time
import arc_agi
from typing import Any
from arcengine import GameAction

# Import our three core modules
from arc_solver.explorer import EpisodicMemory, Explorer
from arc_solver.world_modeler import WorldModeler, SafeSandbox
from arc_solver.planner import MCTSPlanner

def map_action_to_enum(action_val: Any) -> GameAction:
    """
    Maps string actions or integer IDs from the planner to arcengine.GameAction enums.
    """
    if isinstance(action_val, int):
        try:
            return GameAction(action_val)
        except ValueError:
            return list(GameAction)[0]
            
    action_upper = str(action_val).upper()
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
        
    # 4. Offline Planner generates an array of actions
    print("[4/5] Planner: Running MCTS to generate a sequence of actions...")
    
    # Fix Multiprocessing Crash: Sanitize FrameDataRaw to a pure dictionary
    # so the 4 CPU pool workers don't crash trying to unpickle arcengine Enums
    import json
    from arc_solver.explorer import CustomJSONEncoder
    clean_state = json.loads(json.dumps(initial_state, cls=CustomJSONEncoder))
    
    # Run MCTS exactly ONCE to extract the Principal Variation (best sequence of actions)
    best_action_array = planner.plan(initial_state=clean_state)
    planned_actions = best_action_array[:4]  # Take the first 4 actions of the best sequence
        
    print(f"-> Planned Actions: {planned_actions}")
    
    # ---------------------------------------------------------
    # NEW: ADVANCED ARCHITECTURE DEMONSTRATIONS (TRM, REPL, AI)
    # ---------------------------------------------------------
    print("\n[4.1/5] Tiny Recursive Model: Performing recursive forward pass...")
    try:
        import torch
        from arc_solver.trm_reasoner import TinyRecursiveModel
        # Input dim is 64x64 for ARC grid, output dim is action space size
        trm = TinyRecursiveModel(input_dim=64*64, output_dim=8)
        dummy_input = torch.randn(1, 64*64)
        pred, steps = trm(dummy_input)
        print(f"  -> TRM predicted tensor of shape {pred.shape} after {steps} dynamic recursion steps.")
    except Exception as e:
        print(f"  -> TRM Demo Failed: {e}")

    print("\n[4.2/5] Evolutionary Synthesizer: Running System 2 REPL loop...")
    try:
        from arc_solver.evolutionary_synthesis import EvolutionarySynthesizer
        synthesizer = EvolutionarySynthesizer()
        # It attempts to write a programmatic solve() function based on clean_state
        synthetic_program = synthesizer.synthesize_program(task_id="tracer_demo", state=clean_state, max_generations=1)
        print(f"  -> Synthesizer generated program length: {len(synthetic_program)} chars.")
    except Exception as e:
        print(f"  -> Synthesizer Demo Failed: {e}")

    print("\n[4.3/5] Active Inference: Updating belief state and evaluating actions...")
    try:
        from arc_solver.active_inference import ARCGymWrapper, ActiveInferenceAgent
        ai_env = ARCGymWrapper(env)
        ai_agent = ActiveInferenceAgent(state_dim=64*64, action_dim=1)
        
        # Simulate an observation step
        available_actions = clean_state.get("available_actions", ["ACTION1", "ACTION2"])
        best_ai_action = ai_agent.select_action(clean_state, available_actions=available_actions)
        print(f"  -> Active Inference HMC evaluated Expected Free Energy and prefers: {best_ai_action}")
    except Exception as e:
        print(f"  -> Active Inference Demo Failed: {e}")
    # ---------------------------------------------------------
    # EXPLICIT ENUM MAPPER (Bypasses Integer Collisions)
    # ---------------------------------------------------------
    enum_map = {
        "ACTION1": GameAction.ACTION1,
        "ACTION2": GameAction.ACTION2,
        "ACTION3": GameAction.ACTION3,
        "ACTION4": GameAction.ACTION4,
        "ACTION5": GameAction.ACTION5,
        "ACTION6": GameAction.ACTION6,
        "ACTION7": GameAction.ACTION7,
        "RESET": GameAction.RESET
    }

    print(f"\nPlan Formulated! Executing {len(planned_actions)} actions...")

    # 5. EXECUTION & REALITY CHECK PHASE
    for act_dict in planned_actions:
        time.sleep(0.5) 
        
        raw_act = act_dict.get("action")
        
        # DEFENSIVE PARSING: If the LLM still hallucinates an integer, auto-correct it
        if isinstance(raw_act, int):
            act_name = f"ACTION{raw_act}"
            print(f"Warning: Auto-corrected integer {raw_act} to string '{act_name}'")
        else:
            act_name = str(raw_act).upper()
            
        real_action = enum_map.get(act_name)
        
        if not real_action:
            print(f"CRITICAL Error: Could not map '{raw_act}' to a valid GameAction. Aborting.")
            break

        # Hard-ban the planner from trying to execute RESETs to cheat the game
        if real_action == GameAction.RESET:
            print("ALERT: Planner attempted to execute RESET. Halting to prevent loop.")
            break
            
        try:
            # Handle Coordinate-based clicking for ACTION6
            if act_name == "ACTION6":
                x = int(act_dict.get("x", 0))
                y = int(act_dict.get("y", 0))
                print(f"Applying: {act_name} at X:{x}, Y:{y} -> {real_action}")
                step_result = env.step(real_action, x=x, y=y)
            else:
                # Standard simple actions (Move, Rotate, etc.)
                print(f"Applying: {act_name} -> {real_action}")
                step_result = env.step(real_action)
                
            # Bulletproof Unpacking (Handles 4-tuple, 5-tuple, and single-object returns)
            if isinstance(step_result, tuple):
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                elif len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    print(f"Unexpected return signature length: {len(step_result)}")
                    done = False
            else:
                # API returns a single state object (like FrameDataRaw) instead of a Gym tuple
                obs = step_result
                done = False
                
                # Dynamically check for a completion signal inside the object
                if isinstance(obs, dict) and obs.get("state") == "FINISHED":
                    done = True
                elif hasattr(obs, "state") and getattr(obs, "state") == "FINISHED":
                    done = True
                elif hasattr(obs, "done") and getattr(obs, "done"):
                    done = True
                
            if done:
                print("\n🎉 LEVEL COMPLETE! Architecture successfully adapted.")
                break
                
        except Exception as e:
            # Catch true API Errors
            print(f"API Error during step execution: {type(e).__name__} - {e}")
            env.step(GameAction.RESET)
            break
        
    print("\n=== Execution Complete ===")
    print("Scorecard Efficiency:")
    print(arc.get_scorecard())

if __name__ == "__main__":
    main()
