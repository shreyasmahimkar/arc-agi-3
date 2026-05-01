import time
import sys
import os
sys.path.insert(0, os.path.abspath('arc_solver'))

import arc_agi
from typing import Any
from arcengine import GameAction

# Import our core modules
from multimodel_solver.explorer import EpisodicMemory, Explorer
from arc_solver.world_modeler import SafeSandbox
from multimodel_solver.multimodal_world_modeler import MultimodalWorldModeler
from arc_solver.planner import MCTSPlanner

def map_action_to_enum(action_val: Any) -> GameAction:
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
        return getattr(GameAction, "ACTION1", list(GameAction)[0])

def execute_action(env, action_dict: dict):
    action_enum = map_action_to_enum(action_dict["action"])
    try:
        if "x" in action_dict and "y" in action_dict:
            return env.step(action_enum, x=action_dict["x"], y=action_dict["y"])
        else:
            return env.step(action_enum)
    except TypeError:
        return env.step(action_enum)

def main():
    print("=== Phase 1a: End-to-End Tracer Bullet (Multimodal Pivot) ===")
    
    # Initialize core modules
    memory = EpisodicMemory()
    memory.clear() # Ensure we start fresh for this session
    explorer = Explorer(memory)
    planner = MCTSPlanner(num_simulations=20, num_cores=4)
    
    # 1. Instantiate the official arc_agi API
    print("[1/5] Instantiating arc_agi environment (human render mode)...")
    arc = arc_agi.Arcade()
    env = arc.make('ls20', render_mode='human')
    
    try:
        reset_out = env.reset()
        initial_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        info = reset_out[1] if isinstance(reset_out, tuple) and len(reset_out) > 1 else {"level": 0, "lives": 3}
    except Exception:
        initial_state = {"grid": [], "available_actions": ["ACTION1"]}
        info = {"level": 0, "lives": 3}

    current_lives = info.get("lives", 3)
    current_level = info.get("level", 0)
    
    # Outer Loop: Continuously explore, synthesize, plan, and execute
    step_count = 0
    max_steps = 100
    
    while step_count < max_steps:
        step_count += 1
        print(f"\n==============================================")
        print(f"=== OUTER LOOP ITERATION {step_count}/{max_steps} ===")
        print(f"==============================================")
        
        # 2. Gather episodic memory via Exploration
        print("\n[2/5] Gathering episodic memory via strict Explorer...")
        # Take a few exploration steps to gather new data
        for i in range(3):
            action_dict = explorer.sample_action(initial_state)
            
            try:
                step_output = execute_action(env, action_dict)
                if isinstance(step_output, tuple):
                    if len(step_output) == 5:
                        next_state, reward, terminated, truncated, step_info = step_output
                    elif len(step_output) == 4:
                        next_state, reward, terminated, step_info = step_output
                        truncated = False
                    else:
                        next_state = step_output[0]
                        step_info = {"level": current_level, "lives": current_lives}
                        terminated = False
                else:
                    next_state = step_output
                    step_info = {"level": current_level, "lives": current_lives}
                    terminated = False
                    
                if next_state is None:
                    next_state = {"grid": []}
            except Exception as e:
                print(f"Error stepping environment: {e}")
                break
                
            # Log transition to memory
            memory.log(state=initial_state, action=action_dict, next_state=next_state)
            initial_state = next_state
            
            # Check info updates (Lives / Level)
            new_level = step_info.get("level", current_level)
            if new_level > current_level:
                print(f"🌟 EXPLORATION LEVEL UP! Advanced to Level {new_level}")
                current_level = new_level
                
            new_lives = step_info.get("lives", current_lives)
            if new_lives < current_lives:
                current_lives = new_lives
                if current_lives > 0:
                    print(f"⚠️  EXPLORATION FUEL DEPLETED! Lives remaining: {current_lives}/3")
                else:
                    print(f"💀  EXPLORATION GAME OVER! All fuel/lives exhausted.")
                    
            if terminated or current_lives <= 0:
                print("Environment reset during exploration.")
                reset_out = env.reset()
                initial_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                info_out = reset_out[1] if isinstance(reset_out, tuple) and len(reset_out) > 1 else {"level": 0, "lives": 3}
                current_lives = info_out.get("lives", 3)
                current_level = info_out.get("level", 0)
                break
            time.sleep(0.1)
            
        memory.save()
        print(f"-> Accumulated {len(memory.memory)} transitions in {memory.filepath}")
        
        # 3. Synthesize LocalSimulator.py using Gemini REPL Loop
        print("\n[3/5] World Modeler: Synthesizing LocalSimulator.py via LLM REPL Loop...")
        from multimodel_solver.repl_synthesizer import REPLSynthesizer
        repl = REPLSynthesizer(memory_filepath=memory.filepath)
        success = repl.synthesize_with_repl(max_retries=5)
        if not success:
            print("Failed to generate a validated simulator. Will gather more memory next iteration...")
            continue
            
        # 4. Offline Planner generates an array of actions
        print("\n[4/5] Planner: Running MCTS to generate a sequence of actions...")
        import json
        from multimodel_solver.explorer import CustomJSONEncoder
        clean_state = json.loads(json.dumps(initial_state, cls=CustomJSONEncoder))
        
        best_action_array = planner.plan(initial_state=clean_state)
        planned_actions = best_action_array[:4]
            
        print(f"-> Planned Actions: {planned_actions}")
        
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

        print(f"\n[5/5] Plan Formulated! Executing {len(planned_actions)} actions...")

        # 5. EXECUTION & REALITY CHECK PHASE
        for act_dict in planned_actions:
            time.sleep(0.5) 
            
            raw_act = act_dict.get("action")
            if isinstance(raw_act, int):
                act_name = f"ACTION{raw_act}"
                print(f"Warning: Auto-corrected integer {raw_act} to string '{act_name}'")
            else:
                act_name = str(raw_act).upper()
                
            real_action = enum_map.get(act_name)
            
            if not real_action:
                print(f"CRITICAL Error: Could not map '{raw_act}' to a valid GameAction. Aborting.")
                break

            if real_action == GameAction.RESET:
                print("ALERT: Planner attempted to execute RESET. Halting to prevent loop.")
                break
                
            try:
                prev_state = initial_state
                
                # Execute in environment
                if act_name == "ACTION6":
                    x = int(act_dict.get("x", 0))
                    y = int(act_dict.get("y", 0))
                    print(f"Applying: {act_name} at X:{x}, Y:{y} -> {real_action}")
                    step_result = env.step(real_action, x=x, y=y)
                else:
                    print(f"Applying: {act_name} -> {real_action}")
                    step_result = env.step(real_action)
                    
                # Unpack step result
                if isinstance(step_result, tuple):
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, step_info = step_result
                        done = terminated or truncated
                    elif len(step_result) == 4:
                        obs, reward, done, step_info = step_result
                    else:
                        obs = step_result[0]
                        done = False
                        step_info = {"level": current_level, "lives": current_lives}
                else:
                    obs = step_result
                    done = False
                    step_info = {"level": current_level, "lives": current_lives}
                    if isinstance(obs, dict) and obs.get("state") == "FINISHED":
                        done = True
                    elif hasattr(obs, "state") and getattr(obs, "state") == "FINISHED":
                        done = True
                    elif hasattr(obs, "done") and getattr(obs, "done"):
                        done = True
                        
                initial_state = obs
                
                # Check for Level Up or Life Lost in Execution Phase
                new_level = step_info.get("level", current_level)
                if new_level > current_level:
                    print(f"🌟 LEVEL UP! Advanced to Level {new_level}")
                    current_level = new_level
                    
                new_lives = step_info.get("lives", current_lives)
                if new_lives < current_lives:
                    current_lives = new_lives
                    if current_lives > 0:
                        print(f"⚠️  FUEL DEPLETED! Lives remaining: {current_lives}/3")
                    else:
                        print(f"💀  GAME OVER! All fuel/lives exhausted.")
                        done = True
                
                # Save the real execution back into episodic memory!
                # This ensures the REPL can learn from execution mistakes on the next iteration
                memory.log(state=prev_state, action=act_dict, next_state=obs)
                memory.save()
                
                if done:
                    print("\n🎉 LEVEL COMPLETE OR GAME OVER! Resetting for next attempt.")
                    reset_out = env.reset()
                    initial_state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
                    info_out = reset_out[1] if isinstance(reset_out, tuple) and len(reset_out) > 1 else {"level": 0, "lives": 3}
                    current_lives = info_out.get("lives", 3)
                    current_level = info_out.get("level", 0)
                    break
                    
            except Exception as e:
                print(f"API Error during step execution: {type(e).__name__} - {e}")
                env.step(GameAction.RESET)
                break
        
    print("\n=== Execution Complete (Reached Max Iterations) ===")
    print("Scorecard Efficiency:")
    print(arc.get_scorecard())

if __name__ == "__main__":
    main()
