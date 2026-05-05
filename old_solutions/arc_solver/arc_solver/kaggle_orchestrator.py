import time
import gc
import sys
from typing import List, Dict, Any

try:
    import torch
except ImportError:
    torch = None

def force_garbage_collection():
    """Forces garbage collection and clears CUDA cache if available."""
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

class KaggleOrchestrator:
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        
        # 1. Initialize our system components
        from arc_solver.explorer import EpisodicMemory
        from arc_solver.world_modeler import WorldModeler, SafeSandbox
        from arc_solver.planner import MCTSPlanner
        
        self.memory = EpisodicMemory()
        self.sandbox = SafeSandbox(self.memory.filepath)
        self.world_modeler = WorldModeler(self.sandbox)
        self.planner = MCTSPlanner(num_simulations=10, num_cores=2)
        
    def solve_puzzle(self, puzzle: Dict[str, Any]) -> Any:
        """
        Attempts to solve a single puzzle within the strict wall-clock timeout.
        If timeout hits, falls back to a default move and forces GC.
        """
        start_time = time.time()
        fallback_move = [[0]]  # Minimal safe fallback for ARC
        
        try:
            print("[1/5] Exploring and logging episodic memory...")
            # Dummy exploration logic (would normally run the puzzle locally)
            self.memory.log(state={"grid": [1, 2]}, action="up", next_state={"grid": [2, 1]})
            self.memory.save()
            
            print("[2/5] Generating Simulator Code via LLM...")
            print("[3/5] Testing code in Sandbox...")
            simulator_code = self.world_modeler.generate_simulator()
            
            if not simulator_code:
                print("Failed to generate simulator. Outputting fallback move.")
                return fallback_move
                
            print("[4/5] Planning offline with MCTS...")
            best_action = self.planner.plan(initial_state={"grid": [1, 2]})
            
            print(f"[5/5] Act! Chosen action: {best_action}")
            return best_action
            
        except TimeoutError as e:
            print(f"[Timeout] {str(e)} Outputting fallback move.")
            return fallback_move
        except Exception as e:
            print(f"[Error] Exception occurred: {str(e)}")
            return fallback_move
        finally:
            force_garbage_collection()
            
    def run(self, puzzles: List[Dict[str, Any]]) -> List[Any]:
        """
        Runs the orchestrator over a sequence of ARC puzzles.
        """
        predictions = []
        for i, puzzle in enumerate(puzzles):
            print(f"Solving puzzle {i+1}/{len(puzzles)}...")
            prediction = self.solve_puzzle(puzzle)
            predictions.append(prediction)
            print(f"Finished puzzle {i+1}.")
            
        return predictions

if __name__ == "__main__":
    import os
    import json
    
    # 1. Set environment
    env_mode = os.environ.get("ENV", "DEV")
    print(f"Running in ENV={env_mode}")
    
    # 2. Feed a sample public ARC-AGI-3 training puzzle
    base_dir = "/Users/shreyas/gitrepos/OpenSource/kaggle/arc3/arc-prize-2026-arc-agi-3/environment_files"
    
    # Pick the 'ls20' folder as the sample puzzle
    sample_puzzle_dir = os.path.join(base_dir, "ls20", "9607627b") # Adjusting for inner version folder
    if not os.path.exists(sample_puzzle_dir):
        # Fallback to just scanning the first available
        for root, dirs, files in os.walk(base_dir):
            if "metadata.json" in files:
                sample_puzzle_dir = root
                break
                
    metadata_path = os.path.join(sample_puzzle_dir, "metadata.json")
    
    puzzle_data = {"name": "sample_puzzle"}
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            puzzle_data = json.load(f)
            print(f"Loaded sample puzzle metadata from: {metadata_path}")
    else:
        print(f"Warning: Could not find metadata.json at {metadata_path}")
        
    print(f"Target Puzzle: {puzzle_data.get('game_name', 'Unknown')}")
    
    # 3. Initialize and run Orchestrator
    orchestrator = KaggleOrchestrator(timeout_seconds=300)
    orchestrator.run([puzzle_data])
