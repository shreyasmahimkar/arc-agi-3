import json
import traceback
from typing import Dict, Any, List
import sys
import os

# Add the parent directory to the path so we can import from multimodel_solver
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from multimodel_solver.multimodal_world_modeler import MultimodalWorldModeler

class REPLSynthesizer:
    def __init__(self, memory_filepath="multimodel_solver/memory_buffer/episodic_memory.json"):
        self.memory_filepath = memory_filepath
        self.modeler = MultimodalWorldModeler(memory_filepath=self.memory_filepath, output_path="LocalSimulator.py")
        
    def load_memory(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.memory_filepath):
            print(f"Error: memory file {self.memory_filepath} does not exist. Run tracer first.")
            return []
        with open(self.memory_filepath, 'r') as f:
            return json.load(f)

    def evaluate_code(self, code: str, memory: List[Dict[str, Any]]) -> str:
        """
        Executes the code and evaluates it against all transitions.
        Returns empty string if successful, else returns the error log.
        """
        # Execute in an isolated namespace
        namespace = {}
        try:
            exec(code, namespace)
        except Exception as e:
            return f"Failed to compile or execute simulator code:\n{traceback.format_exc()}"
            
        if "predict_next_state" not in namespace:
            return "Error: The code did not define a `predict_next_state` function."
            
        predict_next_state = namespace["predict_next_state"]
        
        for i, transition in enumerate(memory):
            state_raw = transition.get("state_raw", {})
            next_state_raw = transition.get("next_state_raw", {})
            action_dict = transition.get("action", {})
            
            grid_in = state_raw.get("grid")
            grid_out_expected = next_state_raw.get("grid")
            action = action_dict.get("action")
            
            if grid_in is None or grid_out_expected is None:
                continue
                
            try:
                # Some simulators might mutate the input grid, so we pass a copy if needed, 
                # but let's see how it behaves directly first.
                grid_out_actual = predict_next_state(grid_in, action)
            except Exception as e:
                return f"Exception during execution on transition {i} (Action: {action}):\n{traceback.format_exc()}"
                
            if grid_out_actual != grid_out_expected:
                error_msg = (
                    f"Validation failed on transition {i} for action '{action}'.\n"
                    f"Expected Grid: {grid_out_expected}\n"
                    f"Actual Output: {grid_out_actual}\n"
                )
                return error_msg
                
        return "" # Success

    def synthesize_with_repl(self, max_retries: int = 5):
        print(f"=== Starting System 2 REPL loop (Max retries: {max_retries}) ===")
        memory = self.load_memory()
        if not memory:
            return False
            
        print("Initial generation using multimodal visual payload...")
        code = self.modeler.generate_simulator()
        
        if not code:
            print("Failed to generate initial code.")
            return False
            
        for attempt in range(1, max_retries + 1):
            print(f"--- Attempt {attempt}/{max_retries} ---")
            error_log = self.evaluate_code(code, memory)
            
            if not error_log:
                print("✅ Simulator passed all memory transition tests!")
                return True
                
            print(f"❌ Evaluation failed:\n{error_log[:500]}...\n")
            if attempt < max_retries:
                print("Requesting correction from Gemini...")
                code = self.modeler.correct_simulator(code, error_log)
                if not code:
                    print("Failed to get corrected code.")
                    return False
            else:
                print("Reached maximum retries. Simulator synthesis failed.")
                return False

if __name__ == "__main__":
    repl = REPLSynthesizer()
    repl.synthesize_with_repl()
