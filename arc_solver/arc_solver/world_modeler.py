import subprocess
import tempfile
import os
import re
import sys
from typing import Optional, Tuple
from arc_solver.llm_client import get_llm_client

class SafeSandbox:
    def __init__(self, memory_file: str = "episodic_memory.json"):
        self.memory_file = memory_file
        
    def execute(self, code: str) -> Tuple[bool, str]:
        """
        Executes the provided python code in a temporary file.
        Returns a tuple of (success_boolean, output_or_error_string).
        """
        fd, temp_path = tempfile.mkstemp(suffix=".py", text=True)
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(code)
            
            result = subprocess.run(
                [sys.executable, temp_path, self.memory_file],
                capture_output=True,
                text=True,
                timeout=3.0
            )
            
            if result.returncode == 0:
                return True, result.stdout
            else:
                return False, result.stderr
        except subprocess.TimeoutExpired as e:
            return False, f"TimeoutExpired: Execution exceeded 3.0 seconds.\n{e.stderr or ''}"
        except Exception as e:
            return False, f"Execution failed: {str(e)}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

class WorldModeler:
    def __init__(self, sandbox: Optional[SafeSandbox] = None):
        self.llm = get_llm_client()
        self.sandbox = sandbox or SafeSandbox()
        self.max_retries = 5

    def extract_code(self, response: str) -> str:
        """Extracts python code from markdown block."""
        match = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return response.strip()

    def generate_simulator(self) -> Optional[str]:
        """
        Prompts LLM to generate a Numba-optimized LocalSimulator.py class.
        Runs it via SafeSandbox up to 5 times if there are errors.
        """
        prompt = (
            "Write a complete Python script containing a class `LocalSimulator`. The core simulation "
            "logic inside or used by this class MUST be optimized using the Numba `@njit` decorator. "
            "The script should accept a JSON file path as a command-line argument (`sys.argv[1]`), "
            "load the episodic memory from that JSON file, and use the memory deltas to simulate "
            "physical grid changes.\n\n"
            "CRITICAL SCHEMA REQUIREMENTS:\n"
            "1. You MUST use and enforce the following strict TypedDict action schema in your code:\n"
            "   ```python\n"
            "   from typing import TypedDict, Optional\n"
            "   class ActionDict(TypedDict, total=False):\n"
            "       action: str\n"
            "       x: Optional[int]\n"
            "       y: Optional[int]\n"
            "   ```\n"
            "2. The `LocalSimulator` must define a `step(self, state, action_dict: ActionDict)` function that strictly "
            "accepts dictionary payloads with explicit string names. Raw integers (like 1 or 2) for the action key are "
            "STRICTLY FORBIDDEN to prevent Enum collisions.\n"
            "3. You must exactly support this action mapping:\n"
            "   - ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right\n"
            "   - ACTION5=Interact/Rotate\n"
            "   - ACTION6=Click (requires x, y coordinates 0-63)\n"
            "   - ACTION7=Undo, RESET=Restart\n"
            "4. Goal Hypothesis & Heuristic Reward: Visually analyze the initial state from the episodic memory "
            "and hypothesize an Intrinsic Goal (e.g., 'Make the shape match the outline').\n"
            "   You MUST write an `evaluate_state(self, state)` function inside `LocalSimulator` that returns a gradient "
            "heuristic score (+10 for getting closer to the goal, +100 for winning, -1 for wasted moves).\n"
            "   The `step` method MUST use this heuristic and return a tuple `(next_state, float_reward, done_bool)`. "
            "The reward MUST be this flat numerical float so the MCTS planner can climb the gradient.\n\n"
            "Run a dummy validation loop over the memory transitions to prove it works, and print 'Success'. "
            "Wrap your code in a ```python ... ``` block."
        )
        
        for attempt in range(self.max_retries):
            response = self.llm.generate(prompt)
            code = self.extract_code(response)
            
            if not code:
                prompt = "You did not return any code. Please provide the complete Python script in a ```python ... ``` block."
                continue
                
            success, output = self.sandbox.execute(code)
            
            if success:
                print(f"Simulator generated successfully on attempt {attempt + 1}")
                with open("LocalSimulator.py", "w") as f:
                    f.write(code)
                return code
            else:
                print(f"Attempt {attempt + 1} failed. Error:\n{output}")
                prompt = (
                    f"The previous code failed with the following traceback/error:\n```\n{output}\n```\n"
                    "Please fix the errors and provide the updated complete Python script "
                    "containing the `LocalSimulator` optimized with Numba `@njit`. "
                    "Remember to wrap it in a ```python ... ``` block."
                )
                
        print("Failed to generate a working simulator after 5 attempts.")
        return None
