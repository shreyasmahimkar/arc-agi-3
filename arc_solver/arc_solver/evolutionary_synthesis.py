import sys
import tempfile
import subprocess
import os
import json
import traceback
from typing import Dict, Any, Tuple, List
from arc_solver.llm_client import get_llm_client

class REPLSandbox:
    def __init__(self):
        self.globals_dict = {}
        
    def execute(self, code: str, state: Any) -> Tuple[bool, Any, str]:
        """
        Executes generated python code in an isolated environment.
        The code is expected to define a function `solve(state)` which we call.
        """
        try:
            # We create a local namespace
            local_namespace = {}
            exec(code, self.globals_dict, local_namespace)
            
            if 'solve' not in local_namespace:
                return False, None, "Error: Code must define a function named 'solve(state)'."
            
            solve_func = local_namespace['solve']
            result = solve_func(state)
            return True, result, "Success"
        except Exception as e:
            error_trace = traceback.format_exc()
            return False, None, f"Execution Failed:\n{error_trace}"

class EvolutionarySynthesizer:
    def __init__(self, cache_file: str = "core_knowledge_library.json"):
        self.llm = get_llm_client()
        self.repl = REPLSandbox()
        self.cache_file = cache_file
        self.core_knowledge: Dict[str, str] = {}
        self._load_knowledge()
        
    def _load_knowledge(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r') as f:
                    self.core_knowledge = json.load(f)
            except json.JSONDecodeError:
                self.core_knowledge = {}
                
    def _save_knowledge(self, task_id: str, code: str):
        self.core_knowledge[task_id] = code
        with open(self.cache_file, 'w') as f:
            json.dump(self.core_knowledge, f, indent=4)
            
    def synthesize_program(self, task_id: str, state: Any, max_generations: int = 5) -> str:
        """
        Stateful Evolutionary Program Synthesis.
        Generates candidate functions, executes them in the REPL, and uses stack traces for self-improvement.
        """
        # Check core knowledge library first
        if task_id in self.core_knowledge:
            print(f"[{task_id}] Reusing compositional knowledge from Core Library.")
            return self.core_knowledge[task_id]
            
        print(f"[{task_id}] Beginning Evolutionary Synthesis Loop...")
        
        system_prompt = (
            "You are an evolutionary program synthesizer. Write a Python function `solve(state)` "
            "that processes the given ARC environment state dict and returns a valid action dict or transformed grid. "
            "Do NOT include examples or markdown text outside of the code block. "
            "Only return standard Python code wrapped in ```python ... ```."
        )
        
        feedback = f"Current State: {str(state)[:500]}..."
        
        for generation in range(max_generations):
            prompt = f"{system_prompt}\n\nFeedback from previous run (if any):\n{feedback}\n\nGenerate your code:"
            response = self.llm.generate(prompt)
            
            # Extract code from markdown
            code = response
            import re
            match = re.search(r'```(?:python)?\s*(.*?)```', response, re.DOTALL | re.IGNORECASE)
            if match:
                code = match.group(1).strip()
                
            success, result, trace = self.repl.execute(code, state)
            
            if success:
                print(f"[{task_id}] Generation {generation + 1} Succeeded! Saving to Core Knowledge.")
                self._save_knowledge(task_id, code)
                return code
            else:
                print(f"[{task_id}] Generation {generation + 1} Failed. Refining using stack trace...")
                feedback = f"Your previous code crashed with the following traceback:\n{trace}\nFix the logic."
                
        print(f"[{task_id}] Failed to synthesize valid program after {max_generations} generations.")
        return ""
