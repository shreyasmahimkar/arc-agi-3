import math
import random
import multiprocessing
import sys
import os
from typing import Any, List, Dict

# Ensure current directory is in path for dynamic imports
sys.path.append(os.getcwd())

def simulate_rollout(state: Any) -> float:
    """
    Runs a single simulation rollout from the given state.
    Designed to be run in a multiprocessing.Pool worker.
    """
    try:
        from LocalSimulator import LocalSimulator
        simulator = LocalSimulator()
    except ImportError:
        simulator = None
        
    if simulator is None:
        # Fallback if LocalSimulator is not generated
        return random.random()
        
    current_state = state
    depth = 0
    max_depth = 10
    
    while depth < max_depth:
        # Dummy ARC action space
        action = random.choice(["up", "down", "left", "right"])
        
        try:
            current_state, reward, done = simulator.step(current_state, action)
            if done:
                return reward
        except Exception:
            return random.random()
            
        depth += 1
        
    return 0.5  # Neutral reward

class MCTSNode:
    def __init__(self, state: Any, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visits = 0
        self.value = 0.0

class MCTSPlanner:
    def __init__(self, num_simulations: int = 100, num_cores: int = 4):
        self.num_simulations = num_simulations
        self.num_cores = num_cores
        
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Selects a node using UCB1."""
        while node.children:
            node = max(node.children, key=lambda n: n.value / (n.visits + 1e-6) + math.sqrt(2 * math.log(node.visits + 1) / (n.visits + 1e-6)))
        return node
        
    def _expand(self, node: MCTSNode):
        """Expands a node with possible actions."""
        actions = ["up", "down", "left", "right"]
        for action in actions:
            next_state = {"parent_state": node.state, "action": action}
            node.children.append(MCTSNode(state=next_state, parent=node, action=action))
            
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagates the reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
            
    def plan(self, initial_state: Any) -> Any:
        """Runs the MCTS algorithm across 4 CPU cores and returns the best action."""
        root = MCTSNode(state=initial_state)
        
        for _ in range(self.num_simulations):
            leaf = self._select(root)
            
            if leaf.visits > 0:
                self._expand(leaf)
                if leaf.children:
                    leaf = random.choice(leaf.children)
            
            # Using multiprocessing.Pool across 4 CPU cores
            states_to_simulate = [leaf.state for _ in range(self.num_cores)]
            with multiprocessing.Pool(processes=self.num_cores) as pool:
                rewards = pool.map(simulate_rollout, states_to_simulate)
                
            avg_reward = sum(rewards) / len(rewards)
            self._backpropagate(leaf, avg_reward)
            
        if not root.children:
            return "fallback_action"
            
        best_child = max(root.children, key=lambda n: n.visits)
        return best_child.action
