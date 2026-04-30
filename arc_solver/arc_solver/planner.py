import math
import random
import multiprocessing
import sys
import os
from typing import Any, List, Dict
from arc_solver.perception import parse_state, get_active_coordinates

# Ensure current directory is in path for dynamic imports
sys.path.append(os.getcwd())

def sample_random_action(state: Dict[str, Any]) -> Dict[str, Any]:
    """Helper to sample a random valid action dict for rollouts."""
    grid, available_actions = parse_state(state)
    if not available_actions:
        available_actions = ["ACTION1"]
        
    chosen = random.choice(available_actions)
    action_dict = {"action": chosen}
    
    if chosen == "ACTION6":
        coords = get_active_coordinates(grid)
        if coords:
            x, y = random.choice(coords)
            action_dict["x"] = x
            action_dict["y"] = y
        else:
            action_dict["x"] = 0
            action_dict["y"] = 0
            
    return action_dict

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
        action_dict = sample_random_action(current_state)
        
        try:
            current_state, reward, done = simulator.step(current_state, action_dict)
            if done:
                return float(reward)
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
        """Expands a node with possible actions from available_actions."""
        grid, available_actions = parse_state(node.state)
        if not available_actions:
            available_actions = ["ACTION1"]
            
        for action_str in available_actions:
            if action_str == "ACTION6":
                coords = get_active_coordinates(grid)
                if not coords:
                    coords = [(0, 0)]
                # Dynamically prune search tree by only branching on active non-zero pixels
                for x, y in coords:
                    action_dict = {"action": action_str, "x": x, "y": y}
                    next_state = {"grid": grid.tolist(), "available_actions": available_actions}
                    node.children.append(MCTSNode(state=next_state, parent=node, action=action_dict))
            else:
                action_dict = {"action": action_str}
                next_state = {"grid": grid.tolist(), "available_actions": available_actions}
                node.children.append(MCTSNode(state=next_state, parent=node, action=action_dict))
            
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagates the reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
            
    def plan(self, initial_state: Any) -> List[Dict[str, Any]]:
        """Runs the MCTS algorithm across 4 CPU cores and returns an array of best action dicts."""
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
            return [{"action": "ACTION1"}]
            
        best_child = max(root.children, key=lambda n: n.visits)
        return [best_child.action]
