import numpy as np
from numba import njit
import time
import random

# --- Numba Optimized Core Logic ---
# This standalone function contains the performance-critical calculations.
# @njit decorator compiles it to fast machine code.
# Note: Numba works best with NumPy arrays and simple Python types.
@njit
def _numba_step(concentrations, temperature, pH, action, dt, k_values, delta_H, Ea, R):
    """
    Performs one step of the chemical reaction simulation.
    This function is optimized with Numba's Just-In-Time compiler.
    """
    # Unpack parameters for clarity
    k1_ref, k2_ref, k3_ref = k_values
    T_ref = 298.15  # Reference temperature in Kelvin for Arrhenius equation

    # Apply actions: actions are changes to Temperature and pH
    # Action[0] is delta_T, Action[1] is delta_pH
    temperature += action[0]
    pH += action[1]

    # Clamp temperature and pH to realistic bounds
    temperature = max(273.15, min(373.15, temperature))  # 0°C to 100°C
    pH = max(1.0, min(14.0, pH))

    # Arrhenius equation to calculate temperature-dependent rate constants
    k1 = k1_ref * np.exp(Ea[0] / R * (1/T_ref - 1/temperature))
    k2 = k2_ref * np.exp(Ea[1] / R * (1/T_ref - 1/temperature))
    k3 = k3_ref * np.exp(Ea[2] / R * (1/T_ref - 1/temperature))

    # Unpack concentrations
    Ca, Cb, Cc, Cd = concentrations

    # Reaction rates (simplified model)
    # r1: A + B -> C
    # r2: C -> A + B (reverse)
    # r3: C -> D
    r1 = k1 * Ca * Cb
    r2 = k2 * Cc
    r3 = k3 * Cc

    # Update concentrations based on reaction rates over the time step 'dt'
    dCa = (r2 - r1) * dt
    dCb = (r2 - r1) * dt
    dCc = (r1 - r2 - r3) * dt
    dCd = r3 * dt

    new_concentrations = concentrations + np.array([dCa, dCb, dCc, dCd])
    
    # Ensure concentrations do not go below zero
    new_concentrations = np.maximum(0, new_concentrations)

    # Update temperature based on reaction enthalpy
    # This is a simplified model of heat generation/consumption
    heat_generated = (r1 * -delta_H[0] + r2 * delta_H[0] + r3 * -delta_H[2]) * dt
    # Assuming some heat capacity for the system (e.g., 4184 J/L·K for water)
    heat_capacity = 4184 
    delta_T_reaction = heat_generated / heat_capacity
    new_temperature = temperature + delta_T_reaction

    # --- Reward Calculation ---
    # The goal is to maximize the concentration of product D,
    # while minimizing the concentration of A and B, and keeping C moderate.
    reward = (new_concentrations[3] - Cd) * 10.0  # High reward for producing D
    reward -= (new_concentrations[0] + new_concentrations[1]) * 0.1 # Penalty for unreacted A and B
    
    # Penalty for extreme temperatures or pH values to encourage stability
    if new_temperature > 350 or new_temperature < 280:
        reward -= 1
    if pH > 9 or pH < 5:
        reward -= 1

    # --- Done Condition ---
    # Episode ends if concentrations are depleted or product D reaches a high level
    done = bool(new_concentrations[0] < 0.01 or new_concentrations[1] < 0.01 or new_concentrations[3] > 0.95)

    return new_concentrations, new_temperature, pH, reward, done

class LocalSimulator:
    """
    A class-based wrapper for the simulation. It holds the simulation's
    state and constants, and calls the Numba-optimized function for the core logic.
    """
    def __init__(self):
        # State variables
        self.concentrations = np.array([1.0, 1.0, 0.0, 0.0])  # [Ca, Cb, Cc, Cd]
        self.temperature = 298.15  # Kelvin
        self.pH = 7.0
        
        # Simulation constants
        self.dt = 0.1  # Time step in seconds
        self.R = 8.314  # Gas constant

        # Reaction-specific constants
        self.k_values = np.array([0.1, 0.05, 0.02])  # [k1_ref, k2_ref, k3_ref] at T_ref
        self.delta_H = np.array([-50000, 50000, -20000]) # Enthalpy changes in J/mol for r1, r2, r3
        self.Ea = np.array([40000, 60000, 30000]) # Activation energies in J/mol for k1, k2, k3

    def get_state(self):
        """Returns the current state as a single NumPy array."""
        return np.concatenate([self.concentrations, [self.temperature, self.pH]])

    def reset(self):
        """Resets the environment to its initial state."""
        self.concentrations = np.array([1.0, 1.0, 0.0, 0.0])
        self.temperature = 298.15
        self.pH = 7.0
        return self.get_state()

    def step(self, action):
        """
        Executes one time step in the simulation.
        
        Args:
            action (np.array): A 2-element array containing [delta_T, delta_pH].
        
        Returns:
            tuple: A tuple containing (next_state, reward, done).
        """
        new_concentrations, new_temperature, new_pH, reward, done = _numba_step(
            self.concentrations,
            self.temperature,
            self.pH,
            action,
            self.dt,
            self.k_values,
            self.delta_H,
            self.Ea,
            self.R
        )
        
        # Update internal state
        self.concentrations = new_concentrations
        self.temperature = new_temperature
        self.pH = new_pH
        
        next_state = self.get_state()
        
        return next_state, reward, done

def get_random_action():
    """Generates a random action for the simulation."""
    # Small random changes to temperature and pH
    delta_T = random.uniform(-1.0, 1.0)  # Change temperature by -1 to +1 K
    delta_pH = random.uniform(-0.1, 0.1) # Change pH by -0.1 to +0.1
    return np.array([delta_T, delta_pH])

def simulate_episodes(simulator, num_episodes, max_steps_per_episode):
    """
    Runs multiple simulation episodes and stores the results.
    
    Returns:
        list: A list of episodes. Each episode is a list of transition dictionaries.
    """
    episodic_memory = []
    
    for i in range(num_episodes):
        state = simulator.reset()
        episode_memory = []
        
        for t in range(max_steps_per_episode):
            action = get_random_action()
            next_state, reward, done = simulator.step(action)
            
            # Store the transition
            memory_entry = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            }
            episode_memory.append(memory_entry)
            
            state = next_state
            
            if done:
                break
                
        episodic_memory.append(episode_memory)
        print(f"Episode {i+1}/{num_episodes} finished after {t+1} steps.")
        
    return episodic_memory

def main():
    """
    Main function to set up and run the simulation.
    """
    print("Initializing Local Simulator...")
    simulator = LocalSimulator()
    
    # --- Performance Comparison ---
    # First, run a step to trigger Numba's JIT compilation.
    print("Warming up Numba JIT compiler...")
    simulator.step(get_random_action())
    simulator.reset()
    
    # Time the Numba-optimized simulation
    num_steps_timing = 10000
    start_time_numba = time.time()
    for _ in range(num_steps_timing):
        simulator.step(get_random_action())
    end_time_numba = time.time()
    
    numba_duration = end_time_numba - start_time_numba
    print(f"Numba-optimized simulation of {num_steps_timing} steps took: {numba_duration:.4f} seconds.")
    print("-" * 30)

    # --- Run Full Simulation ---
    num_episodes = 5
    max_steps_per_episode = 200
    print(f"Simulating {num_episodes} episodes with max {max_steps_per_episode} steps each...")
    
    episodic_memory = simulate_episodes(simulator, num_episodes, max_steps_per_episode)
    
    print("\n--- Simulation Results ---")
    
    # Check if any episodes were generated
    if not episodic_memory or not episodic_memory[0]:
        print("Simulation did not produce any data.")
        return

    # --- FIX IS HERE ---
    # The original code had `episodic_memory[0]["state_before"]`, which caused a KeyError.
    # The correct way to get the initial state of the first episode is to access
    # the 'state' from the first time step (dictionary) of the first episode (list).
    initial_state = episodic_memory[0][0]["state"]
    
    # Get the final state from the last time step of the first episode.
    final_transition = episodic_memory[0][-1]
    final_state = final_transition["next_state"]

    print(f"Number of episodes recorded: {len(episodic_memory)}")
    print(f"Number of steps in the first episode: {len(episodic_memory[0])}")
    
    print("\nInitial state of the first episode:")
    print(f"  Concentrations [A, B, C, D]: {initial_state[:4]}")
    print(f"  Temperature: {initial_state[4]:.2f} K")
    print(f"  pH: {initial_state[5]:.2f}")
    
    print("\nFinal state of the first episode:")
    print(f"  Concentrations [A, B, C, D]: {final_state[:4]}")
    print(f"  Temperature: {final_state[4]:.2f} K")
    print(f"  pH: {final_state[5]:.2f}")
    print(f"  Episode ended because 'done' was: {final_transition['done']}")

if __name__ == "__main__":
    main()