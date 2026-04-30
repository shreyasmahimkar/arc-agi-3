# It is assumed that you have already installed the required packages
# in your terminal or command prompt using the following command:
# pip install numpy numba

import numpy as np
from numba import njit, prange
import time

# Numba-optimized helper function to apply a single-qubit gate.
# Using @njit tells Numba to compile this function into fast machine code.
# 'parallel=True' enables automatic parallelization for loops like 'prange'.
@njit(parallel=True, cache=True)
def apply_single_qubit_gate_jit(state, gate, target_qubit):
    """
    Applies a single-qubit gate to the quantum state vector. This function
    is designed to be compiled by Numba for high performance.

    Args:
        state (np.ndarray): The complex state vector of the quantum system.
        gate (np.ndarray): The 2x2 unitary matrix for the gate.
        target_qubit (int): The qubit index to apply the gate to.
    """
    # The stride determines the distance between the two state vector elements
    # that are affected by the gate on the target qubit.
    stride = 1 << target_qubit
    num_elements = len(state)
    
    # We can process the state vector in parallel.
    # The loop iterates over pairs of amplitudes that need to be updated.
    for i in prange(num_elements // 2):
        # This indexing trick groups pairs of states that differ only at the
        # target_qubit position. For example, |...0...⟩ and |...1...⟩.
        i0 = (i // stride) * 2 * stride + (i % stride)
        i1 = i0 + stride
        
        # Cache the values before updating them.
        val0 = state[i0]
        val1 = state[i1]
        
        # Apply the matrix multiplication: [new_val0, new_val1] = gate @ [val0, val1]
        state[i0] = gate[0, 0] * val0 + gate[0, 1] * val1
        state[i1] = gate[1, 0] * val0 + gate[1, 1] * val1
        
    return state


class LocalSimulator:
    """
    A simple state vector simulator for quantum circuits, optimized with Numba.
    """
    def __init__(self, num_qubits):
        """
        Initializes the simulator.

        Args:
            num_qubits (int): The number of qubits for the simulation.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        
        self.num_qubits = num_qubits
        self.state_vector_size = 1 << num_qubits  # Equivalent to 2**num_qubits
        
        # Initialize the state to the |00...0> state.
        # The state is represented by a complex-valued NumPy array.
        self.state = np.zeros(self.state_vector_size, dtype=np.complex128)
        self.state[0] = 1.0 + 0.0j

    def apply_gate(self, gate, target_qubit):
        """
        Applies a single-qubit gate to the simulator's state vector.

        This method is a wrapper around the fast, Numba-jitted function.

        Args:
            gate (np.ndarray): The 2x2 unitary matrix representing the gate.
            target_qubit (int): The index of the qubit to apply the gate to.
        """
        if not (0 <= target_qubit < self.num_qubits):
            raise ValueError(f"Target qubit index {target_qubit} is out of bounds for {self.num_qubits} qubits.")
        if not isinstance(gate, np.ndarray) or gate.shape != (2, 2):
            raise ValueError("Gate must be a 2x2 NumPy array.")

        # Call the highly optimized, JIT-compiled function
        self.state = apply_single_qubit_gate_jit(
            self.state, gate, target_qubit
        )

    def get_state(self):
        """Returns the current state vector of the simulation."""
        return self.state

    def __repr__(self):
        return f"LocalSimulator(num_qubits={self.num_qubits})"


# --- Example Usage ---
if __name__ == "__main__":
    print("--- Numba-optimized Quantum Simulator Demonstration ---")

    # Define some common quantum gates
    H_GATE = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    X_GATE = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    num_qubits = 12  # A system large enough to notice the performance benefit
    print(f"\nInitializing a {num_qubits}-qubit simulator.")
    
    sim = LocalSimulator(num_qubits)

    print("\nInitial state (first 4 elements):")
    print(sim.get_state()[:4])
    
    # --- First run: Numba compiles the function ---
    print("\nApplying Hadamard gate to qubit 0...")
    print("The first run will be slower because Numba is compiling the function in the background.")
    start_time_first = time.time()
    sim.apply_gate(H_GATE, 0)
    end_time_first = time.time()
    print("State after H on qubit 0 (first 4 elements):")
    print(sim.get_state()[:4])
    print(f"Time taken (with JIT compilation): {end_time_first - start_time_first:.6f} seconds")

    # --- Second run: Uses the cached, compiled machine code ---
    print("\nApplying X gate to qubit 1...")
    print("Subsequent runs will be much faster, using the cached compiled code.")
    start_time_second = time.time()
    sim.apply_gate(X_GATE, 1)
    end_time_second = time.time()
    print("State after X on qubit 1 (first 4 elements):")
    print(sim.get_state()[:4])
    print(f"Time taken (cached execution): {end_time_second - start_time_second:.6f} seconds")