import numpy as np
from numba import njit

# --- Numba-optimized Core Functions ---

@njit(cache=True)
def _apply_single_qubit_gate(state_vector, gate_matrix, target_qubit):
    """
    Applies a single-qubit gate to the state vector.

    Args:
        state_vector (np.ndarray): The complex state vector of the quantum system.
        gate_matrix (np.ndarray): The 2x2 unitary matrix of the gate.
        target_qubit (int): The qubit index to apply the gate to.
    """
    num_qubits = int(np.log2(state_vector.size))
    stride = 1 << target_qubit
    
    # Iterate through the state vector, applying the gate to pairs of amplitudes.
    # The outer loops iterate through all states, skipping over the target qubit.
    # The inner loop applies the 2x2 matrix.
    for i in range(1 << (num_qubits - 1)):
        # Create a mask to insert the outer loop index 'i' around the target qubit bit
        mask = (i << (target_qubit + 1)) | (i & (stride - 1))
        
        idx0 = mask
        idx1 = mask | stride

        v0 = state_vector[idx0]
        v1 = state_vector[idx1]

        state_vector[idx0] = gate_matrix[0, 0] * v0 + gate_matrix[0, 1] * v1
        state_vector[idx1] = gate_matrix[1, 0] * v0 + gate_matrix[1, 1] * v1

@njit(cache=True)
def _apply_controlled_gate(state_vector, gate_matrix, control_qubit, target_qubit):
    """
    Applies a controlled two-qubit gate with a 2x2 target matrix.

    Args:
        state_vector (np.ndarray): The complex state vector.
        gate_matrix (np.ndarray): The 2x2 unitary matrix for the target qubit.
        control_qubit (int): The control qubit index.
        target_qubit (int): The target qubit index.
    """
    control_mask = 1 << control_qubit
    target_stride = 1 << target_qubit
    
    # Iterate through all state indices
    for i in range(state_vector.size):
        # Apply gate only if control bit is 1
        if (i & control_mask) != 0:
            # And if we are at the base of a pair (target bit is 0) to avoid double-processing
            if (i & target_stride) == 0:
                idx0 = i
                idx1 = i | target_stride
                
                v0 = state_vector[idx0]
                v1 = state_vector[idx1]

                state_vector[idx0] = gate_matrix[0, 0] * v0 + gate_matrix[0, 1] * v1
                state_vector[idx1] = gate_matrix[1, 0] * v0 + gate_matrix[1, 1] * v1

# --- Simulator Class ---

class LocalSimulator:
    """
    A simple quantum circuit simulator that runs on a local machine.
    Uses Numba to accelerate gate applications.

    The simulator uses a "little-endian" convention where the state vector
    indices correspond to the integer value of the qubit basis states
    written as |q_{n-1}...q_1q_0⟩. For example, for 3 qubits, the state
    |110⟩ corresponds to qubit 2 being 1, qubit 1 being 1, and qubit 0
    being 0, which is index 1*2^2 + 1*2^1 + 0*2^0 = 6.
    """
    def __init__(self, num_qubits):
        """
        Initializes the simulator.

        Args:
            num_qubits (int): The number of qubits in the circuit.
        """
        if not isinstance(num_qubits, int) or num_qubits <= 0:
            raise ValueError("Number of qubits must be a positive integer.")
        
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(1 << num_qubits, dtype=np.complex128)
        self.state_vector[0] = 1.0

    def _get_qubit_indices(self, qubits):
        """Helper to validate and return qubit indices."""
        indices = [qubits] if isinstance(qubits, int) else list(qubits)
        for q in indices:
            if not (0 <= q < self.num_qubits):
                raise ValueError(f"Qubit index {q} is out of bounds for {self.num_qubits} qubits.")
        return indices

    def h(self, target_qubit):
        """Applies a Hadamard gate."""
        q = self._get_qubit_indices(target_qubit)[0]
        h_matrix = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
        _apply_single_qubit_gate(self.state_vector, h_matrix, q)

    def x(self, target_qubit):
        """Applies a Pauli-X (NOT) gate."""
        q = self._get_qubit_indices(target_qubit)[0]
        x_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        _apply_single_qubit_gate(self.state_vector, x_matrix, q)

    def cnot(self, control_qubit, target_qubit):
        """Applies a Controlled-NOT (CNOT) gate."""
        c, t = self._get_qubit_indices([control_qubit, target_qubit])
        if c == t:
            raise ValueError("Control and target qubits cannot be the same.")
        
        x_matrix = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        _apply_controlled_gate(self.state_vector, x_matrix, c, t)
        
    def measure(self):
        """
        Measures the state of the system in the computational basis.

        Returns:
            int: The collapsed state, represented as an integer.
        """
        probabilities = np.abs(self.state_vector)**2
        # Normalize to handle potential floating point inaccuracies
        probabilities /= np.sum(probabilities)
        
        # np.random.choice is not supported by Numba, so this part remains in Python
        result = np.random.choice(1 << self.num_qubits, p=probabilities)
        return result

    def get_state_vector(self):
        """Returns a copy of the current state vector."""
        return self.state_vector.copy()

    def __str__(self):
        return f"LocalSimulator(num_qubits={self.num_qubits})"

# --- Example Usage ---

def create_bell_state():
    """Demonstrates creating a Bell state |Φ+⟩."""
    print("--- Creating Bell State |Φ+⟩ ---")
    sim = LocalSimulator(2)
    
    # Apply Hadamard to qubit 0
    sim.h(0)
    print("State after H(0):", np.round(sim.get_state_vector(), 3))
    # Expected state for H on q0: 1/sqrt(2)(|00> + |10>)
    # But due to our |q1q0> convention, h(0) applies I⊗H, giving 1/sqrt(2)(|00> + |01>)
    
    # Apply CNOT with qubit 0 as control and 1 as target
    sim.cnot(0, 1)
    print("State after CNOT(0, 1):", np.round(sim.get_state_vector(), 3))
    # This circuit (I⊗H followed by CNOT_0,1) produces the Bell state.
    # Expected final state: [0.707, 0, 0, 0.707] which is 1/sqrt(2)(|00> + |11>)
    
    print("\nPerforming 1000 measurements...")
    counts = {}
    for _ in range(1000):
        measurement = sim.measure()
        counts[measurement] = counts.get(measurement, 0) + 1
        
    print("Measurement results (integer |binary⟩):")
    for result in sorted(counts.keys()):
        print(f"  State {result} |{result:02b}⟩: {counts[result]} times")
    print("-" * 30)

def create_ghz_state():
    """Demonstrates creating a 3-qubit GHZ state."""
    print("\n--- Creating 3-qubit GHZ State ---")
    num_qubits = 3
    sim = LocalSimulator(num_qubits)
    
    # Apply H to qubit 2 (the most significant qubit in our |q2q1q0> convention)
    sim.h(2)
    
    # Chain of CNOTs
    sim.cnot(2, 1)
    sim.cnot(2, 0)
    
    print("Final GHZ state vector:", np.round(sim.get_state_vector(), 3))
    # Expected state: 1/sqrt(2)(|000> + |111>) -> [0.707, 0, ..., 0, 0.707]
    
    print("\nPerforming 1000 measurements...")
    counts = {}
    for _ in range(1000):
        measurement = sim.measure()
        counts[measurement] = counts.get(measurement, 0) + 1
        
    print("Measurement results (integer |binary⟩):")
    for result in sorted(counts.keys()):
        print(f"  State {result} |{result:0{num_qubits}b}⟩: {counts[result]} times")
    print("-" * 30)


if __name__ == '__main__':
    # Numba JIT compilation happens on the first call.
    # "Warm up" the functions to avoid measuring compilation time in the actual examples.
    print("Warming up Numba JIT compiler...")
    warmup_sim = LocalSimulator(2)
    warmup_sim.h(0)
    warmup_sim.cnot(0, 1)
    print("Warm-up complete.\n")
    
    create_bell_state()
    create_ghz_state()