import numpy as np
from numba import njit, prange

# ==============================================================================
# Numba-JIT Compiled Simulation Core
#
# This section contains the performance-critical code. It is designed to
# work exclusively with NumPy arrays and primitive types for maximum optimization.
# It does not handle any Python-specific objects like dictionaries or strings.
# ==============================================================================

@njit
def apply_gate(state, gate_matrix, target_qubit, num_qubits):
    """
    Applies a single-qubit gate to the state vector. This operation is
    optimized for Numba.
    """
    num_states = 1 << num_qubits
    stride = 1 << target_qubit
    for i in range(0, num_states, 2 * stride):
        for j in range(stride):
            idx0 = i + j
            idx1 = i + j + stride

            v0 = state[idx0]
            v1 = state[idx1]

            state[idx0] = gate_matrix[0, 0] * v0 + gate_matrix[0, 1] * v1
            state[idx1] = gate_matrix[1, 0] * v0 + gate_matrix[1, 1] * v1

@njit
def apply_cnot(state, control_qubit, target_qubit, num_qubits):
    """
    Applies a CNOT gate to the state vector. This is a common 2-qubit gate.
    """
    num_states = 1 << num_qubits
    control_mask = 1 << control_qubit
    target_mask = 1 << target_qubit
    
    for i in range(num_states):
        # Apply CNOT only if the control bit is 1 and we are at the lower index of the pair
        if (i & control_mask) and not (i & target_mask):
            idx0 = i
            idx1 = i | target_mask
            
            # Swap the amplitudes
            temp = state[idx0]
            state[idx0] = state[idx1]
            state[idx1] = temp


@njit(parallel=True)
def _run_circuit_jit(
    num_qubits,
    circuit_ops,
    circuit_qubits,
    circuit_params_indices,
    all_parameters,
    num_shots
):
    """
    The Numba-optimized simulation core. This function is the workhorse.

    Args:
        num_qubits (int): The number of qubits in the circuit.
        circuit_ops (np.ndarray): 1D array of integer IDs for each operation.
        circuit_qubits (np.ndarray): 2D array storing qubit indices for each op.
        circuit_params_indices (np.ndarray): 1D array mapping an op to its parameter.
        all_parameters (np.ndarray): 2D array where rows are shots and columns are parameters.
        num_shots (int): The number of parameter sets to simulate.
    """
    # Define integer constants for gate identification
    RX_ID = 0
    CNOT_ID = 1

    # Array to store the result of each shot
    results = np.zeros(num_shots, dtype=np.float64)

    # The main loop over shots, parallelized with Numba's prange
    for shot_idx in prange(num_shots):
        # Initialize the state vector for this shot to |0...0>
        state = np.zeros(1 << num_qubits, dtype=np.complex128)
        state[0] = 1.0 + 0.0j

        # Get the specific parameters for the current shot
        shot_params = all_parameters[shot_idx]

        # Loop through the circuit operations and apply them
        for op_idx in range(len(circuit_ops)):
            op_id = circuit_ops[op_idx]

            if op_id == RX_ID:
                qubit = circuit_qubits[op_idx, 0]
                param_idx = circuit_params_indices[op_idx]
                angle = shot_params[param_idx]

                # Construct the RX gate matrix
                cos_a = np.cos(angle / 2)
                sin_a = -1j * np.sin(angle / 2)
                rx_matrix = np.array(
                    [[cos_a, sin_a],
                     [sin_a, cos_a]],
                    dtype=np.complex128
                )
                apply_gate(state, rx_matrix, qubit, num_qubits)

            elif op_id == CNOT_ID:
                control = circuit_qubits[op_idx, 0]
                target = circuit_qubits[op_idx, 1]
                apply_cnot(state, control, target, num_qubits)

        # Example measurement: Calculate the probability of qubit 0 being in state |0>
        prob_q0_is_0 = 0.0
        for i in range(1 << num_qubits):
            if (i & 1) == 0:  # Check if the bit for qubit 0 is 0
                prob_q0_is_0 += np.abs(state[i])**2
        results[shot_idx] = prob_q0_is_0

    return results

# ==============================================================================
# Python-Level Simulator Class
#
# This class provides a user-friendly interface. Its main job is to
# translate Python objects (lists, dicts) into NumPy arrays that the
# Numba-jitted core can understand and process efficiently.
# ==============================================================================

class LocalSimulator:
    """
    A quantum circuit simulator that uses a Numba-optimized backend.
    """
    def __init__(self):
        # Map gate names to integer IDs for the Numba core
        self._gate_map = {'rx': 0, 'cnot': 1}

    def run(self, circuit, parameters):
        """
        Prepares data and runs the Numba-optimized simulation.

        Args:
            circuit (list): A list of tuples describing the circuit.
                            e.g., [('rx', 0, 0), ('cnot', 0, 1)]
            parameters (list): A list of parameter sets. Each set can be a
                               dict (e.g., {'p0': 0.5}) or a list/array.
        Returns:
            np.ndarray: An array containing the simulation results for each shot.
        """
        if not circuit or not parameters:
            return np.array([])

        # --- FIX: Convert user-friendly inputs to Numba-friendly NumPy arrays ---

        # 1. Determine circuit properties (num_qubits, num_params)
        max_qubit_idx = 0
        max_param_idx = 0
        for op in circuit:
            gate_name = op[0]
            if gate_name == 'rx':
                max_qubit_idx = max(max_qubit_idx, op[1])
                max_param_idx = max(max_param_idx, op[2])
            elif gate_name == 'cnot':
                max_qubit_idx = max(max_qubit_idx, op[1], op[2])
        
        num_qubits = max_qubit_idx + 1
        num_params = max_param_idx + 1

        # 2. Process the circuit structure into integer-based NumPy arrays
        num_ops = len(circuit)
        circuit_ops = np.zeros(num_ops, dtype=np.int32)
        circuit_qubits = np.zeros((num_ops, 2), dtype=np.int32)
        circuit_params_indices = np.zeros(num_ops, dtype=np.int32)

        for i, op in enumerate(circuit):
            gate_name = op[0]
            circuit_ops[i] = self._gate_map[gate_name]
            if gate_name == 'rx':
                circuit_qubits[i, 0] = op[1]
                circuit_params_indices[i] = op[2]
            elif gate_name == 'cnot':
                circuit_qubits[i, 0] = op[1]
                circuit_qubits[i, 1] = op[2]

        # 3. Process the parameters into a 2D NumPy float array
        # This is the core part of the fix, handling the conversion that
        # prevents the `TypeError: float() argument must be a ... 'dict'`.
        num_shots = len(parameters)
        all_parameters_np = np.zeros((num_shots, num_params), dtype=np.float64)

        if isinstance(parameters[0], dict):
            for i, p_dict in enumerate(parameters):
                for j in range(num_params):
                    key = f'p{j}' # Assumes a convention 'p0', 'p1', etc.
                    if key in p_dict:
                        all_parameters_np[i, j] = p_dict[key]
        else: # Assumes list of lists or numpy array
            all_parameters_np = np.array(parameters, dtype=np.float64)

        # Ensure array is 2D for single-parameter cases
        if all_parameters_np.ndim == 1:
            all_parameters_np = all_parameters_np.reshape(-1, 1)

        # 4. Call the JIT-compiled function with clean, Numba-compatible data
        return _run_circuit_jit(
            num_qubits,
            circuit_ops,
            circuit_qubits,
            circuit_params_indices,
            all_parameters_np,
            num_shots
        )


# ==============================================================================
# Example Usage
# ==============================================================================
if __name__ == '__main__':
    # Define a simple 2-qubit circuit to create a Bell state:
    # - Apply a parameterized rotation (RX) on qubit 0.
    # - Apply a CNOT gate with qubit 0 as control and qubit 1 as target.
    my_circuit = [
        ('rx', 0, 0),   # (gate_name, target_qubit, parameter_index)
        ('cnot', 0, 1)  # (gate_name, control_qubit, target_qubit)
    ]

    # Define the parameters for multiple simulation runs (shots).
    # This list of dictionaries is a user-friendly format that previously
    # would have caused the error if passed directly to a Numba function.
    params_as_dicts = [
        {'p0': 0.0},          # RX(0) is Identity. State -> |00>. Prob(q0=0)=1.0
        {'p0': np.pi / 2},    # Creates a Bell state. Prob(q0=0)=0.5
        {'p0': np.pi}         # RX(pi) is X gate. State -> |11>. Prob(q0=0)=0.0
    ]

    # Create an instance of the simulator
    simulator = LocalSimulator()

    # Run the simulation
    results = simulator.run(my_circuit, params_as_dicts)

    # --- Print Results ---
    print("--- Simulation Results ---")
    print(f"Circuit: {my_circuit}")
    print(f"Input Parameters: {params_as_dicts}")
    print("\nResults (Probability of Qubit 0 being in state |0>):")
    print("-" * 55)
    print(f"{'Input Angle (p0)':<20} | {'Expected Prob.':<15} | {'Actual Result':<15}")
    print("-" * 55)
    
    expected_results = [1.0, 0.5, 0.0]
    for i, res in enumerate(results):
        angle = params_as_dicts[i]['p0']
        expected = expected_results[i]
        print(f"{angle:<20.4f} | {expected:<15.4f} | {res:<15.4f}")
    print("-" * 55)