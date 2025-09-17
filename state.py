# state.py
from qiskit.quantum_info import Statevector
import numpy as np

def measure_quantum_state(circuit, params_dict):
    """Get quantum state amplitudes as classical values"""
    # Handle parameters
    if circuit.parameters:
        binding_dict = {}
        for param in circuit.parameters:
            param_name = param.name
            binding_dict[param] = params_dict.get(param_name, 0.1)
        
        bound_circuit = circuit.assign_parameters(binding_dict)
    else:
        bound_circuit = circuit
    
    # Create statevector from circuit
    statevector = Statevector.from_instruction(bound_circuit)
    
    # Get amplitudes
    amplitudes = np.abs(statevector.data)
    
    # Return first 4 amplitudes (matching hidden_size)
    return amplitudes[:4]