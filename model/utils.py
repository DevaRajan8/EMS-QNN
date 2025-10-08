# model/utils.py
"""
Contains utility functions for building and running the Quantum Reservoir Computer.
This includes circuit creation, observable definition, state generation, and evaluation.
"""
import numpy as np
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from sklearn.metrics import mean_squared_error, r2_score

def create_enhanced_reservoir_circuit(num_data_features, num_recurrent_features, num_qubits=8, depth=3):
    
    qc = QuantumCircuit(num_qubits)
    data_params = ParameterVector('d', num_data_features)
    recurrent_params = ParameterVector('r', num_recurrent_features)
    
    # Enhanced data encoding with multiple rotation axes
    for i in range(num_data_features):
        qubit_idx = i % num_qubits
        qc.ry(data_params[i], qubit_idx)
        if i < num_qubits:
            qc.rz(data_params[i] * 0.5, qubit_idx)
    
    # Recurrent state encoding
    for i in range(min(num_recurrent_features, num_qubits)):
        qc.rx(recurrent_params[i], i)
    
    qc.barrier()
    

    for _ in range(depth):
   
        for i in range(num_qubits - 1):
            qc.cz(i, (i + 1) % num_qubits)
        for i in range(0, num_qubits - 1, 2):
            qc.cz(i, (i + 2) % num_qubits)
        
 
        for i in range(num_qubits):
            qc.ry(np.pi * (0.3 + 0.1 * i), i)
            qc.rz(np.pi * (0.7 + 0.15 * i), i)
        
       
        for i in range(0, num_qubits, 2):
            qc.h(i)
    
    return qc

def create_observables(num_qubits):

    observables = []

    for i in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[i] = 'Z'
        observables.append(SparsePauliOp(''.join(pauli_string)))
    

    for i in range(num_qubits - 1):
        pauli_string = ['I'] * num_qubits
        pauli_string[i] = 'Z'
        pauli_string[i + 1] = 'Z'
        observables.append(SparsePauliOp(''.join(pauli_string)))
    

    for i in range(0, num_qubits, 2):
        pauli_string_x = ['I'] * num_qubits
        pauli_string_x[i] = 'X'
        observables.append(SparsePauliOp(''.join(pauli_string_x)))
        
        if i + 1 < num_qubits:
            pauli_string_y = ['I'] * num_qubits
            pauli_string_y[i + 1] = 'Y'
            observables.append(SparsePauliOp(''.join(pauli_string_y)))
            
    return observables

def get_enhanced_reservoir_states(X, circuit, estimator, observables, memory_decay=0.9):
    
    num_samples, seq_length, _ = X.shape
    
    final_reservoir_states = []
    
    for i in tqdm(range(num_samples), desc="Processing reservoir states"):
        recurrent_state = np.zeros(circuit.num_qubits)
        
        for t in range(seq_length):
            combined_input = np.concatenate([X[i, t, :], recurrent_state])
            
            # Prepare for batch execution with the estimator
            circuits = [circuit] * len(observables)
            parameter_values = [combined_input] * len(observables)
            
            # Run the quantum job
            job = estimator.run(circuits, observables, parameter_values)
            result = job.result()
            measurement_values = np.array(result.values)
            
            # Apply memory decay and update the recurrent state
            recurrent_state = memory_decay * recurrent_state + \
                              (1 - memory_decay) * measurement_values[:circuit.num_qubits]
        
        final_reservoir_states.append(measurement_values)
    
    return np.array(final_reservoir_states)

def evaluate_predictions(y_true, y_pred, target_names):
    
    results = {}
    for i, name in enumerate(target_names):
        mse = mean_squared_error(y_true[:, i], y_pred[:, i])
        r2 = r2_score(y_true[:, i], y_pred[:, i])
        results[name] = {'MSE': mse, 'R2': r2}
    return results