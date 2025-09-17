from qiskit_aer import Aer
# from qiskit import transpile
from qiskit.quantum_info import Statevector
import numpy as np
from model import QuantumLSTMCell
from state import measure_quantum_state

class QLSTM:
    def __init__(self, input_size=6, hidden_size=4, output_size=3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = QuantumLSTMCell(n_qubits=hidden_size)
        
        # Classical output layer
        self.W_out = np.random.randn(hidden_size, output_size) * 0.1
        self.b_out = np.zeros(output_size)
        
        # Initialize quantum parameters
        self.init_quantum_params()
        
    def init_quantum_params(self):
        """Initialize quantum circuit parameters"""
        self.quantum_params = {}
        gate_prefixes = ['f', 'i', 'c', 'o']  # forget, input, candidate, output
        
        for prefix in gate_prefixes:
            for i in range(self.hidden_size):
                self.quantum_params[f'theta_{prefix}_{i}'] = np.random.uniform(0, 2*np.pi)
    
    # In passes.py, fix the forward_step method:
    def forward_step(self, x_t, h_prev, c_prev):
        """Single QLSTM forward step"""
        x_t = np.array(x_t)
        h_prev = np.array(h_prev)
        c_prev = np.array(c_prev)
        
        combined_input = np.concatenate([x_t[:2], h_prev[:2]])
        
        # Quantum gates
        forget_circuit = self.cell.create_quantum_gate('forget', combined_input)
        f_t = measure_quantum_state(forget_circuit, self.quantum_params)
        
        input_circuit = self.cell.create_quantum_gate('input', combined_input)
        i_t = measure_quantum_state(input_circuit, self.quantum_params)
        
        candidate_circuit = self.cell.create_quantum_gate('candidate', combined_input)
        c_candidate = measure_quantum_state(candidate_circuit, self.quantum_params)
        
        output_circuit = self.cell.create_quantum_gate('output', combined_input)
        o_t = measure_quantum_state(output_circuit, self.quantum_params)
        
        # Cell and hidden state updates
        c_t = f_t * c_prev + i_t * c_candidate
        h_t = o_t * np.tanh(c_t)
        
        return h_t, c_t
    
    def forward(self, X_batch):
        """Forward pass through entire sequence"""
        batch_size, seq_len, input_size = X_batch.shape
        predictions = []
        
        for b in range(batch_size):
            h_t = np.zeros(self.hidden_size)
            c_t = np.zeros(self.hidden_size)
            
            # Process sequence
            for t in range(seq_len):
                h_t, c_t = self.forward_step(X_batch[b, t], h_t, c_t)
            
            # Final prediction (classical layer)
            prediction = np.dot(h_t, self.W_out) + self.b_out
            predictions.append(prediction)
            
        return np.array(predictions)