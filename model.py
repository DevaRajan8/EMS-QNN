from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import Parameter
import qiskit.quantum_info as qi

class QuantumLSTMCell:
    def __init__(self, n_qubits=4):
        self.n_qubits = n_qubits
        self.params = {}
        
    def create_quantum_gate(self, gate_type, input_data):
        """Create quantum circuits for different LSTM gates"""
        qc = QuantumCircuit(self.n_qubits)
        
        for i, angle in enumerate(input_data[:self.n_qubits]):
            qc.ry(angle, i)
        
        if gate_type == 'forget':
            # Forget gate quantum circuit
            for i in range(self.n_qubits-1):
                qc.cx(i, i+1)
                qc.rz(Parameter(f'theta_f_{i}'), i+1)
                
        elif gate_type == 'input':
            # Input gate quantum circuit  
            for i in range(self.n_qubits-1):
                qc.cry(Parameter(f'theta_i_{i}'), i, i+1)
                
        elif gate_type == 'candidate':
            # Candidate gate quantum circuit
            for i in range(self.n_qubits):
                qc.rx(Parameter(f'theta_c_{i}'), i)
            for i in range(self.n_qubits-1):
                qc.cz(i, i+1)
                
        elif gate_type == 'output':
            # Output gate quantum circuit
            qc.h(list(range(self.n_qubits)))
            for i in range(self.n_qubits-1):
                qc.crx(Parameter(f'theta_o_{i}'), i, i+1)
        
        return qc