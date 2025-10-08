
import torch
import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN

def create_vector_vqc(num_features, num_qubits):
    """Creates a vector-output Variational Quantum Circuit."""
    qc = QuantumCircuit(num_qubits)
    params = ParameterVector('x', num_features)

    for i in range(num_features):
        qc.ry(params[i], i % num_qubits)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)

    observables = [SparsePauliOp("I" * i + "Z" + "I" * (num_qubits - 1 - i)) for i in range(num_qubits)]
    qnn = EstimatorQNN(circuit=qc, observables=observables, input_params=qc.parameters)
    return TorchConnector(qnn)

class QLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_qubits):
        super(QLSTMCell, self).__init__()
        if hidden_size != num_qubits:
            raise ValueError(f"hidden_size ({hidden_size}) must match num_qubits ({num_qubits}).")
        
        num_vqc_inputs = input_size + hidden_size
        self.forget_gate_qnn = create_vector_vqc(num_vqc_inputs, num_qubits)
        self.input_gate_qnn = create_vector_vqc(num_vqc_inputs, num_qubits)
        self.output_gate_qnn = create_vector_vqc(num_vqc_inputs, num_qubits)
        self.candidate_gate_qnn = create_vector_vqc(num_vqc_inputs, num_qubits)

    def forward(self, input_tensor, hidden_state):
        h_prev, c_prev = hidden_state
        combined = torch.cat((input_tensor, h_prev), dim=1)
        
        f_t = torch.sigmoid(self.forget_gate_qnn(combined))
        i_t = torch.sigmoid(self.input_gate_qnn(combined))
        o_t = torch.sigmoid(self.output_gate_qnn(combined))
        c_tilde = torch.tanh(self.candidate_gate_qnn(combined))

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * torch.tanh(c_t)
        return h_t, c_t

class QLSTM(nn.Module):
    def __init__(self, config):
        super(QLSTM, self).__init__()
        self.hidden_size = config['hidden_size']
        self.qlstm_cell = QLSTMCell(config['input_size'], config['hidden_size'], config['num_qubits'])
        self.fc = nn.Linear(config['hidden_size'], config['output_size'])

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=x.device)

        for t in range(seq_len):
            h_t, c_t = self.qlstm_cell(x[:, t, :], (h_t, c_t))
        
        out = self.fc(h_t)
        return out