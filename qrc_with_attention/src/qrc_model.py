# src/qrc_model.py
import numpy as np
from tqdm import tqdm
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector, state_fidelity
from qiskit_aer.primitives import Estimator as AerEstimator

class QuantumAttentionModel:
    def __init__(self, config):
        self.config = config
        self.circuit = None
        self.observables = self._create_observables()
        self.estimator = AerEstimator(
            backend_options={"method": "statevector", "device": "CPU"},
            transpile_options={"optimization_level": 3}
        )

    def _build_circuit(self, num_data_features):
        cfg = self.config
        qc = QuantumCircuit(cfg['num_qubits'])
        data_params = ParameterVector('d', num_data_features)
        recurrent_params = ParameterVector('r', cfg['num_qubits'])
        
        for i in range(num_data_features):
            idx = i % cfg['num_qubits']
            qc.ry(data_params[i], idx); qc.rz(data_params[i] * 0.5, idx)
        for i in range(cfg['num_qubits']):
            qc.rx(recurrent_params[i], i)
        qc.barrier()
        for _ in range(cfg['depth']):
            for i in range(cfg['num_qubits'] - 1): qc.cz(i, (i + 1) % cfg['num_qubits'])
            for i in range(0, cfg['num_qubits'] - 1, 2): qc.cz(i, (i + 2) % cfg['num_qubits'])
            for i in range(cfg['num_qubits']):
                qc.ry(np.pi * (0.3 + 0.1 * i), i); qc.rz(np.pi * (0.7 + 0.15 * i), i)
            for i in range(0, cfg['num_qubits'], 2): qc.h(i)
        self.circuit = qc

    def _create_observables(self):
        num_qubits = self.config['num_qubits']
        obs = []
        for i in range(num_qubits):
            p_str = ['I'] * num_qubits; p_str[i] = 'Z'; obs.append(SparsePauliOp("".join(p_str)))
        for i in range(num_qubits - 1):
            p_str = ['I'] * num_qubits; p_str[i] = 'Z'; p_str[i+1] = 'Z'; obs.append(SparsePauliOp("".join(p_str)))
        for i in range(0, num_qubits, 2):
            p_str_x = ['I'] * num_qubits; p_str_x[i] = 'X'; obs.append(SparsePauliOp("".join(p_str_x)))
            if i + 1 < num_qubits:
                p_str_y = ['I'] * num_qubits; p_str_y[i+1] = 'Y'; obs.append(SparsePauliOp("".join(p_str_y)))
        return obs

    def transform(self, X):
        if self.circuit is None:
            self._build_circuit(X.shape[2])
        
        all_features, all_weights = [], []
        for i in tqdm(range(X.shape[0]), desc="Processing with Quantum Attention"):
            features, weights = self._process_sequence(X[i])
            all_features.append(features)
            all_weights.append(weights)
        return np.array(all_features), np.array(all_weights)

    def _process_sequence(self, X_seq):
        recurrent_state = np.zeros(self.config['num_qubits'])
        all_measurements, all_states = [], []
        
        for t in range(X_seq.shape[0]):
            combined_input = np.concatenate([X_seq[t, :], recurrent_state])
            if self.config['attention_type'] in ['fidelity', 'hybrid']:
                all_states.append(Statevector.from_instruction(self.circuit.assign_parameters(combined_input)))
            
            job = self.estimator.run([self.circuit] * len(self.observables), self.observables, [combined_input] * len(self.observables))
            measurement = np.array(job.result().values)
            all_measurements.append(measurement)
            recurrent_state = self.config['memory_decay'] * recurrent_state + (1 - self.config['memory_decay']) * measurement[:self.config['num_qubits']]

        # Attention calculation
        if self.config['attention_type'] == 'fidelity':
            weights = self._compute_fidelity_attention(all_states)
        elif self.config['attention_type'] == 'hybrid':
            weights = self._compute_hybrid_attention(all_states, all_measurements)
        else: # 'measurement'
            weights = self._compute_measurement_attention(all_measurements)
        
        attended_features = np.sum(np.array(all_measurements) * weights[:, np.newaxis], axis=0)
        return attended_features, weights
        
    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x)); return exp_x / (np.sum(exp_x) + 1e-10)
    
    def _compute_fidelity_attention(self, all_states):
        fidelities = [state_fidelity(all_states[-1], s) for s in all_states]
        return self._softmax(np.array(fidelities) / self.config['temperature'])
        
    def _compute_measurement_attention(self, all_measurements):
        q = all_measurements[-1]
        sims = [np.dot(q, m) / (np.linalg.norm(q) * np.linalg.norm(m) + 1e-10) for m in all_measurements]
        return self._softmax(np.array(sims) / self.config['temperature'])
        
    def _compute_hybrid_attention(self, all_states, all_measurements, alpha=0.5):
        w_fid = self._compute_fidelity_attention(all_states)
        w_meas = self._compute_measurement_attention(all_measurements)
        hybrid = alpha * w_fid + (1 - alpha) * w_meas
        return hybrid / (np.sum(hybrid) + 1e-10)