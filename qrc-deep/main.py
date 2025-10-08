
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from qiskit_aer.primitives import Estimator as AerEstimator

from preprocess import load_and_preprocess_data
from model import (
    create_enhanced_reservoir_circuit,
    create_observables,
    get_enhanced_reservoir_states,
    evaluate_predictions
)

def train_readout_with_regularization(X_res, y, alpha_range=[0.1, 1.0, 10.0, 100.0]):
    print("Training the classical readout model...")
    best_alpha = alpha_range[0]
    best_score = -np.inf
    
    for alpha in alpha_range:
        model = Ridge(alpha=alpha)
        model.fit(X_res, y)
        score = model.score(X_res, y)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    
    print(f"Best alpha for Ridge regression: {best_alpha}")
    final_model = Ridge(alpha=best_alpha)
    final_model.fit(X_res, y)
    return final_model

def main():
   
    DATA_FILE_PATH = 'data/Dataset.csv' 
    HIDDEN_SIZE_QUBITS = 8
    SEQ_LENGTH = 20
    DATA_FRACTION = 0.05
    RESERVOIR_DEPTH = 4
    MEMORY_DECAY = 0.70
    TARGET_LABELS = ['PV Production', 'Wind Production', 'Electric Demand']
    
    # --- 1. Data Preprocessing ---
    X_train, y_train, X_test, y_test, scaler_targets = load_and_preprocess_data(
        DATA_FILE_PATH,
        seq_length=SEQ_LENGTH,
        data_fraction=DATA_FRACTION
    )
    
    # --- 2. Model & Estimator Setup ---
    num_data_features = X_train.shape[2]
    num_recurrent_features = HIDDEN_SIZE_QUBITS
    
    qrc_circuit = create_enhanced_reservoir_circuit(
        num_data_features,
        num_recurrent_features,
        HIDDEN_SIZE_QUBITS,
        RESERVOIR_DEPTH
    )
    
    observables = create_observables(HIDDEN_SIZE_QUBITS)
    
    estimator = AerEstimator(
        backend_options={"method": "statevector", "device": "GPU"},
        transpile_options={"optimization_level": 3}
    )
    
    # --- 3. Generate Reservoir States (Feature Extraction) ---
    start_time = time.time()
    reservoir_states_train = get_enhanced_reservoir_states(
        X_train, qrc_circuit, estimator, observables, MEMORY_DECAY
    )
    train_time = time.time() - start_time
    
    start_time = time.time()
    reservoir_states_test = get_enhanced_reservoir_states(
        X_test, qrc_circuit, estimator, observables, MEMORY_DECAY
    )
    test_time = time.time() - start_time
    
    # --- 4. Train Classical Readout Layer ---
    readout_model = train_readout_with_regularization(reservoir_states_train, y_train)
    
    # --- 5. Prediction and Evaluation ---
    predictions_scaled = readout_model.predict(reservoir_states_test)
    predictions = scaler_targets.inverse_transform(predictions_scaled)
    y_test_inv = scaler_targets.inverse_transform(y_test)
    
    results = evaluate_predictions(y_test_inv, predictions, TARGET_LABELS)
    
    print(f"\nTraining time: {train_time:.2f}s")
    print(f"Testing time: {test_time:.2f}s")
    print(f"Reservoir features: {reservoir_states_train.shape[1]}")
    
    for name, metrics in results.items():
        print(f"{name} - MSE: {metrics['MSE']:.4f}, R²: {metrics['R2']:.4f}")

    # --- 6. Plotting Results ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (ax, label) in enumerate(zip(axes, TARGET_LABELS)):
        ax.plot(y_test_inv[:200, i], label=f'Actual {label}', linewidth=2)
        ax.plot(predictions[:200, i], label=f'Predicted {label}',
                linestyle='--', linewidth=2, alpha=0.8)
        ax.set_title(f'Enhanced QRC: {label} Prediction')
        ax.set_xlabel('Time Step')
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('qrc_predictions.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()