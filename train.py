from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
import numpy as np
from params import update_quantum_params

def get_batches(X, y, batch_size=4):
    """Create batches from input data"""
    num_samples = len(X)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]

def train_qlstm(model, X_train, y_train, epochs=20, learning_rate=0.01):
    """Training loop for QLSTM"""
    
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_x, batch_y in get_batches(X_train, y_train, batch_size=4):
            try:
                # Forward pass
                predictions = model.forward(batch_x)
                
                # Calculate loss
                loss = np.mean((predictions - batch_y)**2)
                total_loss += loss
                num_batches += 1
                
                # Update quantum parameters
                update_quantum_params(model, batch_x, batch_y, learning_rate)
                
            except Exception as e:
                print(f"Error in batch processing: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    try:
        predictions = model.forward(X_test)
        mse = np.mean((predictions - y_test)**2)
        rmse = np.sqrt(mse)
        
        print(f"Test MSE: {mse:.6f}")
        print(f"Test RMSE: {rmse:.6f}")
        
        return predictions, mse, rmse
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None, float('inf'), float('inf')