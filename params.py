import numpy as np

def update_quantum_params(model, X_batch, y_batch, learning_rate=0.01):
    """Update quantum parameters using parameter shift rule"""
    shift = np.pi / 2  # Parameter shift for quantum gradients
    
    def compute_loss(model_instance):
        predictions = model_instance.forward(X_batch)
        return np.mean((predictions - y_batch)**2)
    
    for param_name in model.quantum_params.keys():
        # Parameter shift rule for quantum gradients
        original_value = model.quantum_params[param_name]
        
        # Forward pass with +shift
        model.quantum_params[param_name] = original_value + shift
        loss_plus = compute_loss(model)
        
        # Forward pass with -shift  
        model.quantum_params[param_name] = original_value - shift
        loss_minus = compute_loss(model)
        
        # Gradient approximation
        gradient = (loss_plus - loss_minus) / 2
        
        # Update parameter with clipping
        new_value = original_value - learning_rate * gradient
        model.quantum_params[param_name] = np.clip(new_value, 0, 2*np.pi)