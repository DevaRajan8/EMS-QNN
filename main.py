import pandas as pd
import numpy as np
from preprocess import preprocess_data
from split import create_sequences, split_data
from passes import QLSTM
from train import train_qlstm, evaluate_model

def main():
    try:
        # 1. Load and preprocess data
        print("Loading and preprocessing data...")
        data = pd.read_csv(r'C:\Users\rdeva\Downloads\sem5\quantum\Dataset.csv')
        X_normalized, y_normalized, scaler_features, scaler_targets = preprocess_data(data)
        print(f"Data shape: X={X_normalized.shape}, y={y_normalized.shape}")
        
        # 2. Create sequences
        print("Creating sequences...")
        sequence_length = 10
        X_sequences, y_sequences = create_sequences(X_normalized, y_normalized, sequence_length)
        print(f"Sequence shape: X={X_sequences.shape}, y={y_sequences.shape}")
        
        # 3. Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = split_data(X_sequences, y_sequences, train_ratio=0.8)
        print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
        print(f"Test shape: X={X_test.shape}, y={y_test.shape}")
        
        # 4. Initialize QLSTM model
        print("Initializing QLSTM model...")
        model = QLSTM(input_size=6, hidden_size=4, output_size=3)
        print("Model initialized successfully")
        
        # 5. Train the model
        print("Starting training...")
        train_qlstm(model, X_train, y_train, epochs=20, learning_rate=0.01)
        
        # 6. Evaluate the model
        print("Evaluating model...")
        predictions, mse, rmse = evaluate_model(model, X_test, y_test)
        
        print("Training and evaluation completed!")
        print(f"Final Test RMSE: {rmse:.6f}")
        
        return model, predictions, scaler_features, scaler_targets
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        return None, None, None, None

if __name__ == "__main__":
    model, predictions, scaler_features, scaler_targets = main()