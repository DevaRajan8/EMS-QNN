from preprocess import preprocess_data
import pandas as pd
import numpy as np

def create_sequences(X, y, sequence_length):
    sequences_X, sequences_y = [], []
    for i in range(len(X) - sequence_length):
        sequences_X.append(X[i:i+sequence_length])
        sequences_y.append(y[i+sequence_length])
    return np.array(sequences_X), np.array(sequences_y)

data = pd.read_csv(r'C:\Users\rdeva\Downloads\sem5\quantum\Dataset.csv')  
X_normalised, y_normalised, scaler_features, scaler_targets = preprocess_data(data)
sequence_length = 10  # for now the sequence length

X_sequences, y_sequences = create_sequences(X_normalised, y_normalised, sequence_length)

def split_data(X, y, train_ratio=0.8):
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(X_sequences, y_sequences)