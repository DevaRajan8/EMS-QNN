import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch

def load_and_preprocess_data(file_path, seq_length=10):
    """Load and preprocess the dataset for training."""
    df = pd.read_csv(file_path)
    feature_cols = ['Day_of_the_week', 'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature']
    target_cols = ['PV_production', 'Wind_production', 'Electric_demand']

    features = df[feature_cols].values
    targets = df[target_cols].values

    scaler_features = MinMaxScaler()
    features_scaled = scaler_features.fit_transform(features)
    scaler_targets = MinMaxScaler()
    targets_scaled = scaler_targets.fit_transform(targets)

    xs, ys = [], []
    for i in range(len(features_scaled) - seq_length):
        xs.append(features_scaled[i:i + seq_length])
        ys.append(targets_scaled[i + seq_length])
    X, y = np.array(xs), np.array(ys)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    return X_train, y_train, X_test, y_test, scaler_targets