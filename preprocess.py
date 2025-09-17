import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    scaler_features = MinMaxScaler(feature_range=(0, np.pi))  # For angle encoding
    scaler_targets = MinMaxScaler()

    features = ['DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature']
    targets = ['PV_production', 'Wind_production', 'Electric_demand']

    X_normalized = scaler_features.fit_transform(data[features])
    y_normalized = scaler_targets.fit_transform(data[targets])
    return X_normalized, y_normalized, scaler_features, scaler_targets
# def main():
#     data = pd.read_csv(r'C:\Users\rdeva\Downloads\sem5\quantum\Dataset.csv')
#     X, y = preprocess_data(data)

# if __name__ == "__main__":
#     main()
