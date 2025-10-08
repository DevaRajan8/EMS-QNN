# src/data_handler.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler_targets = MinMaxScaler()

    def prepare_data(self):
        df = pd.read_csv(self.config['file_path'])
        if self.config['data_fraction'] < 1.0:
            df = df.sample(frac=self.config['data_fraction'], random_state=42).sort_index()

        feature_cols = ['Day_of_the_week', 'DHI', 'DNI', 'GHI', 'Wind_speed', 'Humidity', 'Temperature']
        features = df[feature_cols].values
        targets = df[self.config['target_labels']].values

        features_scaled = MinMaxScaler().fit_transform(features)
        targets_scaled = self.scaler_targets.fit_transform(targets)

        xs, ys = [], []
        for i in range(len(features_scaled) - self.config['seq_length']):
            xs.append(features_scaled[i:i + self.config['seq_length']])
            ys.append(targets_scaled[i + self.config['seq_length']])
        X, y = np.array(xs), np.array(ys)

        return train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)