
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class DataHandler:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.scaler_targets = MinMaxScaler()

    def prepare_dataloaders(self):
        df = pd.read_csv(self.config['file_path'])
        if self.config['data_fraction'] < 1.0:
            df = df.sample(frac=self.config['data_fraction'], random_state=42).sort_index()

        features = df[self.config['feature_cols']].values
        targets = df[self.config['target_cols']].values

        features_scaled = MinMaxScaler().fit_transform(features)
        targets_scaled = self.scaler_targets.fit_transform(targets)

        xs, ys = [], []
        for i in range(len(features_scaled) - self.config['seq_length']):
            xs.append(features_scaled[i:i + self.config['seq_length']])
            ys.append(targets_scaled[i + self.config['seq_length']])
        X, y = np.array(xs), np.array(ys)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).float().to(self.device)
        self.X_test = torch.from_numpy(X_test).float().to(self.device)
        self.y_test = torch.from_numpy(y_test).float().to(self.device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        return train_loader