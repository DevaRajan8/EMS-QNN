
import torch
import torch.nn as nn
import time

class Trainer:
    def __init__(self, model, train_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.config = config
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    def run_training(self):
        print("\nStarting training...")
        total_start_time = time.time()
        self.model.train()
        
        for epoch in range(self.config['num_epochs']):
            epoch_start_time = time.time()
            for i, (X_batch, y_batch) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch [{epoch+1}/{self.config["num_epochs"]}], Loss: {loss.item():.4f}, Time: {epoch_time:.2f}s')

        total_time = time.time() - total_start_time
        print(f"\nTotal Training Time: {total_time:.2f} seconds")

    def evaluate(self, X_test, y_test, scaler_targets):
        self.model.eval()
        with torch.no_grad():
            predictions_t = self.model(X_test)

        predictions = scaler_targets.inverse_transform(predictions_t.cpu().numpy())
        y_test_inv = scaler_targets.inverse_transform(y_test.cpu().numpy())
        return y_test_inv, predictions