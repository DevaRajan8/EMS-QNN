
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

class Trainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def train(self, X_res, y):
        print("\nTraining readout layer...")
        best_score, best_alpha = -float('inf'), self.config['readout_alphas'][0]
        for alpha in self.config['readout_alphas']:
            model = Ridge(alpha=alpha).fit(X_res, y)
            score = model.score(X_res, y)
            if score > best_score:
                best_score, best_alpha = score, alpha
        
        self.model = Ridge(alpha=best_alpha).fit(X_res, y)
        print(f"Best Ridge alpha: {best_alpha}")

    def predict(self, X_res):
        return self.model.predict(X_res)

    def evaluate(self, y_true, y_pred, target_names):
        results = {}
        for i, name in enumerate(target_names):
            results[name] = {
                'MSE': mean_squared_error(y_true[:, i], y_pred[:, i]),
                'R2': r2_score(y_true[:, i], y_pred[:, i]),
                'MAE': mean_absolute_error(y_true[:, i], y_pred[:, i])
            }
        return results