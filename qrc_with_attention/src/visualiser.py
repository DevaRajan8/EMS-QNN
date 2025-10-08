
import matplotlib.pyplot as plt
import seaborn as sns

class Visualizer:
    def __init__(self, config):
        self.config = config

    def plot_predictions(self, y_test, predictions, target_labels):
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        for i, (ax, label) in enumerate(zip(axes, target_labels)):
            steps = range(min(self.config['plot_samples'], len(y_test)))
            ax.plot(steps, y_test[steps, i], label='Actual', color='black')
            ax.plot(steps, predictions[steps, i], label='QRC w/ Attention', linestyle='--', color='red')
            ax.set_title(label, fontsize=14); ax.set_xlabel('Time Step'); ax.legend()
        plt.tight_layout(); plt.savefig('qrc_attention_predictions.png'); plt.show()

    def plot_attention_weights(self, attention_weights):
        num_samples = self.config['attention_bar_samples']
        fig, axes = plt.subplots(1, num_samples, figsize=(20, 4))
        for i in range(num_samples):
            ax = axes[i] if num_samples > 1 else axes
            ax.bar(range(len(attention_weights[i])), attention_weights[i])
            ax.set_title(f'Sample {i + 1}'); ax.set_xlabel('Time Step')
        axes[0].set_ylabel('Attention Weight')
        plt.tight_layout(); plt.savefig('attention_weights.png'); plt.show()

    def plot_attention_heatmap(self, attention_weights):
        plt.figure(figsize=(12, 8))
        subset = attention_weights[:self.config['attention_heatmap_samples']]
        sns.heatmap(subset, cmap='viridis', cbar_kws={'label': 'Attention Weight'})
        plt.title('Quantum Attention Patterns'); plt.xlabel('Time Step'); plt.ylabel('Sample Index')
        plt.tight_layout(); plt.savefig('attention_heatmap.png'); plt.show()