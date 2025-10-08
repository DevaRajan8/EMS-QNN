
import torch
import matplotlib.pyplot as plt

def get_device():
    """Gets the appropriate device (GPU or CPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

def plot_predictions(y_test_inv, predictions, target_labels):

    print("Generating and saving plots...")
    for i, label in enumerate(target_labels):
        plt.figure(figsize=(15, 5))
        plt.plot(y_test_inv[:, i], label=f'Actual {label}')
        plt.plot(predictions[:, i], label=f'Predicted {label}', linestyle='--')
        plt.title(f'Vector QLSTM: {label} Prediction', fontsize=16)
        plt.xlabel('Time Step')
        plt.ylabel(label)
        plt.legend()
        plt.grid(True)
        
        filename = f'qlstm_{label.lower().replace(" ", "_")}.png'
        plt.savefig(filename)
        print(f"Saved plot to {filename}")
        plt.show()