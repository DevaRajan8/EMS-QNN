
import yaml
import time
from src.data_handler import DataHandler
from src.qrc_model import QuantumAttentionModel
from src.trainer import Trainer
from src.visualizer import Visualizer

def main(config_path="config.yaml"):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("--- QUANTUM ATTENTION RESERVOIR COMPUTING ---")
    print(f"Loaded configuration for experiment: {config['model']['attention_type']} attention.")

    # 2. Load and Prepare Data
    data_handler = DataHandler(config['data'])
    X_train, y_train, X_test, y_test = data_handler.prepare_data()

    # 3. Initialize Quantum Model
    qrc_model = QuantumAttentionModel(config['model'])

    # 4. Process Data to Get Reservoir States
    start_time = time.time()
    X_res_train, _ = qrc_model.transform(X_train)
    X_res_test, attention_weights = qrc_model.transform(X_test)
    print(f"Quantum processing time: {time.time() - start_time:.2f}s")

    # 5. Train and Evaluate the Readout Model
    trainer = Trainer(config['training'])
    trainer.train(X_res_train, y_train)
    predictions_scaled = trainer.predict(X_res_test)
    
    predictions = data_handler.scaler_targets.inverse_transform(predictions_scaled)
    y_test_inv = data_handler.scaler_targets.inverse_transform(y_test)
    
    results = trainer.evaluate(y_test_inv, predictions, config['data']['target_labels'])
    print("\n--- Evaluation Results ---")
    for name, metrics in results.items():
        print(f"  {name:20s} - R²: {metrics['R2']:.4f}, MAE: {metrics['MAE']:.4f}")

    # 6. Visualize Results
    print("\nGenerating visualizations...")
    visualizer = Visualizer(config['visualization'])
    visualizer.plot_predictions(y_test_inv, predictions, config['data']['target_labels'])
    visualizer.plot_attention_weights(attention_weights)
    visualizer.plot_attention_heatmap(attention_weights)
    print("\n--- EXPERIMENT COMPLETE ---")

if __name__ == '__main__':
    main()