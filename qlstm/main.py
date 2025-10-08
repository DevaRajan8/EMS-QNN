
import yaml
from src.utils import get_device, plot_predictions
from src.data_handler import DataHandler
from src.model import QLSTM
from src.trainer import Trainer

def main(config_path="config.yaml"):
    # 1. Load Configuration and Set Device
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    device = get_device()

    # 2. Prepare Data
    print("Loading and preprocessing data...")
    data_handler = DataHandler(config['data'], device)
    train_loader = data_handler.prepare_dataloaders()
    
    # 3. Initialize Model
    print("Initializing QLSTM model...")
    qlstm_model = QLSTM(config['model']).to(device)

    # 4. Initialize Trainer and Run Training
    trainer = Trainer(qlstm_model, train_loader, config['training'], device)
    trainer.run_training()

    # 5. Evaluate the Model
    print("\nEvaluating model on test data...")
    y_test_inv, predictions = trainer.evaluate(
        data_handler.X_test, 
        data_handler.y_test, 
        data_handler.scaler_targets
    )

    # 6. Visualize Results
    plot_predictions(y_test_inv, predictions, config['data']['target_cols'])
    print("\nExperiment finished successfully.")

if __name__ == '__main__':
    main()