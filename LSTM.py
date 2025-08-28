import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class QuantumReadyDataProcessor:
    """
    Data processor that prepares energy dataset for both classical LSTM
    and quantum LSTM models with proper quantization
    """
    
    def __init__(self, n_bits=8):
        self.n_bits = n_bits  # Number of bits for quantization
        self.n_levels = 2**n_bits  # Number of discrete levels
        self.scalers = {}
        self.quantization_params = {}
        
    def parse_datetime_features(self, df):
        """Extract and encode temporal features"""
        df = df.copy()
        
        # Convert Time to datetime if it's not already
        df['Time'] = pd.to_datetime(df['Time'])
        
        # Extract temporal features
        df['Hour'] = df['Time'].dt.hour
        df['Month'] = df['Time'].dt.month
        df['DayOfYear'] = df['Time'].dt.dayofyear
        
        # Cyclic encoding for temporal features (quantum-friendly)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfYear_sin'] = np.sin(2 * np.pi * df['DayOfYear'] / 365)
        df['DayOfYear_cos'] = np.cos(2 * np.pi * df['DayOfYear'] / 365)
        
        return df
    
    def logarithmic_transform(self, data, columns):
        """Apply logarithmic transformation to handle wide dynamic ranges"""
        data = data.copy()
        for col in columns:
            if col in data.columns:
                # Ensure no negative values before log transformation
                data[col] = np.maximum(data[col], 0)
                # Add small constant to avoid log(0)
                data[col] = np.log1p(data[col])  # log(1+x)
                # Check for any remaining NaN or inf values
                data[col] = np.nan_to_num(data[col], nan=0.0, posinf=10.0, neginf=0.0)
        return data
    
    def normalize_features(self, data, feature_columns, fit=True):
        """Normalize features to [0,1] range for quantum compatibility"""
        data = data.copy()
        
        for col in feature_columns:
            if col in data.columns:
                # Check for NaN values and handle them
                if data[col].isna().any():
                    print(f"Warning: NaN values found in {col}, filling with median")
                    data[col] = data[col].fillna(data[col].median())
                
                if fit:
                    scaler = MinMaxScaler(feature_range=(0, 1))
                    data[col] = scaler.fit_transform(data[[col]]).flatten()
                    self.scalers[col] = scaler
                else:
                    if col in self.scalers:
                        data[col] = self.scalers[col].transform(data[[col]]).flatten()
                
                # Final check for NaN values after scaling
                data[col] = np.nan_to_num(data[col], nan=0.0)
        
        return data
    
    def uniform_quantization(self, data, feature_columns, fit=True):
        """
        Uniform quantization to discrete levels
        Essential for quantum encoding
        """
        data = data.copy()
        
        for col in feature_columns:
            if col in data.columns:
                values = data[col].values
                
                # Handle NaN values
                if np.isnan(values).any():
                    print(f"Warning: NaN values found in {col} before quantization")
                    values = np.nan_to_num(values, nan=0.0)
                    data[col] = values
                
                if fit:
                    # Calculate quantization parameters
                    min_val = np.min(values)
                    max_val = np.max(values)
                    
                    # Avoid division by zero
                    if max_val == min_val:
                        scale = 1.0
                        print(f"Warning: {col} has constant values, setting scale to 1.0")
                    else:
                        scale = (max_val - min_val) / (self.n_levels - 1)
                    
                    self.quantization_params[col] = {
                        'min_val': min_val,
                        'max_val': max_val,
                        'scale': scale
                    }
                
                # Apply quantization
                params = self.quantization_params[col]
                if params['scale'] > 0:
                    quantized = np.round((values - params['min_val']) / params['scale'])
                else:
                    quantized = np.zeros_like(values)
                    
                quantized = np.clip(quantized, 0, self.n_levels - 1)
                
                # Convert back to normalized range [0,1] for neural network
                data[col] = quantized / (self.n_levels - 1)
                
                # Final NaN check
                data[col] = np.nan_to_num(data[col], nan=0.0)
        
        return data
    
    def preprocess_dataset(self, df, target_columns=['PV_production', 'Wind_production', 'Electric_demand']):
        """Complete preprocessing pipeline"""
        
        print("Starting preprocessing...")
        
        # Check for initial NaN values
        print(f"Initial NaN count: {df.isna().sum().sum()}")
        
        # Parse datetime features
        df = self.parse_datetime_features(df)
        
        # Define feature groups
        solar_features = ['DHI', 'DNI', 'GHI']
        weather_features = ['Wind_speed', 'Humidity', 'Temperature']
        temporal_features = ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 
                           'DayOfYear_sin', 'DayOfYear_cos']
        categorical_features = ['Season', 'Day_of_the_week']
        
        # Handle negative values and outliers before log transform
        for col in target_columns:
            if col in df.columns:
                # Remove negative values
                df[col] = np.maximum(df[col], 0)
                # Cap extreme outliers (above 99.9th percentile)
                upper_bound = np.percentile(df[col], 99.9)
                df[col] = np.minimum(df[col], upper_bound)
        
        # Apply logarithmic transform to high-range variables
        high_range_cols = target_columns.copy()
        df = self.logarithmic_transform(df, high_range_cols)
        
        # Normalize all numerical features
        all_numerical_features = (solar_features + weather_features + 
                                temporal_features + target_columns)
        
        df = self.normalize_features(df, all_numerical_features, fit=True)
        
        # Apply uniform quantization (quantum-ready)
        quantizable_features = (solar_features + weather_features + 
                              temporal_features + categorical_features + 
                              target_columns)
        
        df = self.uniform_quantization(df, quantizable_features, fit=True)
        
        # Final NaN check
        print(f"Final NaN count after preprocessing: {df.isna().sum().sum()}")
        if df.isna().sum().sum() > 0:
            print("Warning: NaN values still present, filling with 0")
            df = df.fillna(0)
        
        # Select final features for model
        feature_columns = (solar_features + weather_features + temporal_features + 
                         categorical_features)
        
        print(f"Preprocessing complete. Using {len(feature_columns)} features.")
        print(f"Quantization: {self.n_levels} discrete levels ({self.n_bits} bits)")
        
        return df, feature_columns, target_columns

def create_sequences(data, features, targets, sequence_length=24):
    """Create sequences for LSTM training"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Input sequence
        X.append(data[features].iloc[i:i+sequence_length].values)
        # Target (next time step)
        y.append(data[targets].iloc[i+sequence_length].values)
    
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_dim, lstm_units=[64, 32], dropout_rate=0.2):
    """
    Build LSTM model architecture compatible with quantum conversion
    """
    model = Sequential()
    
    # First LSTM layer with gradient clipping
    model.add(LSTM(lstm_units[0], 
                   return_sequences=True if len(lstm_units) > 1 else False,
                   input_shape=input_shape,
                   activation='tanh',
                   recurrent_activation='sigmoid',
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal'))
    model.add(Dropout(dropout_rate))
    
    # Additional LSTM layers
    for i, units in enumerate(lstm_units[1:]):
        return_seq = i < len(lstm_units) - 2
        model.add(LSTM(units, 
                       return_sequences=return_seq, 
                       activation='tanh',
                       recurrent_activation='sigmoid',
                       kernel_initializer='glorot_uniform',
                       recurrent_initializer='orthogonal'))
        model.add(Dropout(dropout_rate))
    
    # Output layer
    model.add(Dense(output_dim, activation='linear', kernel_initializer='glorot_uniform'))
    
    # Use a more stable optimizer configuration
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(optimizer=optimizer, 
                  loss='mse', 
                  metrics=['mae'])
    
    return model

# Load and preprocess data
def load_dataset(file_path):
    """
    Load dataset from CSV file
    Expected columns: Time, Season, Day_of_the_week, DHI, DNI, GHI, 
                     Wind_speed, Humidity, Temperature, PV_production, 
                     Wind_production, Electric_demand
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully from: {file_path}")
        print(f"Dataset shape: {df.shape}")
        
        # Display column names
        print(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['Time', 'Season', 'Day_of_the_week', 'DHI', 'DNI', 'GHI',
                          'Wind_speed', 'Humidity', 'Temperature', 'PV_production',
                          'Wind_production', 'Electric_demand']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: Missing columns: {missing_columns}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        print("Creating sample dataset for demonstration...")
        return create_sample_dataset()

def create_sample_dataset():
    """Create sample dataset for demonstration"""
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', periods=2000, freq='5T')
    
    # Create more realistic patterns
    hours = dates.hour
    days = dates.dayofyear
    
    data = {
        'Time': dates,
        'Season': ((dates.month - 1) // 3 + 1),
        'Day_of_the_week': dates.dayofweek + 1,
        'DHI': np.maximum(0, 50 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 20, 2000)),
        'DNI': np.maximum(0, 100 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 30, 2000)),
        'GHI': np.maximum(0, 150 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 40, 2000)),
        'Wind_speed': np.maximum(0, 5 + 3 * np.sin(2 * np.pi * days / 365) + np.random.gamma(2, 2, 2000)),
        'Humidity': 50 + 30 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 10, 2000),
        'Temperature': 15 + 10 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 5, 2000),
    }
    
    # Create correlated target variables
    solar_factor = (data['GHI'] + data['DNI'] + data['DHI']) / 300
    wind_factor = data['Wind_speed'] / 10
    demand_base = 22000 + 3000 * np.sin(2 * np.pi * hours / 24)
    
    data['PV_production'] = np.maximum(0, solar_factor * 5000 + np.random.normal(0, 500, 2000))
    data['Wind_production'] = np.maximum(0, wind_factor * 3000 + np.random.normal(0, 300, 2000))
    data['Electric_demand'] = demand_base + np.random.normal(0, 1000, 2000)
    
    return pd.DataFrame(data)

def plot_comprehensive_results(model, processor, X_test, y_test, y_pred, history, target_columns, feature_columns):
    """
    Create comprehensive plots for model evaluation
    """
    # Set up the plotting style
    plt.rcParams['figure.figsize'] = (15, 10)
    plt.rcParams['font.size'] = 10
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    
    # 1. Training History
    ax1 = plt.subplot(4, 3, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    ax2 = plt.subplot(4, 3, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Training and Validation MAE', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Individual Target Predictions (Predicted vs True)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (target, color) in enumerate(zip(target_columns, colors)):
        ax = plt.subplot(4, 3, 3 + i)
        
        # Scatter plot
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, color=color, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(y_test[:, i]), np.min(y_pred[:, i]))
        max_val = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, alpha=0.8)
        
        # Calculate R²
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        
        plt.title(f'{target}\nR² = {r2:.3f}, MSE = {mse:.6f}', fontsize=12, fontweight='bold')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.grid(True, alpha=0.3)
        
        # Add text box with metrics
        textstr = f'MAE: {mae:.6f}\nRMSE: {np.sqrt(mse):.6f}'
        props = dict(boxstyle='round', facecolor=color, alpha=0.3)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=props)
    
    # 3. Time Series Plots (first 200 samples for clarity)
    n_samples = min(200, len(y_test))
    time_indices = range(n_samples)
    
    for i, (target, color) in enumerate(zip(target_columns, colors)):
        ax = plt.subplot(4, 3, 6 + i)
        
        plt.plot(time_indices, y_test[:n_samples, i], label='True', 
                linewidth=2, color=color, alpha=0.8)
        plt.plot(time_indices, y_pred[:n_samples, i], label='Predicted', 
                linewidth=2, color=color, alpha=0.6, linestyle='--')
        
        plt.title(f'{target} - Time Series Comparison', fontsize=12, fontweight='bold')
        plt.xlabel('Time Step')
        plt.ylabel('Normalized Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 4. Residual Analysis
    ax = plt.subplot(4, 3, 9)
    residuals_all = (y_test - y_pred).flatten()
    plt.hist(residuals_all, bins=50, alpha=0.7, color='skyblue', density=True)
    plt.axvline(np.mean(residuals_all), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(residuals_all):.6f}')
    plt.title('Residuals Distribution (All Targets)', fontsize=12, fontweight='bold')
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Error by Target Variable
    ax = plt.subplot(4, 3, 10)
    mae_by_target = [mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(target_columns))]
    bars = plt.bar(target_columns, mae_by_target, color=colors, alpha=0.7)
    plt.title('Mean Absolute Error by Target Variable', fontsize=12, fontweight='bold')
    plt.ylabel('MAE')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mae in zip(bars, mae_by_target):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{mae:.6f}', ha='center', va='bottom', fontsize=9)
    
    # 6. Feature Importance Approximation (based on model weights)
    ax = plt.subplot(4, 3, 11)
    try:
        # Get first layer weights as proxy for feature importance
        first_layer_weights = model.layers[0].get_weights()[0]  # Input weights
        feature_importance = np.mean(np.abs(first_layer_weights), axis=1)
        
        # Sort features by importance
        sorted_indices = np.argsort(feature_importance)[-10:]  # Top 10
        top_features = [feature_columns[i] for i in sorted_indices]
        top_importance = feature_importance[sorted_indices]
        
        plt.barh(top_features, top_importance, color='lightcoral', alpha=0.7)
        plt.title('Top 10 Feature Importance (Approx.)', fontsize=12, fontweight='bold')
        plt.xlabel('Average Absolute Weight')
        plt.grid(True, alpha=0.3, axis='x')
    except:
        plt.text(0.5, 0.5, 'Feature importance\nanalysis unavailable', 
                ha='center', va='center', transform=ax.transAxes)
        plt.title('Feature Importance', fontsize=12, fontweight='bold')
    
    # 7. Model Summary Statistics
    ax = plt.subplot(4, 3, 12)
    ax.axis('off')
    
    # Calculate overall metrics
    overall_r2 = r2_score(y_test, y_pred)
    overall_mse = mean_squared_error(y_test, y_pred)
    overall_mae = mean_absolute_error(y_test, y_pred)
    
    summary_text = f"""MODEL PERFORMANCE SUMMARY
    
Overall Metrics:
• R² Score: {overall_r2:.6f}
• MSE: {overall_mse:.6f}
• RMSE: {np.sqrt(overall_mse):.6f}
• MAE: {overall_mae:.6f}

Individual Target R² Scores:"""
    
    for i, target in enumerate(target_columns):
        r2_individual = r2_score(y_test[:, i], y_pred[:, i])
        summary_text += f"\n• {target}: {r2_individual:.6f}"
    
    summary_text += f"""

Model Configuration:
• Sequence Length: 24 time steps
• Features: {len(feature_columns)}
• Quantization: {processor.n_bits}-bit ({processor.n_levels} levels)
• Training Samples: {len(y_test) * 4}  # Approx 80/20 split
• Test Samples: {len(y_test)}"""
    
    plt.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('lstm_energy_forecasting_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a separate detailed plot for each target variable
    create_detailed_target_plots(y_test, y_pred, target_columns)

def create_detailed_target_plots(y_test, y_pred, target_columns):
    """Create detailed individual plots for each target variable"""
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (target, color) in enumerate(zip(target_columns, colors)):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Detailed Analysis: {target}', fontsize=16, fontweight='bold')
        
        # 1. Predicted vs True scatter with density
        ax1.scatter(y_test[:, i], y_pred[:, i], alpha=0.6, color=color, s=30)
        
        # Perfect prediction line
        min_val = min(np.min(y_test[:, i]), np.min(y_pred[:, i]))
        max_val = max(np.max(y_test[:, i]), np.max(y_pred[:, i]))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        
        # Calculate and display metrics
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        
        ax1.set_title(f'Predicted vs True Values\nR² = {r2:.6f}', fontweight='bold')
        ax1.set_xlabel('True Values')
        ax1.set_ylabel('Predicted Values')
        ax1.grid(True, alpha=0.3)
        
        # 2. Time series comparison (full length)
        n_samples = min(500, len(y_test))  # Show more samples
        time_indices = range(n_samples)
        
        ax2.plot(time_indices, y_test[:n_samples, i], label='True', 
                linewidth=1.5, color=color, alpha=0.8)
        ax2.plot(time_indices, y_pred[:n_samples, i], label='Predicted', 
                linewidth=1.5, color=color, alpha=0.7, linestyle='--')
        ax2.fill_between(time_indices, y_test[:n_samples, i], y_pred[:n_samples, i], 
                        alpha=0.2, color='red', label='Error')
        
        ax2.set_title('Time Series Comparison', fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Normalized Value')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Residuals analysis
        residuals = y_test[:, i] - y_pred[:, i]
        ax3.scatter(y_pred[:, i], residuals, alpha=0.6, color=color, s=20)
        ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
        ax3.axhline(y=np.mean(residuals), color='green', linestyle=':', linewidth=2, 
                   label=f'Mean: {np.mean(residuals):.6f}')
        
        ax3.set_title('Residuals vs Predicted', fontweight='bold')
        ax3.set_xlabel('Predicted Values')
        ax3.set_ylabel('Residuals')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals distribution
        ax4.hist(residuals, bins=30, alpha=0.7, color=color, density=True, edgecolor='black')
        ax4.axvline(np.mean(residuals), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(residuals):.6f}')
        ax4.axvline(np.median(residuals), color='green', linestyle=':', linewidth=2,
                   label=f'Median: {np.median(residuals):.6f}')
        
        # Add normal distribution overlay
        from scipy import stats
        x = np.linspace(residuals.min(), residuals.max(), 100)
        normal_dist = stats.norm.pdf(x, np.mean(residuals), np.std(residuals))
        ax4.plot(x, normal_dist, 'k-', linewidth=2, alpha=0.7, label='Normal Fit')
        
        ax4.set_title(f'Residuals Distribution\nStd: {np.std(residuals):.6f}', fontweight='bold')
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{target.lower()}_detailed_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main(dataset_path=None):
    """
    Main function to train LSTM model
    Args:
        dataset_path (str): Path to the CSV dataset file
    """
    # Load dataset
    if dataset_path:
        df = load_dataset(dataset_path)
    else:
        print("No dataset path provided. Using sample dataset.")
        df = create_sample_dataset()
    
    print("Dataset shape:", df.shape)
    print("\nDataset info:")
    print(df.describe())
    
    # Initialize processor
    processor = QuantumReadyDataProcessor(n_bits=8)  # 8-bit quantization
    
    # Preprocess data
    df_processed, feature_columns, target_columns = processor.preprocess_dataset(df)
    
    print(f"\nFeature columns ({len(feature_columns)}):", feature_columns)
    print(f"Target columns ({len(target_columns)}):", target_columns)
    
    # Create sequences
    sequence_length = 24  # 2 hours of 5-minute intervals
    X, y = create_sequences(df_processed, feature_columns, target_columns, sequence_length)
    
    print(f"\nSequence data shapes:")
    print(f"X (input): {X.shape}")
    print(f"y (target): {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"\nTrain set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    
    # Build and train model
    model = build_lstm_model(
        input_shape=(sequence_length, len(feature_columns)),
        output_dim=len(target_columns),
        lstm_units=[64, 32],
        dropout_rate=0.2
    )
    
    print("\nModel architecture:")
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    train_loss = model.evaluate(X_train, y_train, verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\nModel Performance:")
    print(f"Train Loss: {train_loss[0]:.6f}, Train MAE: {train_loss[1]:.6f}")
    print(f"Test Loss: {test_loss[0]:.6f}, Test MAE: {test_loss[1]:.6f}")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Check for NaN values in predictions
    if np.isnan(y_pred).any():
        print("Warning: NaN values in predictions, replacing with zeros")
        y_pred = np.nan_to_num(y_pred, nan=0.0)
    
    # Calculate additional metrics
    try:
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        mse = float('inf')
        mae = float('inf')
        r2 = 0.0
    
    print(f"\nDetailed Metrics:")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {np.sqrt(mse):.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    # Individual target metrics
    print(f"\nIndividual Target Metrics:")
    for i, target in enumerate(target_columns):
        target_mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        target_mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        target_r2 = r2_score(y_test[:, i], y_pred[:, i])
        print(f"{target}:")
        print(f"  MSE: {target_mse:.6f}, RMSE: {np.sqrt(target_mse):.6f}")
        print(f"  MAE: {target_mae:.6f}, R²: {target_r2:.6f}")
    
    # Create comprehensive plots
    print("\nGenerating comprehensive plots...")
    plot_comprehensive_results(model, processor, X_test, y_test, y_pred, history, 
                             target_columns, feature_columns)
    
    # Display quantization info for quantum model preparation
    print(f"\n{'='*50}")
    print("QUANTUM MODEL PREPARATION SUMMARY")
    print(f"{'='*50}")
    print(f"Quantization levels: {processor.n_levels} ({processor.n_bits}-bit)")
    print(f"Feature vector size: {len(feature_columns)}")
    print(f"Sequence length: {sequence_length}")
    print(f"All values normalized to [0,1] range")
    print(f"Discrete levels suitable for quantum encoding")
    
    print(f"\nQuantization parameters saved for QLSTM:")
    for col, params in list(processor.quantization_params.items())[:3]:
        print(f"  {col}: min={params['min_val']:.3f}, max={params['max_val']:.3f}")
    
    # Save model and processor for later use
    model.save('lstm_energy_model.h5')
    print("\nModel saved as 'lstm_energy_model.h5'")
    print("Plots saved as PNG files")
    
    return model, processor, X_test, y_test, y_pred, history

def create_correlation_analysis(df_processed, feature_columns, target_columns):
    """Create correlation analysis plots"""
    plt.figure(figsize=(15, 12))
    
    # Select relevant columns for correlation
    corr_columns = feature_columns + target_columns
    correlation_matrix = df_processed[corr_columns].corr()
    
    # Create heatmap
    plt.subplot(2, 2, 1)
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Target correlations
    plt.subplot(2, 2, 2)
    target_corr = correlation_matrix[target_columns].loc[feature_columns]
    sns.heatmap(target_corr, annot=True, cmap='RdYlBu_r', center=0,
                cbar_kws={'shrink': .8})
    plt.title('Features vs Targets Correlation', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Distribution plots for targets
    plt.subplot(2, 2, 3)
    for i, target in enumerate(target_columns):
        plt.hist(df_processed[target], bins=30, alpha=0.7, label=target, density=True)
    plt.title('Target Variables Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Values')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Feature distributions
    plt.subplot(2, 2, 4)
    selected_features = feature_columns[:6]  # Show first 6 features
    for feature in selected_features:
        plt.hist(df_processed[feature], bins=20, alpha=0.6, label=feature, density=True)
    plt.title('Selected Features Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Normalized Values')
    plt.ylabel('Density')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('correlation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_model_architecture_plot(model):
    """Create a visualization of the model architecture"""
    try:
        from tensorflow.keras.utils import plot_model
        plot_model(model, to_file='model_architecture.png', 
                  show_shapes=True, show_layer_names=True, rankdir='TB')
        print("Model architecture plot saved as 'model_architecture.png'")
    except ImportError:
        print("pydot not available for model architecture plotting")
    except Exception as e:
        print(f"Could not create architecture plot: {e}")

if __name__ == "__main__":
    import sys
    
    # Check if dataset path is provided as command line argument
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Using dataset: {dataset_path}")
    else:
        dataset_path = None
        print("No dataset path provided. You can provide it as: python script.py /path/to/dataset.csv")
    
    # Run the main function
    try:
        model, processor, X_test, y_test, y_pred, history = main(dataset_path)
        
        # Additional analysis
        print("\nCreating additional analysis plots...")
        
        # Load processed data for correlation analysis
        if dataset_path:
            df = load_dataset(dataset_path)
        else:
            df = create_sample_dataset()
        
        df_processed, feature_columns, target_columns = processor.preprocess_dataset(df)
        
        # Create correlation analysis
        create_correlation_analysis(df_processed, feature_columns, target_columns)
        
        # Create model architecture plot
        create_model_architecture_plot(model)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("Generated files:")
        print("• lstm_energy_forecasting_results.png - Comprehensive results")
        print("• electric_demand_detailed_analysis.png - Electric demand details")
        print("• pv_production_detailed_analysis.png - PV production details") 
        print("• wind_production_detailed_analysis.png - Wind production details")
        print("• correlation_analysis.png - Feature correlation analysis")
        print("• model_architecture.png - Model architecture diagram")
        print("• lstm_energy_model.h5 - Trained model")
        print("\nAll plots show predicted vs true values with comprehensive metrics!")
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
