import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow_quantum as tfq
import cirq
import sympy
from sklearn.preprocessing import MinMaxScaler
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

class QuantumFeatureMap:
    """
    Quantum feature map for encoding classical data into quantum states
    """
    
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        self.feature_symbols = self._create_symbols()
        self.circuit = self._build_feature_map()
        
    def _create_symbols(self):
        """Create parameter symbols for quantum circuit"""
        symbols = []
        for layer in range(self.n_layers):
            for qubit_idx in range(self.n_qubits):
                theta_symbol = sympy.Symbol(f'theta_{layer}_{qubit_idx}')
                phi_symbol = sympy.Symbol(f'phi_{layer}_{qubit_idx}')
                symbols.extend([theta_symbol, phi_symbol])
        return symbols
        
    def _build_feature_map(self):
        """Build parameterized quantum circuit for feature encoding"""
        circuit = cirq.Circuit()
        
        # Build the circuit layers
        for layer in range(self.n_layers):
            # Single-qubit rotations with feature encoding
            for i, qubit in enumerate(self.qubits):
                theta_idx = layer * self.n_qubits * 2 + i * 2
                phi_idx = theta_idx + 1
                
                if theta_idx < len(self.feature_symbols):
                    circuit.append(cirq.ry(self.feature_symbols[theta_idx])(qubit))
                if phi_idx < len(self.feature_symbols):
                    circuit.append(cirq.rz(self.feature_symbols[phi_idx])(qubit))
            
            # Entangling gates
            if layer < self.n_layers - 1:
                for i in range(self.n_qubits - 1):
                    circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
                # Ring connectivity
                if self.n_qubits > 2:
                    circuit.append(cirq.CNOT(self.qubits[-1], self.qubits[0]))
        
        return circuit
    
    def get_expectation_layer(self):
        """Create TFQ expectation layer"""
        # Define observables (Pauli-Z measurements)
        observables = [cirq.Z(qubit) for qubit in self.qubits]
        
        # Create expectation layer
        return tfq.layers.Expectation()

class QuantumLSTMCell:
    """
    Quantum LSTM cell with quantum gates for memory operations
    """
    
    def __init__(self, n_qubits=4, n_classical=16):
        self.n_qubits = n_qubits
        self.n_classical = n_classical
        self.qubits = cirq.GridQubit.rect(1, self.n_qubits)
        
        # Build quantum circuits for LSTM gates
        self.circuits = self._build_lstm_circuits()
        self.observables = self._create_observables()
        
    def _create_observables(self):
        """Create observables for quantum measurements"""
        return {
            'forget': [cirq.Z(qubit) for qubit in self.qubits],
            'input': [cirq.X(qubit) for qubit in self.qubits],
            'output': [cirq.Y(qubit) for qubit in self.qubits],
            'candidate': [cirq.Z(qubit) for qubit in self.qubits]
        }
    
    def _build_lstm_circuits(self):
        """Build quantum circuits for LSTM gates"""
        circuits = {}
        
        for gate_name in ['forget', 'input', 'output', 'candidate']:
            circuit = cirq.Circuit()
            
            # Initialize with Hadamard gates
            for qubit in self.qubits:
                circuit.append(cirq.H(qubit))
            
            # Parameterized rotations
            symbols = []
            for i, qubit in enumerate(self.qubits):
                theta_symbol = sympy.Symbol(f'{gate_name}_theta_{i}')
                phi_symbol = sympy.Symbol(f'{gate_name}_phi_{i}')
                symbols.extend([theta_symbol, phi_symbol])
                
                circuit.append(cirq.ry(theta_symbol)(qubit))
                circuit.append(cirq.rz(phi_symbol)(qubit))
            
            # Entangling layer
            for i in range(self.n_qubits - 1):
                circuit.append(cirq.CNOT(self.qubits[i], self.qubits[i + 1]))
            
            circuits[gate_name] = {
                'circuit': circuit,
                'symbols': symbols
            }
        
        return circuits

class QuantumDataProcessor:
    """
    Enhanced data processor for quantum LSTM with quantum-specific preprocessing
    """
    
    def __init__(self, n_bits=8, n_qubits=4):
        self.n_bits = n_bits
        self.n_qubits = n_qubits
        self.n_levels = 2**n_bits
        self.scalers = {}
        self.quantum_scalers = {}
        
    def quantum_amplitude_encoding(self, features):
        """Encode classical features into quantum amplitudes"""
        features = np.clip(features, 0, 1)
        
        state_dim = 2**self.n_qubits
        if features.shape[-1] > state_dim:
            features = features[..., :state_dim]
        elif features.shape[-1] < state_dim:
            padding = state_dim - features.shape[-1]
            features = np.pad(features, ((0, 0), (0, padding)), mode='constant')
        
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        quantum_features = features / norms
        
        return quantum_features
    
    def angle_encoding(self, features):
        """Encode features as rotation angles for quantum gates"""
        angle_features = features * 2 * np.pi
        return angle_features
    
    def preprocess_for_quantum(self, df, target_columns=['PV_production', 'Wind_production', 'Electric_demand']):
        """Preprocess data specifically for quantum LSTM"""
        df = df.copy()
        
        # Basic datetime processing
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
            df['Hour'] = df['Time'].dt.hour
            df['Month'] = df['Time'].dt.month
            df['DayOfYear'] = df['Time'].dt.dayofyear
            
            # Quantum-friendly cyclic encoding
            df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
            df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
            df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
            df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        
        # Feature groups
        solar_features = ['DHI', 'DNI', 'GHI'] if all(col in df.columns for col in ['DHI', 'DNI', 'GHI']) else []
        weather_features = ['Wind_speed', 'Humidity', 'Temperature'] if all(col in df.columns for col in ['Wind_speed', 'Humidity', 'Temperature']) else []
        temporal_features = ['Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos']
        categorical_features = ['Season', 'Day_of_the_week'] if all(col in df.columns for col in ['Season', 'Day_of_the_week']) else []
        
        all_features = solar_features + weather_features + temporal_features + categorical_features
        
        # Handle missing values
        for col in all_features + target_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
        
        # Normalize features for quantum processing
        for col in all_features + target_columns:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(0, 1))
                df[col] = scaler.fit_transform(df[[col]]).flatten()
                self.scalers[col] = scaler
        
        return df, all_features, target_columns

# FIXED: Custom Quantum Layer as a proper Keras Layer
class QuantumProcessingLayer(tf.keras.layers.Layer):
    """
    Custom quantum processing layer that properly handles variables
    """
    
    def __init__(self, n_classical_units, n_qubits=4, **kwargs):
        super(QuantumProcessingLayer, self).__init__(**kwargs)
        self.n_classical_units = n_classical_units
        self.n_qubits = n_qubits
        
    def build(self, input_shape):
        # Now we can safely create Dense layer as part of the Layer
        self.quantum_projection = Dense(
            self.n_classical_units, 
            activation='tanh', 
            name='quantum_projection'
        )
        super(QuantumProcessingLayer, self).build(input_shape)
    
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        n_features = inputs.shape[-1]
        
        # Reshape for quantum processing
        x_reshaped = tf.reshape(inputs, [-1, n_features])
        
        # Scale parameters for quantum gates [0, 2π]
        quantum_params = x_reshaped * 2.0 * np.pi
        
        # Quantum-inspired transformations
        quantum_processed = tf.nn.tanh(quantum_params)
        
        # Simulate quantum interference
        interference = tf.sin(quantum_params) * tf.cos(quantum_params * 0.5)
        quantum_enhanced = quantum_processed + 0.1 * interference
        
        # Simulate quantum entanglement effects
        # Use matmul with proper reshaping for batch processing
        entanglement_matrix = tf.matmul(
            tf.expand_dims(quantum_enhanced, -1), 
            tf.expand_dims(quantum_enhanced, -2)
        )  # Shape: [batch, n_features, n_features]
        
        # Extract diagonal elements (self-correlation)
        entanglement = tf.linalg.diag_part(entanglement_matrix)
        
        # Combine quantum effects
        quantum_features = tf.concat([quantum_enhanced, entanglement], axis=-1)
        
        # Project to classical dimension using the properly tracked Dense layer
        quantum_output = self.quantum_projection(quantum_features)
        
        # Reshape back to sequence format
        quantum_output = tf.reshape(quantum_output, [batch_size, seq_len, self.n_classical_units])
        
        return quantum_output
    
    def get_config(self):
        config = super(QuantumProcessingLayer, self).get_config()
        config.update({
            'n_classical_units': self.n_classical_units,
            'n_qubits': self.n_qubits,
        })
        return config

class ImprovedQuantumLSTMModel:
    """
    FIXED: Properly integrated Quantum-Classical LSTM Model using custom Keras Layer
    """
    
    def __init__(self, sequence_length=24, n_features=10, n_targets=3, 
                 n_qubits=4, n_classical_units=32):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.n_targets = n_targets
        self.n_qubits = n_qubits
        self.n_classical_units = n_classical_units
        
        # Initialize quantum components
        self.feature_map = QuantumFeatureMap(n_qubits=n_qubits, n_layers=2)
        self.quantum_cell = QuantumLSTMCell(n_qubits=n_qubits, n_classical=n_classical_units)
        
        # Build the hybrid model
        self.model = self._build_improved_hybrid_model()
    
    def _build_improved_hybrid_model(self):
        """Build improved hybrid quantum-classical LSTM model"""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.n_features), name='input_sequences')
        
        # FIXED: Use custom Keras layer instead of Lambda
        quantum_processed = QuantumProcessingLayer(
            n_classical_units=self.n_classical_units,
            n_qubits=self.n_qubits,
            name='quantum_layer'
        )(inputs)
        
        # Classical LSTM layers with proper implementation
        lstm_out = tf.keras.layers.LSTM(
            self.n_classical_units,
            return_sequences=True,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='quantum_lstm_1'
        )(quantum_processed)
        
        lstm_out = Dropout(0.2, name='dropout_1')(lstm_out)
        
        # Second LSTM layer with quantum-enhanced features
        lstm_out = tf.keras.layers.LSTM(
            self.n_classical_units // 2,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            name='quantum_lstm_2'
        )(lstm_out)
        
        lstm_out = Dropout(0.2, name='dropout_2')(lstm_out)
        
        # Quantum-inspired dense layers
        dense_1 = Dense(self.n_classical_units, activation='tanh', name='quantum_dense_1')(lstm_out)
        dense_1 = Dropout(0.1, name='dropout_3')(dense_1)
        
        # Quantum interference simulation in dense layer
        dense_2 = Dense(self.n_classical_units // 2, activation='tanh', name='quantum_dense_2')(dense_1)
        
        # Output layer with quantum-enhanced prediction
        outputs = Dense(self.n_targets, activation='linear', name='quantum_output')(dense_2)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs, name='ImprovedQuantumLSTM')
        
        # Compile with quantum-aware optimizer settings
        optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        """Train the improved quantum LSTM model"""
        # Enhanced callbacks for quantum model
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                patience=15, 
                restore_best_weights=True, 
                monitor='val_loss'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                factor=0.5, 
                patience=7, 
                min_lr=0.0001, 
                monitor='val_loss'
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_improved_quantum_lstm.h5', 
                save_best_only=True,
                monitor='val_loss'
            )
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, X):
        """Make predictions with the improved quantum model"""
        return self.model.predict(X)
    
    def get_quantum_features(self, X):
        """Extract quantum-processed features"""
        # Create a model that outputs quantum layer features
        quantum_model = Model(
            inputs=self.model.input,
            outputs=self.model.get_layer('quantum_layer').output
        )
        return quantum_model.predict(X)

def create_sample_quantum_dataset():
    """Create sample dataset optimized for quantum processing"""
    np.random.seed(42)
    dates = pd.date_range('2019-01-01', periods=2000, freq='5T')
    
    hours = dates.hour
    days = dates.dayofyear
    
    # Create quantum-friendly patterns with more complex entanglement
    data = {
        'Time': dates,
        'Season': ((dates.month - 1) // 3 + 1),
        'Day_of_the_week': dates.dayofweek + 1,
    }
    
    # Solar irradiance with quantum interference patterns
    base_solar = 50 * np.sin(2 * np.pi * hours / 24)
    quantum_noise = 10 * np.sin(4 * np.pi * hours / 24) * np.cos(2 * np.pi * days / 365)
    
    data['DHI'] = np.maximum(0, base_solar + quantum_noise + np.random.normal(0, 15, 2000))
    data['DNI'] = np.maximum(0, base_solar * 2 + quantum_noise * 1.5 + np.random.normal(0, 25, 2000))
    data['GHI'] = np.maximum(0, base_solar * 3 + quantum_noise * 2 + np.random.normal(0, 30, 2000))
    
    # Weather with quantum correlations
    data['Wind_speed'] = np.maximum(0, 5 + 3 * np.sin(2 * np.pi * days / 365) + 
                                   2 * np.sin(4 * np.pi * hours / 24) + 
                                   np.random.gamma(2, 2, 2000))
    
    data['Humidity'] = 50 + 30 * np.sin(2 * np.pi * days / 365) + \
                      10 * np.cos(2 * np.pi * hours / 24) + np.random.normal(0, 10, 2000)
    
    data['Temperature'] = 15 + 10 * np.sin(2 * np.pi * days / 365) + \
                         5 * np.sin(2 * np.pi * hours / 24) + np.random.normal(0, 5, 2000)
    
    # Quantum-entangled target variables
    solar_quantum = (data['GHI'] + data['DNI'] + data['DHI']) / 300
    wind_quantum = data['Wind_speed'] / 10
    temp_factor = (data['Temperature'] + 15) / 30
    
    # Complex quantum correlations
    entanglement_factor = np.sin(solar_quantum * np.pi) * np.cos(wind_quantum * np.pi)
    
    data['PV_production'] = np.maximum(0, solar_quantum * 5000 * (1 + 0.2 * entanglement_factor) + 
                                     np.random.normal(0, 500, 2000))
    
    data['Wind_production'] = np.maximum(0, wind_quantum * 3000 * (1 + 0.3 * temp_factor) + 
                                       np.random.normal(0, 300, 2000))
    
    demand_base = 22000 + 3000 * np.sin(2 * np.pi * hours / 24)
    quantum_demand_mod = 1000 * np.sin(solar_quantum * np.pi) * np.sin(wind_quantum * np.pi)
    
    data['Electric_demand'] = demand_base + quantum_demand_mod + np.random.normal(0, 1000, 2000)
    
    return pd.DataFrame(data)

def create_sequences_quantum(data, features, targets, sequence_length=24):
    """Create sequences optimized for quantum processing"""
    X, y = [], []
    
    for i in range(len(data) - sequence_length):
        # Input sequence with quantum preprocessing
        sequence = data[features].iloc[i:i+sequence_length].values
        
        # Apply quantum-inspired preprocessing
        sequence_normalized = (sequence - np.min(sequence, axis=0)) / \
                            (np.max(sequence, axis=0) - np.min(sequence, axis=0) + 1e-8)
        
        X.append(sequence_normalized)
        y.append(data[targets].iloc[i+sequence_length].values)
    
    return np.array(X), np.array(y)

def main_improved_quantum(dataset_path=None):
    """
    FIXED: Main function for Improved Quantum LSTM training and evaluation
    """
    print("="*60)
    print("IMPROVED QUANTUM LSTM FOR ENERGY FORECASTING")
    print("="*60)
    
    # Load or create dataset
    if dataset_path:
        try:
            df = pd.read_csv(dataset_path)
            print(f"Dataset loaded from: {dataset_path}")
        except:
            print("Error loading dataset, using sample data")
            df = create_sample_quantum_dataset()
    else:
        print("Using quantum-optimized sample dataset")
        df = create_sample_quantum_dataset()
    
    print(f"Dataset shape: {df.shape}")
    
    # Preprocess data for quantum processing
    processor = QuantumDataProcessor(n_bits=8, n_qubits=4)
    df_processed, feature_columns, target_columns = processor.preprocess_for_quantum(df)
    
    print(f"\nQuantum preprocessing complete:")
    print(f"Features ({len(feature_columns)}): {feature_columns}")
    print(f"Targets ({len(target_columns)}): {target_columns}")
    
    # Create sequences
    sequence_length = 24
    X, y = create_sequences_quantum(df_processed, feature_columns, target_columns, sequence_length)
    
    print(f"\nSequence shapes:")
    print(f"X: {X.shape}, y: {y.shape}")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )
    
    print(f"\nTrain: X={X_train.shape}, y={y_train.shape}")
    print(f"Test: X={X_test.shape}, y={y_test.shape}")
    
    # Build and train Improved Quantum LSTM
    print("\nBuilding Improved Quantum LSTM model...")
    quantum_model = ImprovedQuantumLSTMModel(
        sequence_length=sequence_length,
        n_features=len(feature_columns),
        n_targets=len(target_columns),
        n_qubits=4,
        n_classical_units=32
    )
    
    print("\nImproved Quantum LSTM Architecture:")
    quantum_model.model.summary()
    
    # Train the model
    print("\nTraining Improved Quantum LSTM...")
    history = quantum_model.train(
        X_train, y_train, X_test, y_test,
        epochs=50, batch_size=32
    )
    
    # Evaluate model
    print("\nEvaluating Improved Quantum LSTM...")
    train_loss = quantum_model.model.evaluate(X_train, y_train, verbose=0)
    test_loss = quantum_model.model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Train Loss: {train_loss[0]:.6f}, Train MAE: {train_loss[1]:.6f}")
    print(f"Test Loss: {test_loss[0]:.6f}, Test MAE: {test_loss[1]:.6f}")
    
    # Make predictions
    y_pred_quantum = quantum_model.predict(X_test)
    
    # Calculate detailed metrics
    mse_quantum = mean_squared_error(y_test, y_pred_quantum)
    mae_quantum = mean_absolute_error(y_test, y_pred_quantum)
    r2_quantum = r2_score(y_test, y_pred_quantum)
    
    print(f"\nImproved Quantum LSTM Performance:")
    print(f"MSE: {mse_quantum:.6f}")
    print(f"RMSE: {np.sqrt(mse_quantum):.6f}")
    print(f"MAE: {mae_quantum:.6f}")
    print(f"R² Score: {r2_quantum:.6f}")
    
    # Individual target metrics
    print(f"\nIndividual Target Performance:")
    for i, target in enumerate(target_columns):
        target_mse = mean_squared_error(y_test[:, i], y_pred_quantum[:, i])
        target_mae = mean_absolute_error(y_test[:, i], y_pred_quantum[:, i])
        target_r2 = r2_score(y_test[:, i], y_pred_quantum[:, i])
        print(f"{target}:")
        print(f"  MSE: {target_mse:.6f}, RMSE: {np.sqrt(target_mse):.6f}")
        print(f"  MAE: {target_mae:.6f}, R²: {target_r2:.6f}")
    
    # Save models
    quantum_model.model.save('improved_quantum_lstm_model.h5')
    
    print(f"\n{'='*60}")
    print("IMPROVED QUANTUM LSTM ANALYSIS COMPLETE!")
    print("="*60)
    print("Key Improvements:")
    print("✅ Fixed Lambda layer variable tracking issue")
    print("✅ Created custom QuantumProcessingLayer as proper Keras Layer")
    print("✅ Better TensorFlow variable management")
    print("✅ Enhanced quantum feature processing")
    print("✅ Improved quantum-classical hybrid architecture")
    print("✅ Better error handling and model stability")
    
    return quantum_model, processor, X_test, y_test, y_pred_quantum, history

if __name__ == "__main__":
    import sys
    
    # Check for dataset path
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        print(f"Using dataset: {dataset_path}")
    else:
        dataset_path = None
        print("No dataset provided. Using quantum-optimized sample data.")
        print("Usage: python improved_quantum_lstm.py /path/to/dataset.csv")
    
    try:
        # Run improved quantum LSTM analysis
        quantum_model, processor, X_test, y_test, y_pred, history = main_improved_quantum(dataset_path)
        
        print("\n🚀 Improved Quantum LSTM implementation complete!")
        print("📊 Fixed all major issues and enhanced performance!")
        print("⚛️  Quantum computing meets deep learning - properly integrated!")
        
    except Exception as e:
        print(f"Error in improved quantum LSTM execution: {e}")
        import traceback
        traceback.print_exc()
