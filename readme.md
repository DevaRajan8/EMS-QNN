# EMS-QNN: Quantum Neural Networks for Energy Management Systems

## Overview

EMS-QNN is a comprehensive collection of quantum-classical hybrid machine learning models designed for energy management and time-series forecasting applications. The project implements multiple approaches including Quantum Reservoir Computing (QRC) variants and Quantum Long Short-Term Memory (QLSTM) networks. Each implementation explores different quantum computing paradigms combined with classical machine learning techniques to leverage quantum computational advantages for predictive modeling.

The repository contains four distinct implementations:
- **Basic QRC**: Fundamental quantum reservoir computing pipeline
- **Deep QRC**: Enhanced QRC with configurable depth and memory mechanisms
- **QRC with Attention**: Advanced QRC incorporating attention mechanisms for improved temporal dependencies
- **QLSTM**: Quantum-enhanced LSTM using PyTorch and variational quantum circuits

## Features

### Quantum Reservoir Computing Implementations

- **Basic QRC Pipeline**
  - Quantum reservoir circuit construction using Qiskit
  - Observable-based state extraction
  - Classical readout layer training
  - Performance evaluation metrics

- **Deep QRC Variant**
  - Multi-layer reservoir architecture
  - Configurable circuit depth
  - Memory decay mechanisms
  - Enhanced feature extraction

- **QRC with Attention Mechanism**
  - Quantum fidelity-based attention computation
  - Temporal dependency modeling
  - Measurement similarity scoring
  - Advanced visualization capabilities

### Quantum LSTM

- **Vector QLSTM Implementation**
  - Custom quantum LSTM cells in PyTorch
  - Variational quantum circuit integration
  - Hybrid quantum-classical training pipeline
  - Configurable quantum encoding strategies

### Utilities and Tools

- Comprehensive data preprocessing pipelines
- Reservoir state extraction and management
- Training and evaluation frameworks
- Visualization tools for quantum states and predictions
- Configuration-driven experiment management
- Performance metrics and evaluation functions

## Tech Stack

### Core Dependencies

- **Python**: 3.10+
- **Qiskit**: Quantum circuit construction and simulation
  - qiskit
  - qiskit-aer
  - qiskit-machine-learning
- **PyTorch**: Deep learning framework for QLSTM
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Classical machine learning utilities

### Visualization and Utilities

- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization
- **tqdm**: Progress bars
- **PyYAML**: Configuration file management

## Folder Structure

```
EMS-QNN/
├── qlstm/
│   ├── config.yaml                 # QLSTM configuration
│   ├── main.py                     # QLSTM entry point
│   └── src/
│       ├── data_handler.py         # Data loading and preprocessing
│       ├── model.py                # QLSTM, QLSTMCell, VQC models
│       ├── trainer.py              # Training pipeline
│       └── utils.py                # Helper functions
│
├── qrc_with_attention/
│   ├── app.py                      # Application entry point
│   ├── config.yaml                 # Configuration file
│   └── src/
│       ├── data_handler.py         # Data management
│       ├── qrc_model.py            # QuantumAttentionModel
│       ├── trainer.py              # Training orchestration
│       └── visualiser.py           # Visualization utilities
│
├── qrc-basic/
│   ├── main.py                     # Basic QRC pipeline
│   ├── preprocess.py               # Data preprocessing
│   └── model/
│       ├── __init__.py
│       └── utils.py                # Reservoir utilities
│
└── qrc-deep/
    ├── main.py                     # Deep QRC pipeline
    ├── preprocess.py               # Data preprocessing
    └── model/
        ├── __init__.py
        └── utils.py                # Enhanced reservoir utilities
```

## Setup

### 1. Environment Setup

Create and activate a virtual environment:

**Linux / macOS:**
```sh
python -m venv .venv
source .venv/bin/activate
```

**Windows:**
```sh
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

Install core scientific computing libraries:
```sh
pip install numpy pandas scikit-learn matplotlib seaborn tqdm pyyaml
```

Install Qiskit and quantum computing libraries:
```sh
pip install qiskit qiskit-aer qiskit-machine-learning
```

Install PyTorch (for QLSTM):
```sh
# For CPU-only:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For CUDA (GPU):
pip install torch
```

### 3. Verify Installation

Ensure Qiskit Aer Estimator is available:
```python
from qiskit_aer.primitives import Estimator
print("Qiskit Aer installed successfully")
```

## Usage

### Basic QRC

Run the basic quantum reservoir computing pipeline:

```sh
cd qrc-basic
python main.py
```

**Key Components:**
- `create_enhanced_reservoir_circuit`: Constructs the quantum reservoir
- `create_observables`: Defines measurement operators
- `get_enhanced_reservoir_states`: Extracts quantum states
- `evaluate_predictions`: Computes performance metrics

### Deep QRC

Execute the deep QRC variant with enhanced architecture:

```sh
cd qrc-deep
python main.py
```

**Features:**
- Multi-layer reservoir architecture
- Configurable depth parameters
- Memory decay mechanisms
- Advanced state extraction

### QRC with Attention

Run the attention-enhanced QRC model:

```sh
cd qrc_with_attention
python app.py
```

**Configuration:**
Edit `config.yaml` to adjust:
- Data paths
- Quantum circuit parameters
- Attention mechanism settings
- Training hyperparameters

**Key Classes:**
- `QuantumAttentionModel`: Main model implementation
- `DataHandler`: Data loading and preprocessing
- `Trainer`: Training orchestration
- `Visualiser`: Results visualization

### QLSTM (PyTorch)

Train the quantum LSTM model:

```sh
cd qlstm
python main.py
```

**Configuration:**
Modify `config.yaml` for:
- Dataset paths
- LSTM architecture parameters
- VQC configuration
- Training settings

**Core Components:**
- `QLSTM`: Main quantum LSTM model
- `QLSTMCell`: Individual QLSTM cell
- `create_vector_vqc`: Variational quantum circuit builder
- `DataHandler`: Data pipeline management
- `Trainer`: Training and validation

### Configuration Files

Each implementation uses YAML configuration files:

**Example (qrc_with_attention/config.yaml):**
```yaml
data:
  path: "path/to/dataset.csv"
  train_split: 0.8
  
model:
  n_qubits: 4
  depth: 2
  
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### Preprocessing

Data preprocessing utilities are available in:
- `qrc-basic/preprocess.py`
- `qrc-deep/preprocess.py`

Ensure your dataset follows the expected CSV format with appropriate time-series columns.

## Future Enhancements

### Testing and Validation
- Implement comprehensive unit tests for all components
- Add integration tests for end-to-end pipelines
- Create test datasets and benchmarks
- Implement continuous integration workflows

### Infrastructure Improvements
- Develop unified CLI interface for all experiments
- Create orchestrator script for batch experiments
- Implement experiment tracking and logging
- Add Docker containerization for reproducibility
- Provide pre-built environments with all dependencies

### Performance Optimization
- Implement batched quantum circuit execution
- Add asynchronous estimator runs for improved throughput
- Optimize memory usage for large-scale experiments
- Implement distributed computing support

### Model Enhancements
- Add model checkpointing and serialization
- Implement early stopping and learning rate scheduling
- Extend to multi-output and multi-variate forecasting
- Explore hybrid quantum-classical architectures
- Add support for real quantum hardware backends

### Documentation and Examples
- Create detailed API documentation
- Add Jupyter notebook tutorials
- Provide example datasets
- Include performance benchmarking results
- Add visualization examples

### Research Extensions
- Implement quantum feature maps
- Explore different entanglement strategies
- Add noise-aware training mechanisms
- Investigate quantum advantage analysis
- Compare with classical baselines

## License

This project is part of academic research. Please refer to the repository for licensing information.

## Contributors

Developed as part of quantum machine learning research for energy management systems.

## Contact

For questions, issues, or contributions, please open an issue in the repository.
