# Wind Turbine Predictive Maintenance System

![Wind Turbine Maintenance](https://via.placeholder.com/800x200.png?text=Wind+Turbine+Predictive+Maintenance)

## Project Overview
This project implements advanced predictive maintenance techniques for wind turbines using SCADA data. By leveraging machine learning and deep learning approaches, we aim to detect potential failures before they occur, reducing downtime and maintenance costs.

## Repository Structure

### 📁 src/
Core Python modules for data processing:
- Data loading and preprocessing utilities
- Feature engineering pipelines
- Data splitting and transformation tools

### 📁 model/
Machine learning implementations:
- Multiple anomaly detection approaches
- Model checkpoints and configurations
- Evaluation notebooks and scripts

### 📁 data_analysis/
Exploratory data analysis tools:
- Statistical profiling scripts
- Visualization utilities
- Correlation analysis

### 📁 Wind Farm A/
Dataset and metadata:
- SCADA time-series data
- Event information with anomaly labels
- Sensor descriptions and specifications

## Model Architecture

### 🧠 Anomaly Detection Approaches

#### Deep Learning Models
- **LSTM Autoencoder**: Captures temporal dependencies in sequential data, reconstructing normal patterns to identify anomalies
- **Variational Autoencoder (VAE)**: Probabilistic approach that learns latent space representations of normal operation

#### Traditional Machine Learning
- **Isolation Forest**: Efficiently isolates anomalies through recursive partitioning
- **Random Forest Feature Selection**: Identifies most important features for anomaly detection

### 📉 Dimensionality Reduction Techniques
- **Principal Component Analysis (PCA)**: Linear dimensionality reduction preserving variance
- **Feature Selection with Random Forest**: Importance-based feature filtering

## Implementation Strategy

### Data Processing Pipeline
```
Raw SCADA Data → Preprocessing → Feature Selection → Model Training → Anomaly Detection → Performance Evaluation
```

## Key Features

### Early Fault Detection
- Detection of anomalies 7-14 days before failure
- Component-specific anomaly identification
- Confidence scoring for maintenance prioritization

### Performance Metrics
- CARE Score (Coverage, Accuracy, Reliability, Earliness)

## Installation and Usage

### Prerequisites
```bash
pip install -r requirements.txt
```

### Data Preparation
```python
# Load and preprocess data
from src.data_loader import WindFarmDataLoader
loader = WindFarmDataLoader(base_path)
data = loader.load_datasets()
```

### Model Training
```python
# Train LSTM Autoencoder
from model.lstm_autoencoder import build_model
model = build_model(input_shape=(sequence_length, num_features))
model.fit(train_generator, epochs=50, validation_data=val_generator)
```

### Anomaly Detection
```python
# Calculate reconstruction error
reconstruction_error = model.predict(test_data)
anomaly_scores = calculate_anomaly_scores(test_data, reconstruction_error)
```

## Future Work
- Integration with real-time SCADA systems
- Multi-component failure prediction
- Transfer learning for cross-turbine application
- Explainable AI integration for maintenance decision support

## References
- CARE Benchmark Dataset methodology
- TensorFlow and PyTorch documentation
- Anomaly detection research papers
```