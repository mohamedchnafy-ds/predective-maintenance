# Wind Turbine Predictive Maintenance

This repository contains code for predictive maintenance of wind turbines based on SCADA data, inspired by the paper "CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data".

## Description

This project aims to implement early fault detection algorithms for wind turbines using real-world SCADA data. The goal is to detect potential failures before they occur, reducing downtime and maintenance costs.

## Repository Structure

The repository is organized as follows:

- `src/`: Contains the Python scripts for data loading, preprocessing, and model training
- `notebooks/`: Jupyter notebooks for exploratory data analysis and visualization
- `models/`: Saved trained models
- `results/`: Performance metrics and visualizations

## Prerequisites

Before starting, make sure you have installed:

- Python 3.8 or higher
- Required Python libraries:
  ```
  pip install pandas numpy scikit-learn matplotlib seaborn tensorflow
  ```

## Getting Started

To use this repository:

1. Clone this repository:
   ```
   git clone https://github.com/mohamedchnafy-ds/predective-maintenance.git
   cd predective-maintenance
   ```

2. Download the wind farm dataset and place it in the appropriate directory.

3. Run the data loader to prepare the dataset:
   ```
   python src/data_loader.py
   ```

## The CARE Score

The CARE score is a comprehensive evaluation method that considers four key aspects of a high-quality early fault detection model for predictive maintenance:

- **Coverage**: The ability to detect many anomalies
- **Accuracy**: The ability to correctly recognize normal behavior
- **Reliability**: The ability to minimize false alarms
- **Earliness**: The ability to detect anomalies as early as possible

## Contributing

Contributions to this repository are welcome. If you find errors, potential improvements, or wish to add new models or analyses, please submit a pull request.

## References

- Gück, C.; Roelofs, C.M.A.; Faulstich, S. CARE to Compare: A Real-World Benchmark Dataset for Early Fault Detection in Wind Turbine Data. Data 2024, 9, 138. https://doi.org/10.3390/data9120138
- Dataset: https://doi.org/10.5281/zenodo.14006163
