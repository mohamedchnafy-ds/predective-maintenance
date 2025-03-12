import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
#plt.style.use('seaborn')

# Read the data
data_path = Path('C:/Users/moham/Desktop/GDM5 project/Wind Farm A/combined_data.csv')
df = pd.read_csv(data_path, sep=';')

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

print("\nFirst few rows of the data:")
print(df.head())

print("\nBasic statistics of numerical columns:")
print(df.describe())

# Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Create a directory for saving plots if it doesn't exist
output_dir = Path('../results')
output_dir.mkdir(exist_ok=True)

# Time series plot of a few key variables
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Time Series of Key Variables')

# Assuming there's a timestamp column, convert it to datetime
df['time_stamp'] = pd.to_datetime(df['time_stamp'])

# Plot wind speed
df.plot(x='time_stamp', y='wind_speed_3_avg', ax=axes[0,0])
axes[0,0].set_title('Wind Speed Over Time')
axes[0,0].set_xlabel('Time')
axes[0,0].set_ylabel('Wind Speed (m/s)')

# Plot power output
df.plot(x='time_stamp', y='power_30_avg', ax=axes[0,1])
axes[0,1].set_title('Power Output Over Time')
axes[0,1].set_xlabel('Time')
axes[0,1].set_ylabel('Power (kW)')

# Plot ambient temperature
df.plot(x='time_stamp', y='sensor_0_avg', ax=axes[1,0])
axes[1,0].set_title('Ambient Temperature Over Time')
axes[1,0].set_xlabel('Time')
axes[1,0].set_ylabel('Temperature (°C)')

# Plot wind direction
df.plot(x='time_stamp', y='sensor_1_avg', ax=axes[1,1])
axes[1,1].set_title('Wind Direction Over Time')
axes[1,1].set_xlabel('Time')
axes[1,1].set_ylabel('Direction (degrees)')

plt.tight_layout()
plt.savefig(output_dir / 'time_series_plots.png')

# Correlation matrix of key variables
key_vars = ['wind_speed_3_avg', 'power_30_avg', 'sensor_0_avg', 
           'sensor_1_avg', 'sensor_5_avg', 'sensor_18_avg']

corr_matrix = df[key_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix of Key Variables')
plt.tight_layout()
plt.savefig(output_dir / 'correlation_matrix.png')

# Distribution plots
plt.figure(figsize=(15, 10))

for i, var in enumerate(key_vars, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[var], kde=True)
    plt.title(f'Distribution of {var}')
    plt.xlabel(var)

plt.tight_layout()
plt.savefig(output_dir / 'distributions.png')

# Scatter plot of wind speed vs power output
plt.figure(figsize=(10, 6))
plt.scatter(df['wind_speed_3_avg'], df['power_30_avg'], alpha=0.5)
plt.title('Wind Speed vs Power Output')
plt.xlabel('Wind Speed (m/s)')
plt.ylabel('Power Output (kW)')
plt.savefig(output_dir / 'wind_speed_vs_power.png')

print("\nExploratory data analysis completed. Plots have been saved in the 'results' directory.")