# Re-import libraries after reset and re-execute the plotting code
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import numpy as np

# Set style for better visualizations
plt.style.use('seaborn-v0_8')  # Using a valid style name
sns.set_theme()  # Using seaborn's default theme

def load_and_visualize_dataset(file_path, title_prefix):
    """
    Load and visualize a dataset with proper error handling
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Convert datetime column
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Time series of values
        plt.subplot(2, 1, 1)
        for meter in df['api_reading_name'].unique():
            meter_data = df[df['api_reading_name'] == meter]
            plt.plot(meter_data['datetime'], meter_data['value'], label=meter, alpha=0.7)
        
        plt.title(f'{title_prefix} - Time Series')
        plt.xlabel('Datetime')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        
        # Plot 2: Box plot of values by meter
        plt.subplot(2, 1, 2)
        sns.boxplot(x='api_reading_name', y='value', data=df)
        plt.title(f'{title_prefix} - Value Distribution by Meter')
        plt.xlabel('Meter ID')
        plt.ylabel('Value')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Print basic statistics
        print(f"\nStatistics for {title_prefix}:")
        print(df.groupby('api_reading_name')['value'].describe())
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")

def main():
    # Get the raw data directory path
    raw_data_dir = Path('raw data')
    
    # List of datasets to process
    datasets = [
        ('wolfheze reading heat meters for 2 weeks.csv', 'Heat Meters'),
        ('wolfheze reading soalr prediction for 16 days.csv', 'Solar Prediction'),
        ('wolfheze reading electrical meters 2 for weeks.csv', 'Electrical Meters'),
        ('wolfheze reading p1 meter for 2 weeks.csv', 'P1 Meter')
    ]
    
    # Process each dataset
    for filename, title in datasets:
        file_path = raw_data_dir / filename
        if file_path.exists():
            print(f"\nProcessing {title} dataset...")
            load_and_visualize_dataset(file_path, title)
        else:
            print(f"\nFile not found: {file_path}")

if __name__ == "__main__":
    main()
