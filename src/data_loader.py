import pandas as pd
from pathlib import Path

class WindFarmDataLoader:
    """
    A class to handle loading of wind farm SCADA data.
    """
    
    def __init__(self, base_path):
        """
        Initialize the data loader with the base path to data files.
        
        Args:
            base_path (str or Path): Root directory containing wind farm data
        """
        self.base_path = Path(base_path)
    
    def load_datasets(self):
        """
        Load and combine all CSV datasets from the datasets directory.
        
        Returns:
            pandas.DataFrame: Combined dataset from all CSV files
            
        Raises:
            ValueError: If no datasets are found or loaded successfully
        """
        datasets_path = self.base_path / 'Wind Farm A' / 'datasets'
        all_data = []
        
        for file in datasets_path.glob('*.csv'):
            try:
                df = pd.read_csv(file)
                all_data.append(df)
                print(f"Loaded: {file.name}")
            except Exception as e:
                print(f"Error loading {file.name}: {e}")
                
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            return combined_data
        else:
            raise ValueError("No datasets found or could be loaded")

def main():
    """
    Main execution function that loads and combines the datasets.
    """
    base_path = Path(r'c:\Users\moham\Desktop\GDM5 project')
    loader = WindFarmDataLoader(base_path)
    
    try:
        combined_data = loader.load_datasets()
        print(f"\nCombined dataset shape: {combined_data.shape}")
        
        # Save the combined dataset
        output_path = base_path / 'Wind Farm A' / 'combined_data.csv'
        combined_data.to_csv(output_path, index=False)
        print(f"Combined dataset saved to: {output_path}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()