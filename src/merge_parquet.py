import pandas as pd
import os

class ParquetMerger:
    def __init__(self, input_dir='.', output_dir='.'):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def merge_files(self, part1_name='combined_data_part1.parquet', 
                   part2_name='combined_data_part2.parquet',
                   output_name='combined_data.parquet'):
        try:
            # Construct full paths
            part1_path = os.path.join(self.input_dir, part1_name)
            part2_path = os.path.join(self.input_dir, part2_name)
            output_path = os.path.join(self.output_dir, output_name)

            # Read the parquet files
            df_part1 = pd.read_parquet(part1_path)
            df_part2 = pd.read_parquet(part2_path)

            # Merge the dataframes
            df_combined = pd.concat([df_part1, df_part2], axis=0)
            df_combined = df_combined.reset_index(drop=True)

            # Save the merged file
            df_combined.to_parquet(output_path)

            print("Files merged successfully!")
            print(f"Total number of rows: {len(df_combined)}")
            return True

        except Exception as e:
            print(f"Error during merge: {str(e)}")
            return False

if __name__ == "__main__":
    # Example usage
    merger = ParquetMerger()
    merger.merge_files()