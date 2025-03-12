import pandas as pd
import os
import argparse

class ParquetSplitter:
    def __init__(self, input_dir='.', output_dir='.'):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def split_file(self, input_name='combined_data.parquet',
                  part1_name='combined_data_part1.parquet',
                  part2_name='combined_data_part2.parquet'):
        try:
            # Construct full paths
            input_path = os.path.join(self.input_dir, input_name)
            part1_path = os.path.join(self.output_dir, part1_name)
            part2_path = os.path.join(self.output_dir, part2_name)

            # Read the parquet file
            df = pd.read_parquet(input_path)

            # Calculate the middle index
            middle_index = len(df) // 2

            # Split the dataframe
            df_part1 = df.iloc[:middle_index]
            df_part2 = df.iloc[middle_index:]

            # Save the parts
            df_part1.to_parquet(part1_path)
            df_part2.to_parquet(part2_path)

            print("File split successfully!")
            print(f"Part 1 size: {len(df_part1)} rows")
            print(f"Part 2 size: {len(df_part2)} rows")
            return True

        except Exception as e:
            print(f"Error during split: {str(e)}")
            return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split a parquet file into two parts')
    parser.add_argument('--input-dir', default='.', help='Input directory')
    parser.add_argument('--output-dir', default='.', help='Output directory')
    parser.add_argument('--input-name', default='combined_data.parquet', help='Input file name')
    parser.add_argument('--part1-name', default='combined_data_part1.parquet', help='Output part 1 name')
    parser.add_argument('--part2-name', default='combined_data_part2.parquet', help='Output part 2 name')
    
    args = parser.parse_args()
    
    splitter = ParquetSplitter(input_dir=args.input_dir, output_dir=args.output_dir)
    splitter.split_file(
        input_name=args.input_name,
        part1_name=args.part1_name,
        part2_name=args.part2_name
    )