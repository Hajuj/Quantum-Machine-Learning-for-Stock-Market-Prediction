import os
import re

# Define the base directory where your stock directories are located
base_directory = '../results/test/QRNN/'

# Define the patterns to match the filenames
pattern_1 = re.compile(r'1_seed(\d+)_arch(\d+\.\d+)_qubits(\d+)_qlayers(\d+)_lookback(\d+)_batch(\d+)_(.+)\.csv$')
pattern_10 = re.compile(r'10_seed(\d+)_arch(\d+\.\d+)_qubits(\d+)_qlayers(\d+)_lookback(\d+)_batch(\d+)_(.+)\.csv$')


# Function to rename files
def rename_files_for_stock(stock_directory):
    for filename in os.listdir(stock_directory):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Match the pattern to extract relevant parts
            match_1 = pattern_1.match(filename)
            match_10 = pattern_10.match(filename)
            if match_1:
                parts = filename.split('_')
                seed = parts[1]
                arch = parts[2]
                lookback = parts[5]
                new_filename = f"1_{arch}_{lookback}_{seed}.csv"
            elif match_10:
                parts = filename.split('_')
                seed = parts[1]
                arch = parts[2]
                lookback = parts[5]
                new_filename = f"10_{arch}_{lookback}_{seed}.csv"
            else:
                continue  # Skip files that don't match the expected patterns

            # Construct the full paths for old and new filenames
            old_path = os.path.join(stock_directory, filename)
            new_path = os.path.join(stock_directory, new_filename)
            # Rename the file
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")


# Iterate over all stocks
for stock in os.listdir(base_directory):
    stock_directory = os.path.join(base_directory, stock)
    if os.path.isdir(stock_directory):
        rename_files_for_stock(stock_directory)