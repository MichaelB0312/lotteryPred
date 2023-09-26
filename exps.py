import pandas as pd
import numpy as np

def concatenate_and_histogram(df, column1_name, column2_name):
    """
    Create and display a histogram based on the concatenated values of two columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column1_name (str): The name of the first column to concatenate.
        column2_name (str): The name of the second column to concatenate.
    """
    # Concatenate the values of the two specified columns
    concatenated_values = df[column1_name].astype(str) + '-' + df[column2_name].astype(str)

    # Create a histogram using unique concatenated values as bins
    unique_values, counts = np.unique(concatenated_values, return_counts=True)

    # Sort the values by count in descending order
    sorted_indices = np.argsort(counts)[::-1]
    unique_values = unique_values[sorted_indices]
    counts = counts[sorted_indices]

    # Print the histogram (values and frequencies)
    print(f"Histogram based on Concatenated Columns: {column1_name} and {column2_name}")
    for i, value in enumerate(unique_values):
        print(f"Bin {i + 1}: Value = {value}, Frequency = {counts[i]}")

# Example usage:
data = {'column5': [1, 2, 2, 3, 3, 3, 4, 4, 4, 4],
        'column6': [5, 6, 6, 7, 7, 7, 8, 8, 8, 8]}
df = pd.DataFrame(data)
concatenate_and_histogram(df, 'column5', 'column6')
