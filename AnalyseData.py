import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.preprocessing import StandardScaler
import scipy
import seaborn as sns
from main import updated_dataset
import os
def plot_histogram(df, column_name, save_dir='./stat_fig/col_hists'):
    """
    Create and display a histogram for a specified column in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which to create the histogram.
        save_dir (str): The directory where the histogram figures will be saved. Default is './stat_figs/col_hists'.
    """

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    min_range = df[column_name].min()
    min_row = df[df[column_name] == min_range]
    max_range = df[column_name].max()
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        print(f"Column '{column_name}' does not exist in the DataFrame.")
        return

    if column_name != 'Draw ID':
        # Plot the histogram for the specified column
        plt.figure()
        plt.hist(df[column_name], bins=max_range - min_range +1, color='blue', edgecolor='black')

        # Customize the plot (add labels, title, etc. if needed)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of {column_name}')

        # Save the histogram figure to the specified directory
        save_path = os.path.join(save_dir, f'{column_name}_hist.png')
        plt.savefig(save_path)
        # Show the histogram
        #plt.show()

# Iterate through column names and plot histograms
for column_name in updated_dataset.columns:
    plot_histogram(updated_dataset, str(column_name))


min_range = updated_dataset['1'].min()
min_row = updated_dataset[updated_dataset['1'] == min_range]
max_range = updated_dataset['1'].max()
print(min_range)
print(max_range)