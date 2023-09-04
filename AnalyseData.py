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
        custom_bin_edges = range(min(df[column_name]),
                                 max(updated_dataset[column_name]) + 2)  # +2 to include the maximum value
        plt.hist(df[column_name], bins=custom_bin_edges, color='blue', edgecolor='black')

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


min_range = updated_dataset['5'].min()
min_row = updated_dataset[updated_dataset['1'] == min_range]
max_range = updated_dataset['5'].max()

def most_prominent_vals(df,column_name,num_of_cols = 3):
    """
    Create and display a number of most frequented bins.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column for which to create the histogram.
        num_of_cols : amount of col. members in most frequented values list
    """
    # Specify custom bin edges to ensure integer values in the bins
    custom_bin_edges = range(min(df[column_name]), max(df['5']) + 2)  # +2 to include the maximum value
    # Create a histogram using custom bin edges
    hist, bins = np.histogram(df[column_name], bins=custom_bin_edges)


    # Find the indices of the top 5 bins with the highest frequencies
    top_indices = np.argsort(hist)[-num_of_cols:][::-1]

    # Extract the values and frequencies of the top 5 bins
    top_values = [bins[i] for i in top_indices]
    top_frequencies = [hist[i] for i in top_indices]

    # Print the top 5 bins (values and frequencies)
    print(f"Most frequented values of Column number: {column_name}")
    for i in range(num_of_cols):

        print(f"Bin {i+1}: Value = {top_values[i]}, Frequency = {top_frequencies[i]}")
    return top_values


most_prominent_vals(updated_dataset,'5')
most_prominent_vals(updated_dataset,'4')

### focus only on "weak" balls
columns_to_drop = ['Draw ID', 'Strong Number']
weak_balls_data = updated_dataset.drop(columns=columns_to_drop)


#### perform PCA ###

random_state = 42
# define 5 first cols as data, last column third most frequented values are the labels
labelsSix = most_prominent_vals(updated_dataset,'6')
X_five = weak_balls_data.loc[:, weak_balls_data.columns != '6'].values

#print(X)
y0 = weak_balls_data['6'].values == labelsSix[0]
y1 = weak_balls_data['6'].values == labelsSix[1]
y2 = weak_balls_data['6'].values == labelsSix[2]
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_five)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(X_2d[y0,0], X_2d[y0, 1], color='r', marker='*', label='ball {}'.format(labelsSix[0]))
ax.scatter(X_2d[y1,0], X_2d[y1, 1], color='b', marker='x', label='ball {}'.format(labelsSix[1]))
ax.scatter(X_2d[y2,0], X_2d[y2, 1], color='g', marker='o', label='ball {}'.format(labelsSix[2]))
ax.grid()
ax.legend()
ax.set_title("2D PCA of five columns")
if not os.path.exists('./stat_fig/PCA/col6_3labels'):
    os.makedirs('./stat_fig/PCA/col6_3labels')
save_path = os.path.join('./stat_fig/PCA/col6_3labels', '2D_PCA.png')
plt.savefig(save_path)
#plt.show()
