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
import itertools


#### read data ####
dataset = pd.read_csv('./data/Lotto.csv')

#MAX_BALL_NUM
#
#for some reason num of balls has decreased from 48 to 37
# Find the first row where any column has a value greater than 37
condition = (dataset.drop(columns=['Draw ID']) > 37).any(axis=1)
first_row = dataset.loc[condition].iloc[0]
# Get the value from the first column of that row
first_column_value = first_row['Draw ID']
print(f"The first column value where any column > 37 is: {first_column_value}")
first_column_value = first_row.iloc[0]
# Find the index of the row with the 'first_column_value'
target_index = dataset[dataset.iloc[:, 0] == first_column_value].index[0]
# Create a new DataFrame with rows from the beginning up to (but not including) the target row
updated_dataset = dataset.iloc[:target_index]
# Save the updated_dataset to a new CSV file
updated_dataset.to_csv('./data/Updated_Lotto.csv', index=False)

MAX_BALL_NUM = updated_dataset['Strong Number'].max()
MAX_BALL_NUM
#for strong balls:
MAX_BALL_NUM = pd.read_csv('./data/strong_balls.csv').iloc[:, 0].max()
MAX_BALL_NUM
df = pd.read_csv('./data/strong_balls.csv')
# Find the first row where the second column has a value greater than 7
first_row = df[df.iloc[:, 1] > 7].iloc[0]
print("The first row where the second column has a value greater than 7 is:")
print(first_row[0])
updated_strong_balls = df.iloc[:(first_row[0]-1)]
updated_strong_balls.to_csv('./data/Updated_Sballs.csv', index=False)
#lotto_dataset

##### some histograms of columns #####
min_range = updated_dataset['5'].min()
min_row = updated_dataset[updated_dataset['5'] == min_range]
max_range = updated_dataset['5'].max()
print(min_row )
print("max_range is:", max_range)
# Plot a histogram for the specified column
plt.hist(updated_dataset['5'], bins=max_range - min_range +1, color='blue', edgecolor='black')
# Customize the plot (add labels, title, etc. if needed)
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of column 5')
# Show the histogram
plt.show()

##### show PCA  #####
random_state = 42
# using scikit-learn
X_val = dataset.loc[:, dataset.columns != 'Strong Number'].values
#print(X)
# for strong_num_idx in range
# y0 = dataset['Strong Number'].values == 0
# y1 = dataset['Strong Number'].values == 1
# y2 = dataset['Strong Number'].values == 2
# pca = PCA(n_components=2)
# X_2d = pca.fit_transform(X_val)


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
    custom_bin_edges = range(min(df[column_name]), max(df[column_name]) + 2)  # +2 to include the maximum value
    print(custom_bin_edges)
    # Create a histogram using custom bin edges
    hist, bins = np.histogram(df[column_name], bins=custom_bin_edges)


    # Find the indices of the top 5 bins with the highest frequencies
    top_indices = np.argsort(hist)[-num_of_cols:][::-1]

    # Extract the values and frequencies of the top 5 bins
    top_values = [bins[i] for i in top_indices]
    top_frequencies = [hist[i] for i in top_indices]

    # Print the top 5 bins (values and frequencies)
    print(f"Most frequented values of Column number: {column_name}")
    tot_percent = 0
    for i in range(num_of_cols):

        print(f"Bin {i+1}: Value = {top_values[i]}, Frequency = {top_frequencies[i]},"
              f" Portion: = {(top_frequencies[i]/len(df[column_name])):.2%}")
        tot_percent += top_frequencies[i]/len(df[column_name])
    print(f" Total Portion: = {tot_percent:.2%}")
    return top_values


most_prominent_vals(updated_dataset,'4', num_of_cols=10)
most_prominent_vals(updated_dataset,'4')

########examine labeling of last two columns################
def concatenate_and_histogram(df, column1_name, column2_name, save_directory, num_of_cols=10, all_bins=False):
    """
    Create and display a histogram based on the concatenated values of two columns.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column1_name (str): The name of the first column to concatenate.
        column2_name (str): The name of the second column to concatenate.
        save_directory (str): The directory where the histogram figures will be saved.
    """
    # Concatenate the values of the two specified columns
    concatenated_values = df[column1_name].astype(str) + '-' + df[column2_name].astype(str)

    # Create a histogram using unique concatenated values as bins
    all_unique_values, all_counts = np.unique(concatenated_values, return_counts=True)

    # Sort the values by count in descending order
    sorted_indices = np.argsort(all_counts)[-num_of_cols:][::-1]
    all_sorted_indices = np.argsort(all_counts)[::-1]
    unique_values = all_unique_values[sorted_indices]
    all_unique_values = all_unique_values[all_sorted_indices]
    counts = all_counts[sorted_indices]
    all_counts = all_counts[all_sorted_indices]

    # Plot the histogram
    if all_bins:
        plt.bar(all_unique_values, all_counts)
    else:
        plt.bar(unique_values, counts)
    plt.xlabel("Concatenated Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram based on {column1_name} and {column2_name}")

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Define the save filename including the directory path
    save_filename = os.path.join(save_directory, f"{column1_name}_{column2_name}_histogram.png")

    # Save the figure to the specified directory
    plt.savefig(save_filename)
    # Clear the current figure to prepare for the next plot
    plt.clf()

    # Display the plot
    #plt.show()
    tot_percent = 0
    print(f"Histogram based on Concatenated Columns: {column1_name} and {column2_name}")
    for i, value in enumerate(unique_values):
        print(f"Bin {i + 1}: Value = {value}, Frequency = {counts[i]}",
              f" Portion: = {(counts[i] / len(df[column1_name])):.2%}")
        tot_percent += counts[i] / len(df[column1_name])
        print(f" Total Portion: = {tot_percent:.2%}")


### focus only on "weak" balls
columns_to_drop = ['Draw ID', 'Strong Number']
weak_balls_data = updated_dataset.drop(columns=columns_to_drop)
strong_balls_data = updated_dataset['Strong Number']
# Save the updated_dataset to a new CSV file
weak_balls_data.to_csv('./data/weak_balls.csv', index=True)
strong_balls_data.to_csv('./data/strong_balls.csv', index=True, header=False)

# Specify the directory where the histogram figures will be saved
save_directory = "./stat_fig/pairs_hists_top10"

# Get the column names as a list
column_names = weak_balls_data.columns.tolist()

# Generate all possible pairs of two columns
column_pairs = list(itertools.combinations(column_names, 2))

# Iterate through the pairs and run the function for each pair
for pair in column_pairs:
    column_name1, column_name2 = pair
    concatenate_and_histogram(weak_balls_data, column_name1, column_name2, save_directory)


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

#### perform tSNE ########
def plot_tsne(X, dim=2, perplexity=30.0, scale_data=False):
    t_sne = TSNE(n_components=dim, perplexity=perplexity)
    X_embedded_TSNE = t_sne.fit_transform(X)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.scatter(X_embedded_TSNE[y0,0], X_embedded_TSNE[y0, 1], color='r', marker='*', label='class 0')
    ax.scatter(X_embedded_TSNE[y1,0], X_embedded_TSNE[y1, 1], color='b', marker='x', label='class 1')
    ax.scatter(X_embedded_TSNE[y2,0], X_embedded_TSNE[y2, 1], color='g', marker='o', label='class 2')
    ax.grid()
    ax.legend()
    ax.set_title("2D t-SNE")
    if not os.path.exists('./stat_fig/tSNE/col6_3labels'):
        os.makedirs('./stat_fig/tSNE/col6_3labels')
    save_path = os.path.join('./stat_fig/tSNE/col6_3labels', '2D_tSNE.png')
    plt.savefig(save_path)
    #plt.show()
    return X_embedded_TSNE

X_embedded_TSNE = plot_tsne(X_five, dim=2, perplexity=20.0)#we've already scaled the data


##### KPCA ######
kpca = KernelPCA(n_components=2, kernel='sigmoid')  # We've found cosine is the best from all the kernels
X_KPCA = kpca.fit_transform(X_five)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.scatter(X_KPCA[y0,0], X_KPCA[y0, 1], color='r', marker='*', label='ball {}'.format(labelsSix[0]))
ax.scatter(X_KPCA[y1,0], X_KPCA[y1, 1], color='b', marker='x', label='ball {}'.format(labelsSix[1]))
ax.scatter(X_KPCA[y2,0], X_KPCA[y2, 1], color='g', marker='o', label='ball {}'.format(labelsSix[2]))
ax.grid()
ax.legend()
ax.set_title("2D KsigmoPCA")
if not os.path.exists('./stat_fig/KsigmoPCA/col6_3labels'):
    os.makedirs('./stat_fig/KsigmoPCA/col6_3labels')
save_path = os.path.join('./stat_fig/KsigmoPCA/col6_3labels', '2D_KsigmoPCA.png')
plt.savefig(save_path)