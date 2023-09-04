import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE, LocallyLinearEmbedding, Isomap
from sklearn.preprocessing import StandardScaler
import scipy
import seaborn as sns

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
MAX_BALL_NUM = updated_dataset['Strong Number'].max()
MAX_BALL_NUM

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