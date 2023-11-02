import random
import csv

# Load your data from the CSV file
data = []  # To store the rows from the CSV file

# Assuming 'your_dataset.csv' is your CSV file with six columns of numbers
with open('./data/weak_balls.csv', 'r') as csv_file:
    reader = csv.reader(csv_file)
    for row in reader:
        data.append(row[1:])

# Set a random seed for reproducibility
random.seed(42)

# Shuffle your data
random.shuffle(data)

# Define the proportions for the splits (e.g., 80% train, 10% validation, 10% test)
split_ratio = [0.85, 0.1, 0.05]

# Calculate the number of samples for each split
total_samples = len(data)
train_size = int(split_ratio[0] * total_samples)
val_size = int(split_ratio[1] * total_samples)

# Split the data into training, validation, and test sets
train_data = data[:train_size]
val_data = data[train_size:train_size + val_size]
test_data = data[train_size + val_size:]

import torch

# Convert the data to PyTorch tensors after converting strings to float
train_tensors = torch.tensor([[int(value) for value in row] for row in train_data])

# data split into train_data, val_data, and test_data while maintaining the structure of each row.
print((train_tensors))
print("#########", train_data)