import random
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# pytorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision


# Split data to train-val-test randomely
def split_data(csv_data):
  # Load your data from the CSV file
  data = []  # To store the rows from the CSV file

  # Assuming 'your_dataset.csv' is your CSV file with six columns of numbers
  with open(csv_data, 'r') as csv_file:
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
  print(len(train_data))
  val_data = data[train_size:train_size + val_size]
  test_data = data[train_size + val_size:]

  # Convert the data to PyTorch tensors after converting strings
  train_data = torch.tensor([[int(value) for value in row] for row in train_data])
  val_data = torch.tensor([[int(value) for value in row] for row in val_data])
  test_data = torch.tensor([[int(value) for value in row] for row in test_data])

  return train_data, val_data, test_data

