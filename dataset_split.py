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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Split data to train-val-test randomely
def split_data(csv_data, toShuffle=True):
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
  if toShuffle:
    print("shuffling data")
    random.shuffle(data)

  # Define the proportions for the splits (e.g., 80% train, 10% validation, 10% test)
  split_ratio = [0.9, 0.05, 0.05]

  # Calculate the number of samples for each split
  total_samples = len(data)
  train_size = int(split_ratio[0] * total_samples)
  val_size = int(split_ratio[1] * total_samples)

  # Split the data into training, validation, and test sets
  train_data = data[:train_size]
  print("len of train_data", len(train_data))
  val_data = data[train_size:train_size + val_size]
  test_data = data[train_size + val_size:]

  # Convert the data to PyTorch tensors after converting strings
  train_data = torch.tensor([[int(value) for value in row] for row in train_data])
  val_data = torch.tensor([[int(value) for value in row] for row in val_data])
  test_data = torch.tensor([[int(value) for value in row] for row in test_data])

  return train_data, val_data, test_data

# train_data, val_data, test_data = split_data('./data/strong_balls.csv', toShuffle=False)
# type(train_data[0])

def batchify(data, bsz):
  """Divides the data into bsz separate sequences, removing extra elements
  that wouldn't cleanly fit.

  Args:
      data: Tensor, shape [N]
      bsz: int, batch size

  Returns:
      Tensor of shape [N // bsz, bsz]
  """

  seq_len = len(data) // bsz
  data = data[:seq_len * bsz]
  #print(type(data))
  # Reshape the list of tensors
  data = data.view(bsz, seq_len).t().contiguous()

  return data.to(device)


def get_batch(source, i, bptt):
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int
        bptt: int
    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i+1:i+seq_len+1]# compelte
    return data, target


