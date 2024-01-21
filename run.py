import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

# pytorch imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import argparse
from parser_code import args
from dataset_split import split_data
from train_loss import train


if os.path.exists(args.exp_dir) == False:
    os.mkdir(args.exp_dir)
print("exp. directory:", args.exp_dir)
# check if there is gpu avilable, if there is, use it
if torch.cuda.is_available():
 torch.cuda.current_device()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data, val_data, test_data = split_data('./data/weak_balls.csv')
sample_dataloader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
print(len(sample_dataloader))
for batch in sample_dataloader:
    # Access the numerical data (assuming you have 6 features per sample)
    numerical_data = batch

print("running calculations on: ", device)
# load the data
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, drop_last=True)
#train
train_losses, train_kl_losses, train_recon_losses = train(train_loader, args)

###### display results #####
from IPython.display import display, Math
fig = plt.figure(figsize=(8, 10))
ax = fig.add_subplot(2, 1, 1)
ax.plot(train_kl_losses[0:args.n_epochs-1], color='r', label='beta = 0.5')
ax.plot(train_kl_losses[args.n_epochs:2*args.n_epochs-1], color='b', label='beta = 1.5')
ax.plot(train_kl_losses[2*args.n_epochs:3*args.n_epochs-1], color='g', label='beta = 2')
ax.grid()
ax.legend()
ax.set_title('KL Divergence')

ax = fig.add_subplot(2, 1, 2)
ax.plot(train_recon_losses[0:args.n_epochs-1], color='r', label='beta = 0.5')
ax.plot(train_recon_losses[args.n_epochs:2*args.n_epochs-1], color='b', label='beta = 1.5')
ax.plot(train_recon_losses[2*args.n_epochs:3*args.n_epochs-1], color='g', label='beta = 2')
ax.grid()
ax.legend()
ax.set_title('Reconstruction Loss')

import csv

# Define the file name for the CSV file
csv_filename = args.exp_dir + "/losses.csv"

# Create a list of dictionaries to store the data of beta 2(most intresting)
data = [
    {"Epoch": epoch, "Total Loss": loss, "KL Loss": kl_loss, "Reconstruction Loss": recon_loss}
    for epoch, loss, kl_loss, recon_loss in zip(range(args.n_epochs), train_losses[2*args.n_epochs:3*args.n_epochs-1], train_kl_losses[2*args.n_epochs:3*args.n_epochs-1]
                                                , train_recon_losses[2*args.n_epochs:3*args.n_epochs-1])
]

# Save the data to a CSV file
with open(csv_filename, "w", newline="") as csv_file:
    fieldnames = ["Epoch", "Total Loss", "KL Loss", "Reconstruction Loss"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)

print(f"Saved the losses to {csv_filename}")

fig.savefig(args.exp_dir + "/losses.png")  # Save the plot as a PNG file

plt.tight_layout()

import pickle

# Save the args to a file
with open('/args.pkl', 'wb') as f:
    pickle.dump(args, f)
