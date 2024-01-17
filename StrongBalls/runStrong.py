import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import optuna

# pytorch imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torchvision
import argparse
import torch.optim as optim
from parser_code import args, MAX_STRONG
import dataset_split
from trainStrong import train, evaluate
from Models import TransformerModel
from utils import parse_trial_parameters


train_data, val_data, test_data = dataset_split.split_data('../data/Updated_Sballs.csv', toShuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#batchify : train,val and test
train_data = dataset_split.batchify(train_data,args.strong_batch)# complete
val_data = dataset_split.batchify(val_data,args.strong_batch)# complete
test_data = dataset_split.batchify(test_data,args.strong_batch)# complete
 #dim 1: number of tokens in a single batch,  #dim2:number of batches,
print(train_data.shape)
#first 20 numbers of one training sample
sample = train_data[0]
# Print the first 20 words of the sample
# for i in range(train_data.shape[1]):
#     word = sample[i]
#     print(word)
#print(train_data)

#Print a sample from data and target
# Set the starting index
i = 0
# Set the BPTT length
bptt = 2
data,target = dataset_split.get_batch(train_data,i,bptt)

#print a sample from target and data:
# for i in range(0,5):
#     data, targets = dataset_split.get_batch(train_data, i, bptt)
#     print("data sample:\t" , data[0][0], data[1][0])
#     print("target will be:\t" , targets[0][0], targets[1][0])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("exp. directory:", args.exp_dir)
if not os.path.exists(args.exp_dir):
    os.makedirs(args.exp_dir)

batch_size = args.strong_batch
epochs = args.strg_epochs
#n_train_examples = batch_size * 30
#n_valid_examples = batch_size * 10
ntoken = MAX_STRONG+1#ntokens, +1 for special token
bptt = args.bptt

# Specify the path to your text file
text_file_path = args.best_exp

# Parse the trial parameters
val_loss, trial_params = parse_trial_parameters(text_file_path)

# Print or use the parsed information
print(f"Validation Loss After Optuna Trial: {val_loss}")
print("Trial Parameters:")
for key, value in trial_params.items():
    print(f"  {key}: {value}")

ntoken = MAX_STRONG + 1  # ntokens, +1 for special token
# Now, instantiate your model using trial parameters and train it

# Instantiate the model with the best hyperparameters
best_model = TransformerModel(
    ntoken, trial_params['ninp'], trial_params['nhead'],
    trial_params['nhid'], trial_params['nlayers'],
    trial_params['dropout'], trial_params['norm_first']
)
#
opt = trial_params['optimizer'](best_model.parameters(), trial_params['lr'])
Learning_Scheduler = torch.optim.lr_scheduler.StepLR(opt, args.step_size, gamma=args.gama)
train_losses = []
val_losses = []
# Train the model with the best hyperparameters
for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    # ... (training and evaluation code)
    train_loss = train(best_model, train_data, opt, bptt)
    val_loss = evaluate(best_model, val_data)

    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | train loss {:5.2f} | valid loss {:5.2f} |'
          .format(epoch, (time.time() - epoch_start_time), train_loss, val_loss))
    print('-' * 89)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    Learning_Scheduler.step()

# Save the trained model
torch.save(best_model.state_dict(), 'best_model.pth')

# Plot the new graph
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('CE')
plt.legend()


# Save the updated plot
plt.savefig('performance.png')  # Adjust the path to save the updated plot