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
from utils import save_best_trial

train_data, val_data, test_data = dataset_split.split_data('../data/Updated_Sballs.csv', toShuffle=False)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# batchify : train,val and test
train_data = dataset_split.batchify(train_data, args.strong_batch)  # complete
val_data = dataset_split.batchify(val_data, args.strong_batch)  # complete
test_data = dataset_split.batchify(test_data, args.strong_batch)  # complete
# dim 1: number of tokens in a single batch,  #dim2:number of batches,
print(train_data.shape)
# first 20 numbers of one training sample
sample = train_data[0]
# Print the first 20 words of the sample
# for i in range(train_data.shape[1]):
#     word = sample[i]
#     print(word)
# print(train_data)

# Print a sample from data and target
# Set the starting index
i = 0
# Set the BPTT length
bptt = 2
data, target = dataset_split.get_batch(train_data, i, bptt)

# print a sample from target and data:
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
# n_train_examples = batch_size * 30
# n_valid_examples = batch_size * 10
ntoken = MAX_STRONG + 1  # ntokens, +1 for special token
bptt = args.bptt
best_model = None
optimizers = {
    "SGD": optim.SGD,
    "Adam": optim.Adam,
    "RAdam": optim.RAdam,
}

def objective(trial):
    # Define search space for hyperparameters
    nhead = trial.suggest_int('nhead', 2, 4)
    ninp = trial.suggest_int('ninp', (200 // (2 * nhead)) * (2 * nhead), (250 // (2 * nhead)) * (2 * nhead),
                             step=2 * nhead)  # Embedding dimension
    nhid = trial.suggest_int('nhid', 200, 500)  # nhid--the MLP's dimension
    nlayers = trial.suggest_int('nlayers', 2, 4)  # nlayers-->how many encoders
    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    norm_first = trial.suggest_categorical('norm_first', [True, False])
    best_val_loss = float("inf")
    best_model_state_dict = None

    # Optimizer-related hyperparameters
    lr = trial.suggest_float('lr', 5e-3, 5.0, log=True)

    # Create the model using the suggested hyperparameters
    model = TransformerModel(ntoken, ninp, nhead, nhid, nlayers, dropout, norm_first)

    # Choose an optimizer based on a categorical choice
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RAdam'])
    opt = optimizers[optimizer_name](model.parameters(), lr)
    Learning_Scheduler = torch.optim.lr_scheduler.StepLR(opt, args.step_size, gamma=args.gama)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        # ... (training and evaluation code)
        # Lists to store training and validation losses for each epoch
        val_losses = []

        epoch_start_time = time.time()
        # ... (training and evaluation code)
        train_loss = train(model, train_data, opt, bptt)
        val_loss = evaluate(model, val_data)

        # Append training and validation losses for plotting
        val_losses.append(val_loss)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              .format(epoch, (time.time() - epoch_start_time),
                      val_loss))
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()

        Learning_Scheduler.step()

        # report back to Optuna how far it is (epoch-wise) into the trial and how well it is doing
        trial.report(best_val_loss, epoch)

        # then, Optuna can decide if the trial should be pruned
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the metric to be optimized (e.g., validation loss) and training loss(wouldn't be optimized)
    return val_loss


# now we can run the experiment
sampler = optuna.samplers.TPESampler()
study = optuna.create_study(study_name="Strong Ball", direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print("  Number of finished trials: ", len(study.trials))
print("  Number of pruned trials: ", len(pruned_trials))
print("  Number of complete trials: ", len(complete_trials))

print("Best trial:")
trial = study.best_trial
best_val_loss = trial.value

study_data = trial.intermediate_values
val_value = list(study_data.values())
# Plotting the losses
plt.plot(val_value, label='Best Study Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('CE Validation')
plt.legend()

# Save the plot
plt.savefig(args.exp_dir + "/losses.png")

print("  Value: ", trial.value)

save_best_trial(args.exp_dir, trial, best_val_loss)