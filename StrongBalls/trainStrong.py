
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
from parser_code import args, MAX_STRONG
import dataset_split


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def train(model, train_data, opt, bptt):
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = model.generate_square_subsequent_mask(bptt).to(device)
    Loss = nn.CrossEntropyLoss()

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = dataset_split.get_batch(train_data, i, bptt)
        seq_len = data.size(0)
        if seq_len != bptt:  # only on last batch
            src_mask = src_mask[:seq_len, :seq_len]
        output = model(data, src_mask) # complete
        loss = Loss(output.view(-1, MAX_STRONG+1), targets.view(-1)) # complete

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = Learning_Scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            #ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} ')
            total_loss = 0
            start_time = time.time()


def evaluate(model, eval_data):
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = model.generate_square_subsequent_mask(args.bptt).to(device)
    Loss = nn.CrossEntropyLoss()
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, args.bptt):
            data, targets = dataset_split.get_batch(eval_data, i, args.bptt)
            seq_len = data.size(0)
            if seq_len != args.bptt:
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            output_flat = output.view(-1, MAX_STRONG+1)
            total_loss += seq_len * Loss(output_flat, targets.view(-1)).item()
    return total_loss / (len(eval_data) - 1)