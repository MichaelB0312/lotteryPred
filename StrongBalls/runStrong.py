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
import dataset_split

train_data, val_data, test_data = dataset_split.split_data('../data/strong_balls.csv', toShuffle=False)
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
for i in range(train_data.shape[1]):
    word = sample[i]
    print(word)
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


