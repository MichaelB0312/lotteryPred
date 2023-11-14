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
import argparse
from Models import Vae
#from run import args
#from parser_code import args
import pickle

# Load the args from the file
with open('args.pkl', 'rb') as f:
    args = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae = Vae(x_dim=args.x_dim, z_dim=args.z_dim, hidden_size=args.hidden_size, device=device, num_layers=args.trans_layers, extraDecFC_dim=args.extra_dec_dim, isExtraLayer=args.extraDec).to(device)
betas = {'0.5':args.exp_dir + '/beta_0.5.pth',
         '1.5':args.exp_dir + '/beta_1.5.pth',
         '2':args.exp_dir + '/beta_2.pth'}

print(args)
#print("loaded checkpoint from", vae.load_state_dict(torch.load('/content/beta_0.05_vae_50_epochs.pth')))

for i in range(0,3):
  # now let's sample from the vae
  n_samples = 5
  #print(vae.sample(num_samples=n_samples).shape)
  vae_samples = vae.sample(num_samples=n_samples).data.cpu().numpy().reshape(5,6)
  print("shape", vae_samples.shape)
  fig = plt.figure(figsize=(8 ,5))
  for j in range(vae_samples.shape[0]):
    vae.load_state_dict(torch.load('./vaeTrans/100_epochs_2_layers_64dim_extraDec/beta_2.pth'))
    print(vae_samples[j])