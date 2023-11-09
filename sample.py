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

vae = Vae(x_dim=X_DIM, z_dim=Z_DIM, hidden_size=HIDDEN_SIZE, device=device, num_layers=TRANS_LAYERS, extraDecFC_dim=EXTRA_DEC_DIM, isExtraLayer=EXTRA_DEC).to(device)
betas = {'0.5':'./beta_0.5_vaeTrans_' + str(NUM_EPOCHS) + '_epochs_' + str(TRANS_LAYERS) + '_layers.pth',
         '1.5':'./beta_1.5_vaeTrans_' + str(NUM_EPOCHS) + '_epochs_' + str(TRANS_LAYERS) + '_layers.pth',
         '2':'./beta_2_vaeTrans_' + str(NUM_EPOCHS) + '_epochs_' + str(TRANS_LAYERS) + '_layers.pth'}

if EXTRA_DEC:
  betas = {'0.5':'./beta_0.5_vaeTrans_' + str(NUM_EPOCHS) + '_epochs_' + str(TRANS_LAYERS) + '_layers' + str(EXTRA_DEC_DIM) + "dim_extraDec.pth",
         '1.5':'./beta_1.5_vaeTrans_' + str(NUM_EPOCHS) + '_epochs_' + str(TRANS_LAYERS) + '_layers' + str(EXTRA_DEC_DIM) + "dim_extraDec.pth",
         '2':'./beta_2_vaeTrans_' + str(NUM_EPOCHS) + '_epochs_' + str(TRANS_LAYERS) + '_layers' + str(EXTRA_DEC_DIM) + "dim_extraDec.pth"}
#print("loaded checkpoint from", vae.load_state_dict(torch.load('/content/beta_0.05_vae_50_epochs.pth')))

for i in range(0,3):
  # now let's sample from the vae
  n_samples = 5
  #print(vae.sample(num_samples=n_samples).shape)
  vae_samples = vae.sample(num_samples=n_samples).data.cpu().numpy().reshape(5,6)
  print("shape", vae_samples.shape)
  fig = plt.figure(figsize=(8 ,5))
  for j in range(vae_samples.shape[0]):
    vae.load_state_dict(torch.load(list(betas.values())[i]))
    print(vae_samples[j])