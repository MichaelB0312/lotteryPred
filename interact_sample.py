import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Models import Vae
from parser_code import args
import tkinter as tk
from tkinter import Button, Label, StringVar, Tk, Text, END
import pickle

# Load the args from the file
with open('args.pkl', 'rb') as f:
    args = pickle.load(f)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae = Vae(x_dim=args.x_dim, z_dim=args.z_dim, hidden_size=args.hidden_size, device=device, num_layers=args.trans_layers, extraDecFC_dim=args.extra_dec_dim, isExtraLayer=args.extraDec).to(device)
betas = {'0.5': args.exp_dir + '/beta_0.5.pth',
         '1.5': args.exp_dir + '/beta_1.5.pth',
         '2': args.exp_dir + '/beta_2.pth'}

print(args)
# GUI Functions
def load_checkpoint(beta_value):
    vae.load_state_dict(torch.load(betas[beta_value]))
    result_text.delete(1.0, END)  # Clear the previous results when loading a new checkpoint

def sample_and_display():
    n_samples = 5
    vae_samples = vae.sample(num_samples=n_samples).data.cpu().numpy().reshape(5, 6)
    result_text.delete(1.0, END)  # Clear previous results
    result_text.insert(tk.END, f"Shape: {vae_samples.shape}\n")
    for j in range(vae_samples.shape[0]):
        result_text.insert(tk.END, f"Sample {j+1}: {vae_samples[j]}\n")

# Create Tkinter GUI
root = Tk()
root.title("VAE Sample GUI")

# Label
label = Label(root, text="Select Beta Value:")
label.pack()

# Dropdown menu
selected_beta = StringVar(root)
selected_beta.set('0.5')  # default value
beta_menu = tk.OptionMenu(root, selected_beta, *betas.keys())
beta_menu.pack()

# Load Checkpoint Button
load_button = Button(root, text="Load Checkpoint", command=lambda: load_checkpoint(selected_beta.get()))
load_button.pack()

# Sample and Display Button
sample_button = Button(root, text="Sample and Display", command=sample_and_display)
sample_button.pack()

# Results Text Box
result_text = Text(root, height=10, width=40)
result_text.pack()

# Run Tkinter event loop
root.mainloop()
