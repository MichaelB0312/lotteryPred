import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Models import Vae, TransformerModel
from parser_code import args
import tkinter as tk
from tkinter import Button, Label, StringVar, Tk, Text, END
import pickle
from utils import parse_trial_parameters


# Load the args from the file
with open(args.exp_dir + '/args.pkl', 'rb') as f:
    args_vae = pickle.load(f)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

vae = Vae(x_dim=args_vae.x_dim, z_dim=args_vae.z_dim, hidden_size=args_vae.hidden_size, device=device, num_layers=args_vae.trans_layers, extraDecFC_dim=args_vae.extra_dec_dim, isExtraLayer=args_vae.extraDec).to(device)


betas = {'0.5': args_vae.exp_dir + '/beta_0.5.pth',
         '1.5': args_vae.exp_dir + '/beta_1.5.pth',
         '2': args_vae.exp_dir + '/beta_2.pth'}

print(args)
# GUI Functions
# Function to change the background color
def change_background_color():
    current_color = root.cget("bg")
    new_color = "cyan" if current_color == "yellow" else "red"
    root.config(bg=new_color)
    root.after(2000, change_background_color)  # Repeat every 2000 milliseconds (2 seconds)
def load_checkpoint(beta_value):
    vae.load_state_dict(torch.load(betas[beta_value]))
    result_text.delete(1.0, END)  # Clear the previous results when loading a new checkpoint

def sample_and_display():
    vae_samples = vae.sample(num_samples=args.n_samples).data.cpu().numpy().reshape(args.n_samples, 6)
    result_text.delete(1.0, END)  # Clear previous results
    result_text.insert(tk.END, f"Shape: {vae_samples.shape}\n")
    for j in range(vae_samples.shape[0]):
        result_text.insert(tk.END, f"Sample {j+1}: {vae_samples[j]}\n")


# Create Tkinter GUI
root = Tk()
root.title("VAE Sample GUI")

# Improve Layout
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=20, pady=20)

# Set the initial background color
root.config(bg="red")

# Call the function to start the breathing effect
change_background_color()

# Label
label = Label(frame, text="Select Beta Value:")
label.grid(row=0, column=0, columnspan=2, pady=10)

# Dropdown menu
selected_beta = StringVar(root)
selected_beta.set('0.5')  # default value
beta_menu = tk.OptionMenu(frame, selected_beta, *betas.keys())
beta_menu.grid(row=1, column=0, columnspan=2, pady=10)

# Load Checkpoint Button
load_button = Button(frame, text="Load Checkpoint", command=lambda: load_checkpoint(selected_beta.get()), padx=10, pady=5)
load_button.grid(row=2, column=0, pady=10)

# Sample and Display Button
sample_button = Button(frame, text="Sample and Display", command=sample_and_display, padx=10, pady=5)
sample_button.grid(row=2, column=1, pady=10)

# Results Text Box
result_text = Text(frame, height=40, width=80)
result_text.grid(row=3, column=0, columnspan=2, pady=10)

# Add a Heading
heading_label = Label(frame, text="Variational Autoencoder (VAE) Sample GUI", font=("Helvetica", 16, "bold"))
heading_label.grid(row=0, column=0, columnspan=2, pady=10)

# Run Tkinter event loop
root.mainloop()
