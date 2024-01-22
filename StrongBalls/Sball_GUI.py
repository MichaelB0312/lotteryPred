
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Models import TransformerModel
from parser_code import args, MAX_STRONG
import tkinter as tk
from tkinter import Button, Label, StringVar, Tk, Text, END
import pickle
from utils import parse_trial_parameters
import dataset_split

# Load the args from the file
_, trial_params = parse_trial_parameters(args.best_exp)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_, _, test_data = dataset_split.split_data('../data/Updated_Sballs.csv', toShuffle=False)
model_input = test_data[0]
ntoken = MAX_STRONG + 1
model = TransformerModel(
    ntoken, trial_params['ninp'], trial_params['nhead'],
    trial_params['nhid'], trial_params['nlayers'],
    trial_params['dropout'], trial_params['norm_first']
).to(device)


print(args)
# GUI Functions
# Function to change the background color
def change_background_color():
    current_color = root.cget("bg")
    new_color = "cyan" if current_color == "yellow" else "red"
    root.config(bg=new_color)
    root.after(2000, change_background_color)  # Repeat every 2000 milliseconds (2 seconds)
def load_checkpoint():
    model.load_state_dict(torch.load('./best_model.pth'))
    result_text.delete(1.0, END)  # Clear the previous results when loading a new checkpoint

def sample_and_display():
    result_text.delete(1.0, END)  # Clear previous results
    for j in range(args.n_samples):
        sballs = model.generate(model_input=model_input)
        result_text.insert(tk.END, f"Sample {j+1}: {sballs[0]}\n")


# Create Tkinter GUI
root = Tk()
root.title("Strong Ball Sample GUI")

# Improve Layout
frame = tk.Frame(root, padx=10, pady=10)
frame.pack(padx=20, pady=20)

# Set the initial background color
root.config(bg="red")

# Call the function to start the breathing effect
change_background_color()


# Load Checkpoint Button
load_button = Button(frame, text="Load Checkpoint", command=load_checkpoint, padx=10, pady=5)
load_button.grid(row=2, column=0, pady=10)

# Sample and Display Button
sample_button = Button(frame, text="Sample and Display", command=sample_and_display, padx=10, pady=5)
sample_button.grid(row=2, column=1, pady=10)

# Results Text Box
result_text = Text(frame, height=40, width=80)
result_text.grid(row=3, column=0, columnspan=2, pady=10)

# Add a Heading
heading_label = Label(frame, text="Strong Ball Sample GUI", font=("Helvetica", 16, "bold"))
heading_label.grid(row=0, column=0, columnspan=2, pady=10)

# Run Tkinter event loop
root.mainloop()
