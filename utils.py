
import torch.optim as optim

# Save the losses and trial parameters to txt file
def save_best_trial(directory, best_val_loss, trial):
    with open(directory + '/stats.txt', 'w') as f:
        f.write("Validation Loss: {}\n".format(best_val_loss))
        f.write("Trial Parameters:\n")
        print("  Params: ")
        for key, value in trial.params.items():
            f.write("  {}: {}\n".format(key, value))
            print("    {}: {}".format(key, value))

# Function to parse trial parameters from the text file
def parse_trial_parameters(file_path):
    trial_params = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Extract validation loss
    validation_loss = float(lines[0].split(': ')[1].strip())

    # Extract trial parameters
    for line in lines[2:]:
        key, value = line.strip().split(': ')
        if key == 'optimizer':
            trial_params[key] = getattr(optim, value)  # Use getattr to dynamically fetch the optimizer class
        else:
            trial_params[key] = eval(value)  # Use eval to interpret other values correctly

    return validation_loss, trial_params




