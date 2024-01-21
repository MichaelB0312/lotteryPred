import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from parser_code import args, MAX_STRONG
from Models import TransformerModel
import dataset_split
from utils import parse_trial_parameters

_, trial_params = parse_trial_parameters(args.best_exp)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
_, _, test_data = dataset_split.split_data('../data/Updated_Sballs.csv', toShuffle=False)
model_input = test_data[0]
def generate(model, model_input, nwords=1, temp=1.0):
    model.eval()
    sballs = []
    with torch.no_grad():
        for i in range(nwords):
            output = model(model_input, None)
            output = torch.softmax(output, dim=-1)
            numbers_weights = output[-1].squeeze().div(temp).exp().cpu()
            while True: #reject sampling special token
                num_idx = torch.multinomial(numbers_weights, 1)[0]
                if num_idx != 0:
                    break
            num_tensor = torch.Tensor([num_idx]).long().to(device)
            model_input = torch.cat([model_input, num_tensor], 0)
            sballs.append(num_idx)
    return sballs

ntoken = MAX_STRONG + 1
model = TransformerModel(
    ntoken, trial_params['ninp'], trial_params['nhead'],
    trial_params['nhid'], trial_params['nlayers'],
    trial_params['dropout'], trial_params['norm_first']
)


model.load_state_dict(torch.load('./best_model.pth'))

for i in range(0,args.n_samples):
  sballs = generate(model=model, model_input=model_input)
  print(sballs)
