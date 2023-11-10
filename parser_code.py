import argparse

# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./vaeTrans", help="experiment name")
parser.add_argument("--exp-par", type=str, default="/", help="experiment params")
parser.add_argument("--csv_file", type=str, default="/vaeTrans", help="experiment name")
# training and optimization args
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=80, help="number of maximum training epochs")
# model args
parser.add_argument("--trans_layers", type=int, default=2, help="number of Transformer Encoder layers")
parser.add_argument("--x_dim", type=int, default=6, help="input dimension")
parser.add_argument("--z_dim", type=int, default=10, help="latent dimension")
parser.add_argument("--hidden_size", type=int, default=128, help="size of the hidden layers for reparametrization trick and for decoder")
parser.add_argument("--extraDec", type=bool, default=False, help="extra layer for VAE decoder")
parser.add_argument("--extra_dec_dim", type=int, default=64, help="dim of extra layer")
parser.add_argument("--recon_loss", type=str, default="mse", help="the reconstruction loss function", choices=["bce", "mse", "l1"])

args = parser.parse_args()
args.exp_par = args.exp_par + str(args.n_epochs) + "_epochs_" + str(args.trans_layers) + "_layers"
if args.extraDec:
    args.exp_par = args.exp_par + "_" + str(args.extra_dec_dim) + "dim_extraDec"
args.exp_dir = args.exp_dir + args.exp_par