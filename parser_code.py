import argparse


# I/O args
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--exp-dir", type=str, default="./StrongBallsExp", help="experiment name")
parser.add_argument("--exp-par", type=str, default="/", help="experiment params")
parser.add_argument("--csv_file", type=str, default="/vaeTrans", help="experiment name")
# training and optimization args
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--n-epochs", type=int, default=80, help="number of maximum training epochs")
# MAX number of balls
parser.add_argument("--max-weak", type=int, default=37, help="max id of weak balls")
parser.add_argument("--max-strong", type=int, default=7, help="max id of weak balls")

# model args
parser.add_argument("--trans_layers", type=int, default=2, help="number of Transformer Encoder layers")
parser.add_argument("--x_dim", type=int, default=6, help="input dimension")
parser.add_argument("--z_dim", type=int, default=10, help="latent dimension")
parser.add_argument("--hidden_size", type=int, default=128, help="size of the hidden layers for reparametrization trick and for decoder")
parser.add_argument("--extraDec", type=bool, default=False, help="extra layer for VAE decoder")
parser.add_argument("--extra_dec_dim", type=int, default=64, help="dim of extra layer")
parser.add_argument("--recon_loss", type=str, default="mse", help="the reconstruction loss function", choices=["bce", "mse", "l1"])

# Strong Balls Exp args
parser.add_argument('-sb', '--strong-batch', default=20, type=int, metavar='N', help='mini-batch size for Strong Balls')
parser.add_argument('-esb', '--eval-strong-batch', default=10, type=int, metavar='N', help='eval-batch size for Strong Balls')
parser.add_argument('--embdSize', default=220, type=int, help='embedding size for transformer')
parser.add_argument('--hidden_units', default=250, type=int, help='dim. for MLP layers')
parser.add_argument('--enc_layers', default=4, type=int, help='num of encoder layers')
parser.add_argument('--atn_layers', default=2, type=int, help='num of attention units')
parser.add_argument('--drop', default=0.1, type=float, help='dropout')
parser.add_argument("--norm_first", type=bool, default=False, help="the reconstruction loss function", choices=[False,True])
parser.add_argument('--strg_epochs', default=40, type=int, help='num of epochs')
parser.add_argument('--bptt', default=10, type=int, help='len sample and target for seq-to-seq')
parser.add_argument('--gama', default=0.95, type=float, help='factor reducing for lr scheduler')
parser.add_argument('--step_size', default=1, type=int, help='step size for lr scheduler')
parser.add_argument('--n_trials', default=100, type=int, help='num of trials for optuna')
parser.add_argument('--timeout', default=600, type=int, help='timeout for optuna')

args = parser.parse_args()
MAX_WEAK = args.max_weak
MAX_STRONG = args.max_strong
if args.exp_dir == '/vaeTrans':
    args.exp_par = args.exp_par + str(args.n_epochs) + "_epochs_" + str(args.trans_layers) + "_layers"
    if args.extraDec:
        args.exp_par = args.exp_par + "_" + str(args.extra_dec_dim) + "dim_extraDec"
elif args.exp_dir == './StrongBallsExp':
    args.exp_par = args.exp_par + str(args.strg_epochs) + "_epochs_" + str(args.strong_batch) + "_batchSize_" + str(args.bptt) + "_seqLen"
args.exp_dir = args.exp_dir + args.exp_par