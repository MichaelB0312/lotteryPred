#!/bin/bash
#SBATCH -p sm
#SBATCH -x sls-sm-1,sls-2080-[1,3],sls-1080-3,sls-sm-5
#SBATCH -p gpu
#SBATCH -x sls-titan-[0-2,9]
#SBATCH --gres=gpu:4
#SBATCH -c 4
#SBATCH -n 1
#SBATCH --mem=48000
#SBATCH --job-name="psla_fsd"
#SBATCH --output=./log_%j.txt

set -x
source ../../venv-psla/bin/activate
export TORCH_HOME=./


batch_size=16
lr=0.001
epoch=80
exp_name=vaeTrans
transLayers=2
latent_dim=z_dim
extraDec=True
extra_dec_dim=64

exp_dir=./exp/exp--${exp_name}-lr${lr}-batch${batch_size}
mkdir -p $exp_dir

CUDA_CACHE_DISABLE=1 python ../../run.py --exp-dir $exp_dir --n-epochs ${epoch} --batch-size ${batch_size} --lr $lr \
--recon_loss mse --trans_layers ${transLayers} --z_dim ${latent_dim} --hidden_size ${hidden_size} \
--extraDec ${extraDec} --extra_dec_dim ${extra_dec_dim} \
