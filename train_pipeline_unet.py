
import torch.nn as nn
import torch.optim as optim
from networks_hilo_unet import Res_Auto_3d_Model_Unet_Parallel, Network_Generator
from data_interface import WeaponDataset, many_to_one_collate_fn
from config import device
import argparse
import layers

parser = argparse.ArgumentParser(
    description='Training Pipeline for a combined HiLo Network')
# Side len of extracted window
parser.add_argument('-sl', '-side_len', required='True',
                    choices=['16', '32', '48', '64'])
# Side len of downsampled extracted window
parser.add_argument('-sld', '-side_len_down',
                    required='True', choices=['2', '4', '8', '16', '24', '32'])
# Downsampling factor
parser.add_argument('-df', '-down_factor', required='True',
                    choices=['2', '4', '8', '16', '32'])
# Learning rate
parser.add_argument('-lr', '-learning_rate', required='True',
                    choices=['3', '4', '5', '6'])
# Cache Type
parser.add_argument('-ct', '-cache_type', required='True',
                    choices=['fifo', 'counts', 'hardness'])
# Loss
parser.add_argument('-cr', '-criterion', required='True',
                    choices=['bce', 'mse', 'focal', 'dice'])
# Whether to restore given network
parser.add_argument('-l', '-load', required='True', choices=['True', 'False'])
# Name to save and restore networks
parser.add_argument('-n', '-name', required='False', default='')
args = parser.parse_args()

side_len = int(args.sl)
lr = 1 * 10**(-float(args.lr))
cache_type = str(args.ct)
side_len_down = int(args.sld)
down_fact = int(args.df)
load = bool(args.l == 'True')
name = str(args.n)
win_sampled_size = 16

if args.cr == 'bce':
    oj_loss = nn.BCELoss(reduction='mean')
elif args.cr == 'mse':
    oj_loss = nn.MSELoss(reduction='mean')
elif args.cr == 'focal':
    oj_loss = layers.FocalLoss()
elif args.cr == 'dice':
    oj_loss = layers.DiceLoss(win_sampled_size)

print("Load Datasets:", end=" ", flush=True)
train_dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                              length=2600)
print("Training Set Completed", end=" - ", flush=True)
val_dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                            length=128,
                            offset=2600)
print("Validation Set Completed", flush=True)

print("", flush=True)
print("Building Network", end=" ", flush=True)
network = Network_Generator(rate_learn=lr,
                            size_iter=2**10,
                            size_print_every=2**6,
                            oj_loss=oj_loss,
                            optimizer=optim.Adam,
                            oj_model=Res_Auto_3d_Model_Unet_Parallel().to(device),
                            collate_fn=many_to_one_collate_fn)

print("Completed", flush=True)
print("", flush=True)
print("Training", flush=True)
network.train(train_dataset=train_dataset,
              val_dataset=val_dataset,
              side_len=side_len,
              npoints=0,
              name=name,
              load=load,
              down_fact=down_fact,
              side_len_down=side_len_down,
              cache_type=cache_type,
              win_sampled_size=win_sampled_size)
