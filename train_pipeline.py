
import torch.nn as nn
import torch.optim as optim
from networks_hilo import Res_Auto_3d_Model_Occu_Parallel, Network_Generator
from data_interface import WeaponDataset, many_to_one_collate_fn
from config import device
import argparse

parser = argparse.ArgumentParser(description='Training Pipeline for a combined HiLo Network')
parser.add_argument('-sl', '-side_len', required='True', choices=['16', '32', '48', '64'])
parser.add_argument('-sld', '-side_len_down', required='True', choices=['4', '8', '16', '32'])
parser.add_argument('-df', '-down_factor', required='True', choices=['4', '8', '16', '32'])
parser.add_argument('-np', '-npoints', required='True', type=int, choices=range(14))
parser.add_argument('-lr', '-learning_rate', required='True', choices=['3', '4', '5', '6'])
parser.add_argument('-l', '-load', required='True', choices=['True', 'False'])
parser.add_argument('-n', '-name', required='False', default='')
args = parser.parse_args()

side_len = int(args.sl)
npoints = 2**int(args.np)
lr = 1 * 10**(-float(args.lr))
side_len_down = int(args.sld)
down_fact = int(args.df)
load = bool(args.l == 'True')
name = args.n

print("Load Datasets:", end = " ", flush=True)
train_dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                              length=2600)
print("Training Set Completed" , end=" - ", flush=True)
val_dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                            length=128, 
                            offset=2600)
print("Validation Set Completed", flush=True)

print("", flush=True)
print("Building Network", end=" ", flush=True)
network = Network_Generator(rate_learn=lr, 
                            size_iter=2**14, 
                            size_print_every=2**8, 
                            oj_loss=nn.BCELoss(reduction='mean'), 
                            optimizer=optim.Adam, 
                            oj_model=Res_Auto_3d_Model_Occu_Parallel().to(device), 
                            collate_fn=many_to_one_collate_fn)                           
print("Completed", flush=True)
print("", flush=True)
print("Training", flush=True)
network.train(train_dataset=train_dataset, 
              val_dataset=val_dataset, 
              side_len=side_len, 
              npoints=npoints, 
              name=name, 
              load=load, 
              down_fact=down_fact, 
              side_len_down=side_len_down)


