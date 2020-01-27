
import torch.nn as nn
import torch.optim as optim
from networks_hilo import *
from data_interface import WeaponDataset, many_to_one_collate_fn
import argparse

parser = argparse.ArgumentParser(
    description='Testing and Drawing Pipeline for a combined HiLo Network')
# Side len of extracted window
parser.add_argument('-sl', '-side_len', required='True',
                    choices=['16', '32', '48', '64'])
# Side len of downsampled extracted window
parser.add_argument('-sld', '-side_len_down',
                    required='True', choices=['4', '8', '16', '32'])
# Downsampling factor
parser.add_argument('-df', '-down_factor', required='True',
                    choices=['2', '4', '8', '16', '32'])
# Number of points to sample
parser.add_argument('-np', '-npoints', required='True',
                    type=int, choices=range(14))
# Learning rate
parser.add_argument('-lr', '-learning_rate', required='True',
                    choices=['3', '4', '5', '6'])
# Number of iterations for testing TODO: implement
parser.add_argument('-i', '-iterations', required='True',
                    type=int, choices=range(1, 1000))
# Whether to draw or test
parser.add_argument('-a', '-action', required='True', choices=['draw', 'test'])
# Name to save and restore networks
parser.add_argument('-n', '-name', required='False', default='')
args = parser.parse_args()

side_len = int(args.sl)
npoints = 2**int(args.np)
lr = 1 * 10**(-float(args.lr))
side_len_down = int(args.sld)
down_fact = int(args.df)
name = args.n
i = int(args.i)
action = args.a

length = 1 if action == 'draw' else 200
offset = 2791 if action == 'draw' else 2729
print("Load Dataset:", end=" ", flush=True)
test_dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                             length=length,
                             offset=offset)
print("Test Set Completed", end=" - ", flush=True)


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

print("Testing", flush=True)

if action == 'draw':
    network.draw_fast(test_dataset, 1, name, down_fact=down_fact,
                      side_len_down=side_len_down)
else:
    print(network.test(test_dataset=test_dataset,
                       side_len=side_len,
                       npoints=npoints,
                       name=name,
                       down_fact=down_fact,
                       side_len_down=side_len_down))
