import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from networks import *
from layers import dice_loss
from data_interface import Dataset_Vertex_To_Euclidean_Sparse, collate_fn_sparse


"""
Measuring error on vertices ? Inverse is really costly.
At least measruing roundig error.

(Input) Data normalization, Optimizer, Activation
"""


number_device = 1
print("GPU Used:", number_device)
torch.cuda.set_device(number_device)


print("Load Datasets:", end = " ", flush=True)
training_set = 
print("Training Set Completed" , end=" - ", flush=True)
val_set = 

print("", flush=True)
print("Building Network", end=" ", flush=True)
network = Network_Generator(1e-3, 2**4, 2**0, 2**6, nn.MSELoss(reduction='mean'), optim.Adam, Res_Auto_3d_Model_Sparse_Occu_Prog().to(device), collate_fn_sparse)
print("Completed", flush=True)
print("", flush=True)
print("Training", flush=True)
network.train(training_set, val_set, True)


