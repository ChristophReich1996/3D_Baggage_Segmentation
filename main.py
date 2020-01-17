import torch
from torch.nn import BCELoss
import torchsummary
import numpy as np
import matplotlib.pyplot as plt


import Models
import Datasets
import ModelWrapper

if __name__ == '__main__':
    model = Models.OccupancyNetwork()
    model(torch.rand([2, 1, 80, 52, 77]), torch.rand([2 ** 15, 3]))
#     ModelWrapper.OccupancyNetworkWrapper(occupancy_network=model,
#                                          occupancy_network_optimizer=torch.optim.Adam(model.parameters(),lr=1e-05),
#                                          training_data=Datasets.WeaponDataset(
#                                                 target_path="/fastdata/Smiths_LKA_Weapons/len_1/",
#                                                 npoints=2**14,
#                                                 side_len=8,
#                                                 length=2600),
#                                          validation_data=None,
#                                          test_data=None,
#                                          loss_function=BCELoss(reduction='mean')
#                                          ).train(epochs=2)