import torch
from torch.nn import BCELoss
import torchsummary
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import Lossfunctions
import os

import Models
import Datasets
import ModelWrapper
import Misc

if __name__ == '__main__':
    # Batch size has to be a factor of the number of devices used in data parallel
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2'  # "0, 1, 3, 4, 5"
    # Init model
    model = torch.nn.DataParallel(Models.OccupancyNetwork().cuda())
    # Print model
    print(model)
    # Print number of parameters included in the model
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    # Init data parallel for multi GPU usage
    # model = torch.nn.DataParallel(model)
    # Init model wrapper
    model_wrapper = ModelWrapper.OccupancyNetworkWrapper(occupancy_network=model,
                                                         occupancy_network_optimizer=torch.optim.Adam(
                                                             model.parameters(), lr=1e-03),
                                                         training_data=DataLoader(Datasets.WeaponDataset(
                                                             target_path_volume='/fastdata/Smiths_LKA_Weapons_Down/len_8/',
                                                             target_path_label='/fastdata/Smiths_LKA_Weapons_Down/len_1/',
                                                             npoints=2 ** 14,
                                                             side_len=8,
                                                             length=2580),
                                                             batch_size=30, shuffle=True,
                                                             collate_fn=Misc.many_to_one_collate_fn_sample,
                                                             num_workers=30, pin_memory=True),
                                                         test_data=DataLoader(Datasets.WeaponDataset(
                                                             target_path_volume='/fastdata/Smiths_LKA_Weapons_Down/len_8/',
                                                             target_path_label='/fastdata/Smiths_LKA_Weapons_Down/len_1/',
                                                             npoints=2 ** 17,
                                                             side_len=8,
                                                             length=276,  # 200,
                                                             offset=2580,  # 2600,
                                                             test=True,
                                                             share_box=0),
                                                             batch_size=1, shuffle=True,
                                                             collate_fn=Misc.many_to_one_collate_fn_sample_down,
                                                             num_workers=1, pin_memory=True,
                                                         ),
                                                         validation_data=DataLoader(Datasets.WeaponDataset(
                                                             target_path_volume='/fastdata/Smiths_LKA_Weapons_Down/len_8/',
                                                             target_path_label='/fastdata/Smiths_LKA_Weapons_Down/len_1/',
                                                             npoints=2 ** 17,
                                                             side_len=8,
                                                             length=100,  # 200,
                                                             offset=2856,  # 2600,
                                                             test=True,
                                                             share_box=0),
                                                             batch_size=1, shuffle=True,
                                                             collate_fn=Misc.many_to_one_collate_fn_sample_down,
                                                             num_workers=1, pin_memory=True,
                                                         ),
                                                         loss_function=torch.nn.BCELoss(reduction='mean'),
                                                         device='cuda')

    model_wrapper.train(epochs=100, model_save_path='/visinf/home/vilab15/Projects/3D_baggage_segmentation/')
    model_wrapper.test(threshold=0.5, offset=torch.tensor([0.0, 0.0, 0.0]))  # Best offset=10 threshold=0.8

    # model = torch.load('/visinf/home/vilab16/3D_baggage_segmentation/' + 'occupancy_network_lo_lo_dice_cuda.pt').module
    # ModelWrapper.OccupancyNetworkWrapper(occupancy_network=model,
    #                                      occupancy_network_optimizer=torch.optim.Adam(model.parameters(), lr=1e-05),
    #                                      training_data=None,
    #                                      validation_data=None,
    #                                      test_data=DataLoader(Datasets.WeaponDataset(
    #                                          target_path_volume= '/fastdata/Smiths_LKA_Weapons_Down/len_8/',
    #                                          target_path_label= '/fastdata/Smiths_LKA_Weapons_Down/len_1/',
    #                                          npoints=2 ** 17,
    #                                          side_len=8,
    #                                          length=200,#200,
    #                                          offset=2600,#2600,
    #                                          test=True,
    #                                          share_box=0),
    #                                          batch_size=1, shuffle=True, collate_fn=Misc.many_to_one_collate_fn_sample_down,
    #                                          num_workers=1, pin_memory=True,
    #                                          ),
    #                                     loss_function=BCELoss(reduction='mean')
    #                                     ).test()
