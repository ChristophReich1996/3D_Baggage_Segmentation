
import torch.nn as nn
import torch.optim as optim
from networks import *
from data_interface import WeaponDataset, many_to_one_collate_fn



#number_device = 7
#print("GPU Used:", number_device)
#torch.cuda.set_device(number_device)

side_len = -1
npoints = 2**10
print("Load Datasets:", end = " ", flush=True)
training_set = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
                        length=2600)
print("Training Set Completed" , end=" - ", flush=True)
val_set = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
                        length=128, 
                        offset=2600)
print("Validation Set Completed", flush=True)

print("", flush=True)
print("Building Network", end=" ", flush=True)
network = Network_Generator(rate_learn=1e-4, 
                            size_iter=2**14, 
                            size_print_every=2**6, 
                            oj_loss=nn.BCELoss(reduction='mean'), 
                            optimizer=optim.Adam, 
                            oj_model=Res_Auto_3d_Model_Occu_Parallel().to(device), 
                            collate_fn=many_to_one_collate_fn)                           
print("Completed", flush=True)
print("", flush=True)
print("Training", flush=True)
network.train(training_set, val_set, side_len, npoints, "Low", load=True, cache_size=1000, win_sampled_size=10)


