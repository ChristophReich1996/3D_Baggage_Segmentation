
import torch.nn as nn
import torch.optim as optim
from networks import *
from data_interface import WeaponDataset, many_to_one_collate_fn



number_device = 7
#print("GPU Used:", number_device)
#torch.cuda.set_device(number_device)

train_end = 2**10

print("Load Datasets:", end = " ", flush=True)
training_set = WeaponDataset(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                        start_index=0,
                        end_index=train_end,
                        threshold_min=1700,
                        threshold_max=2700,
                        npoints=2**11,
                        side_len=2**6,
                        sampling='boxed')
print("Training Set Completed" , end=" - ", flush=True)
val_set = WeaponDataset(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                        start_index=2**4,
                        end_index=train_end + 2**3,
                        threshold_min=1700,
                        threshold_max=2700,
                        npoints=2**11,
                        side_len=2**6,
                        sampling='default')
print("Validation Set Completed", flush=True)

print("", flush=True)
print("Building Network", end=" ", flush=True)
network = Network_Generator(rate_learn=1e-4, 
                            size_batch=2**3, 
                            size_iter=2**0, 
                            size_print_every=2**5, 
                            oj_loss=nn.MSELoss(reduction='mean'), 
                            optimizer=optim.Adam, 
                            oj_model=Res_Auto_3d_Model_Occu_Parallel().to(device), 
                            collate_fn=many_to_one_collate_fn)                           
print("Completed", flush=True)
print("", flush=True)
print("Training", flush=True)
network.train(training_set, val_set, False)


