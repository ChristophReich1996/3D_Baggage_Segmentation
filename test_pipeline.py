
import torch.nn as nn
import torch.optim as optim
from networks import *
from data_interface import WeaponDataset, many_to_one_collate_fn_test



#number_device = 7
#print("GPU Used:", number_device)
#torch.cuda.set_device(number_device)

train_end = 2**10

print("Load Dataset:", end = " ", flush=True)
test_set = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
                        npoints=2**16,
                        side_len=8,
                        length=1,
                        offset=2802,
                        test=True,
                        sampling='default')
print("Test Set Completed" , end=" - ", flush=True)


print("", flush=True)
print("Building Network", end=" ", flush=True)
network = Network_Generator(rate_learn=1e-4, 
                            size_batch=2**0, 
                            size_iter=2**8, 
                            size_print_every=2**5, 
                            oj_loss=nn.MSELoss(reduction='mean'), 
                            optimizer=optim.Adam, 
                            oj_model=Res_Auto_3d_Model_Occu_Parallel().to(device), 
                            collate_fn=many_to_one_collate_fn_test)                           
print("Completed", flush=True)
print("", flush=True)

print("Testing", flush=True)
network.draw(test_set, 8)
#print(network.test(test_set, True))
model = Res_Auto_3d_Model_Occu_Parallel()
#print(sum(p.numel() for p in model.parameters()))
