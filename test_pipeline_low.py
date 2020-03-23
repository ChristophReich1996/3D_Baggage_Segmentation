
import torch.nn as nn
import torch.optim as optim
from networks import *
from data_interface import WeaponDataset, many_to_one_collate_fn


#number_device = 7
#print("GPU Used:", number_device)
# torch.cuda.set_device(number_device)

npoints = 2**10
side_len = 32

print("Load Dataset:", end=" ", flush=True)
test_set = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons_Down/len_8/",
                         length=1,
                         offset=2802)
print("Test Set Completed", end=" - ", flush=True)


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

print("Testing", flush=True)
network.draw_low(test_set, 2, "Low")
# print(network.test(test_set))
#model = Res_Auto_3d_Model_Occu_Parallel()
#print(sum(p.numel() for p in model.parameters()))
