import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
# import tikzplotlib.save as to_tikz
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_interface import *


class Statistics(object):
    """
    Statistic class
    """

    def __init__(self, dataset: torch.utils.data.Dataset,
                 batch_size: int = 1, num_workers: int = 0) -> None:
        """
        Constructor method
        :param torch.utils.data.Dataset dataset: Weapon Dataset
        """
        # Check if dataset is in train mode
        assert dataset.test, 'Dataset must me in train mode'
        # Init dataset
        self.dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=many_to_one_collate_fn_test)

    def plot_histogram(self) -> None:
        
        gpu_device = 2

        print('calculating histogram ...')

        data_hist = torch.tensor([]).cuda(device=gpu_device)

        pro_bar = tqdm(total=len(self.dataset))

        for volume, coordinates, label, label_n in self.dataset:
            
            pro_bar.update(1)

            # print('vol shape', volume.shape)

            vol = volume.cuda(device=gpu_device)

            vol_flat = vol.flatten()

            # print('vol_flat shape', vol_flat.shape)

            # bincount not implemented for float
            vol_flat_long = (vol_flat * 10000).long()
        
            vol_bincount = torch.bincount(vol_flat_long)

            # print(vol_bincount[:10])
            # print(vol_flat.min(), vol_flat.max(), vol_flat.mean())

            if len(vol_bincount) > len(data_hist):
                data_hist = nn.ConstantPad1d((0, len(vol_bincount) - len(data_hist)), 0)(data_hist)
            elif len(vol_bincount) < len(data_hist):
                vol_bincount = nn.ConstantPad1d((0, len(data_hist) - len(vol_bincount)), 0)(vol_bincount)

            data_hist = data_hist + vol_bincount # torch.cat((data_hist,vol_bincount),dim=0)

            # print('data_hist shape', data_hist.shape)

        pro_bar.close()

        hist = torch.histc(data_hist.float(), bins=100)

        # index of hist represents values and hist[index] represents occurence
        # note that values are not absolute but the 'long' that were produced by multiplying the 'float' by 10000 and converting them to 'long'
        torch.save(hist, './hist.pt')
        # print(hist)

        return hist

    def calc_class_balance(self) -> float:

        print('calculating class balance ...')

        # Init class balance variable
        class_balance = 0.0

        pro_bar = tqdm(total=len(self.dataset))
        
        # Itt over dataset
        for volume, coordinates, label, label_n in self.dataset:

            pro_bar.update(1)

            # Calc number of elements in volume and label_n
            volume_size = torch.prod(torch.tensor(volume.shape))
            label_size = torch.tensor(label_n.shape[0])

            # Calcc class balance
            class_balance += label_size.float() / volume_size.float()

        pro_bar.close()

        # Average over length of number of batches
        class_balance /= len(self.dataset)

        print('class balance final: ', class_balance.item())

        np.save('./class_balance.npy', np.array(class_balance))
        
        return class_balance


if __name__ == '__main__':
    stats = Statistics(WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/", npoints=2 ** 14,
                                     side_len=16, length=2400, test=True), batch_size=1, num_workers=0)
    stats.calc_class_balance()
    stats.plot_histogram()

# added 'mmap_mode' param in data_interface
# volume_n = np.load(self.target_path + str(index) + ".npy", mmap_mode="r")
# label_n = np.load(self.target_path + str(index) + "_label.npy", mmap_mode="r")