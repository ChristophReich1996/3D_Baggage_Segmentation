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

    def __init__(self, dataset: torch.utils.data.Dataset, long_factor: int = 10000,
                 batch_size: int = 1, num_workers: int = 0) -> None:
        """
        Constructor method
        :param torch.utils.data.Dataset dataset: Weapon Dataset
        """
        # Check if dataset is in train mode
        assert dataset.test, 'Dataset must me in train mode'
        # Init dataset
        self.dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=many_to_one_collate_fn_test)
        # Temporary: factor that scales down the float32 to long values
        self.long_factor = long_factor

    def calculcate_histogram(self, cuda=False) -> None:
        
        # Set GPU device if cuda is set to True
        gpu_device = 1

        print('calculating histogram ...')

        data_hist = torch.tensor([])

        # Check if cuda is set to True and use GPU if so
        if cuda:
            data_hist = data_hist.cuda(device=gpu_device)

        pro_bar = tqdm(total=len(self.dataset))

        # Get batches from dataset
        for volume, coordinates, label, label_n in self.dataset:
            
            pro_bar.update(1)

            if cuda:
                volume = volume.cuda(device=gpu_device)

            # Flatten volume to work with it as if it would be a list
            vol_flat = volume.flatten()

            # bincount not implemented for float32 so conversion to long is needed
            vol_flat_long = (vol_flat * self.long_factor).long()

            # Use bincount to count value occurences
            vol_bincount = torch.bincount(vol_flat_long)

            # Add padding to enable sum operation
            if len(vol_bincount) > len(data_hist):
                data_hist = nn.ConstantPad1d((0, len(vol_bincount) - len(data_hist)), 0)(data_hist)
            elif len(vol_bincount) < len(data_hist):
                vol_bincount = nn.ConstantPad1d((0, len(data_hist) - len(vol_bincount)), 0)(vol_bincount)

            # sum the final bins and the bins of the volume
            data_hist = data_hist + vol_bincount

        pro_bar.close()

        # Save to later load in plot_histogram
        torch.save(data_hist, './data_hist.pt')

        # index of hist represents values and hist[index] represents occurence
        # note that values are not absolute but the 'long' that were produced by multiplying the 'float' by 10000 and converting them to 'long'


    def plot_histogram(self) -> None:

        print('plotting histogram ...')
        
        # Load bins
        data_hist = torch.load('./data_hist.pt')

        # Get minimum and maximum value
        # For the minimum check if index 0 is not 0 meaning that 0 does not occur
        # Contiue until a value appears which is indicated by a value > 0 at the given index
        min_val = None
        for i in range(len(data_hist)):
            if data_hist[i] != 0:
                if i != 0:
                    min_val = i / self.long_factor
                else:
                    min_val = i
                break

        max_val = (data_hist.shape[0]-1)/self.long_factor

        print('min', min_val, 'max', max_val)

        # Create bar plots of the bins -> histograms
        # Save them and clear plot

        # Plot all bins
        plt.bar(np.arange(data_hist.shape[0]), data_hist)
        # plt.locator_params(axis='x', nbins=25)
        # plt.locator_params(axis='y', nbins=4)
        # plt.axes.set_xticklabels(np.arange(min_val, max_val, ((max_val-min_val)/data_hist.shape[0])))
        plt.savefig('histogram_full.png')
        plt.clf()

        # Squeeze into 100 bins
        hist = torch.histc(data_hist.float(), bins=100)

        # plt.bar((np.arange(hist.shape[0])), hist)
        # plt.savefig('histogram_100.png')
        # plt.clf()

        # plt.bar(np.arange(hist.shape[0]-1), hist[1:])
        # plt.savefig('histogram_100_reduced_01.png')
        # plt.clf()

        plt.bar(np.arange(hist.shape[0]-2), hist[2:])
        plt.savefig('histogram_100_reduced_02.png')
        plt.clf()


    def calc_class_balance(self) -> float:

        print('calculating class balance ...')

        # Init class balance variable
        class_balance = 0.0

        pro_bar = tqdm(total=len(self.dataset))
        
        # Itt over dataset
        for volume, coordinates, label, label_n in self.dataset:
            
            pro_bar.update(1)

            # Calc number of elements in volume and label_n
            volume_size = torch.tensor(166656000) # torch.prod(torch.tensor(volume.shape))
            label_size = torch.tensor(label_n.shape[0])

            label_size_tmp = torch.tensor(label.shape[0])

            # print('balance_batch', (label_size.float() / volume_size.float()), 'label_size', label_size.float(), 'volume_size', volume_size.float(), 'label_size_tmp', label_size_tmp.float())

            # Calc class balance
            class_balance += (label_size.float() / volume_size.float())

        pro_bar.close()

        # Average over length of number of batches
        class_balance /= len(self.dataset)

        print('class balance final: ', class_balance.item())

        np.save('./class_balance.npy', np.array(class_balance))
        
        return class_balance


if __name__ == '__main__':
    stats = Statistics(WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_16/", npoints=2 ** 14,
                                     side_len=16, length=2400, test=True), batch_size=1, num_workers=0)
    stats.calc_class_balance()
    stats.calculcate_histogram()
    stats.plot_histogram()

# added 'mmap_mode' param in data_interface
# volume_n = np.load(self.target_path + str(index) + ".npy", mmap_mode="r")
# label_n = np.load(self.target_path + str(index) + "_label.npy", mmap_mode="r")