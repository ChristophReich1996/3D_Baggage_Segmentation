import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib.save as to_tikz
from torch.utils.data import DataLoader


class Statistics(object):
    """
    Statistic class
    """

    def __init__(self, dataset: torch.utils.data.Dataset,
                 batch_size: int = 1, num_workers: int = 0) -> None:
        """
        Constructor method
        :param torch.utils.data.Dataset training_dataset:
        :param torch.utils.data.Dataset test_dataset:
        """
        # Check if dataset is in train mode
        assert self.dataset.test, 'Dataset must me in train mode'
        # Init dataset
        self.dataset = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers)

    def plot_histogram(self) -> None:
        pass

    def calc_class_balance(self) -> float:
        # Init class balance variable
        class_balance = 0.0
        # Itt over dataset
        for volume, coordinates, label, label_n in self.dataset:
            # Get number of pixels belonging to gun class
            pixels_gun = torch.sum((label == 1.0).float()).item()
            # Get number of pixels belonging to non gun class
            pixels_non_gun = torch.sum((label == 0.0).float()).item()
            class_balance += pixels_gun / pixels_non_gun
        # Average over length of number of batches
        class_balance /= len(self.dataset)
        return class_balance


if __name__ == '__main__':
    stats = Statistics(WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_16/", npoints=2 ** 14,
                                     side_len=16, length=2600), batch_size=100, num_workers=10)
