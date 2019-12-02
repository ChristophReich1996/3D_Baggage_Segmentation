import torch
import torch.utils.data
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib.save as to_tikz


class Statistics(object):
    """
    Statistic class
    """

    def __init__(self, training_dataset: torch.utils.data.Dataset, test_dataset: torch.utils.data.Dataset) -> None:
        """
        Constructor method
        :param torch.utils.data.Dataset training_dataset:
        :param torch.utils.data.Dataset test_dataset:
        """
        # Init training and test dataset
        self.training_dataset = training_dataset
        self.test_dataset = test_dataset

    def plot_histogram(self) -> None:
        pass

    def calc_class_balance(self) -> float:
        pass
