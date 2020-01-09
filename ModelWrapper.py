from typing import Callable

import torch
import torch.nn as nn
import torch.utils.data.dataloader


class OccupancyNetworkWrapper(object):
    """
    Implementation of a occupancy network wrapper including training and test method
    """

    def __init__(self, occupancy_network: nn.Module, occupancy_network_optimizer: torch.optim.Optimizer,
                 training_data: torch.utils.data.dataloader, validation_data: torch.utils.data.dataloader,
                 test_data: torch.utils.data.dataloader, loss_function: Callable[[torch.tensor], torch.tensor],
                 device: str = 'cuda') -> None:
        """
        Class constructor
        :param occupancy_network: (nn.Module) Occupancy network for binary segmentation
        :param occupancy_network_optimizer: (torch.optim.Optimizer) Optimizer of the occupancy network
        :param training_data: (torch.utils.data.dataloader) Dataloader including the training dataset
        :param validation_data: (torch.utils.data.dataloader) Dataloader including the validation dataset
        :param test_data: (torch.utils.data.dataloader) Dataloader including the test dataset
        :param loss_function: (Callable[[torch.tensor], torch.tensor]) Loss function to use
        :param device: (str) Device to use while training, validation and testing
        """
        # Init class variables
        self.occupancy_network = occupancy_network
        self.occupancy_network_optimizer = occupancy_network_optimizer
        self.training_data = training_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.loss_function = loss_function
        self.device = device

    def train(self):
        """
        Training method
        :return:
        """
        raise NotImplementedError()

    def test(self):
        """
        Testing method
        :return:
        """
        raise NotImplementedError()
