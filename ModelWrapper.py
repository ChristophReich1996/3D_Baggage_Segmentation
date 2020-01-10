from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
from pykdtree.kdtree import KDTree
from torchsummary import summary


class OccupancyNetworkWrapper(object):
    """
    Implementation of a occupancy network wrapper including training and test method
    """

    def __init__(self, occupancy_network: nn.Module, occupancy_network_optimizer: torch.optim.Optimizer,
                 training_data: torch.utils.data.dataloader, validation_data: torch.utils.data.dataloader,
                 test_data: torch.utils.data.dataloader,
                 loss_function: Callable[[torch.tensor, torch.tensor], torch.tensor], device: str = 'cuda') -> None:
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
        self.metrics = dict()

    def train(self, epochs: int = 100, save_best_model: bool = True, model_save_path: str = '') -> None:
        """
        Training loop
        :param epochs: (int) Number of epochs to perform
        :param save_best_model: (int) If true the best model is saved
        :param model_save_path: (str) Path to save the best model
        """
        # Model into train mode
        self.occupancy_network.train()
        self.occupancy_network.to(self.device)
        # Init progress bar
        progress_bar = tqdm(total=epochs * len(self.training_data))
        # Init best loss variable
        best_loss = np.inf
        # Perform training
        for epoch in range(epochs):
            for volumes, coordinates, labels in self.training_data:
                # Update progress bar
                progress_bar.update(volumes.shape[0])
                # Reset gradients
                self.occupancy_network.zero_grad()
                # Data to device
                volumes = volumes.to(self.device).unsqueeze_(0)
                coordinates = coordinates.to(self.device)
                labels = labels.to(self.device)
                # Perform model prediction
                prediction = self.occupancy_network(volumes, coordinates)
                # Compute loss
                loss = self.loss_function(prediction, labels)
                # Compute gradients
                loss.backward()
                # Update parameters
                self.occupancy_network_optimizer.step()
                # Update loss info in progress bar
                progress_bar.set_description('Epoch {}/{}, Loss={:.4f}'.format(epoch + 1, epoch + 1, loss.item()))
                # Save loss value and current epoch
                self.logging(metric_name='train_loss', value=loss.item())
                self.logging(metric_name='epoch', value=epoch)
            # Save best model
            average_loss = self.get_average_metric_for_epoch(metric_name='train_loss', epoch=epoch)
            if save_best_model and (best_loss > average_loss):
                torch.save(self.occupancy_network,
                           model_save_path + '/occupancy_network_' + self.device + '.pt')

    def test(self, test_dataset, draw=True, side_len=16) -> None:
        """
        Testing method
        """
        pass

    def logging(self, metric_name: str, value: float) -> None:
        """
        Method writes a given metric value into a dict including list for every metric
        :param metric_name: (str) Name of the metric
        :param value: (float) Value of the metric
        """
        if metric_name in self.metrics:
            self.metrics[metric_name].append(value)
        else:
            self.metrics[metric_name] = [value]

    def get_average_metric_for_epoch(self, metric_name: str, epoch: int) -> float:
        """
        Method calculates the average of a metric for a given epoch
        :param metric_name: (str) Name of the metric
        :param epoch: (int) Epoch to average over
        :return: (float) Average metric
        """
        # Convert lists to np.array
        metric = np.array(self.metrics[metric_name])
        epochs = np.array(self.metrics['epoch'])
        # Calc mean
        metric_average = np.mean(metric[np.argwhere(epochs == epoch)])
        return metric_average

    @staticmethod
    def save_metrics(metrics: Dict[str, List[float]], path: str, add_time_to_file_name: bool = False) -> None:
        """
        Static method to save dict of metrics
        :param metrics: (Dict[str, List[float]]) Dict including metrics
        :param path: (str) Path to save metrics
        :param add_time_to_file_name: (bool) True if time has to be added to filename of every metric
        """
        # Iterate items in metrics dict
        for metric_name, values in metrics.items():
            # Convert list of values to torch tensor to use build in save method from torch
            values = torch.tensor(values)
            # Save values
            if add_time_to_file_name:
                torch.save(values, path + '/' + metric_name + '_' + datetime.now().strftime("%H:%M:%S") + '.pt')
            else:
                torch.save(values, path + '/' + metric_name + '.pt')
