from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
from pykdtree.kdtree import KDTree
from torchsummary import summary
import Misc


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
        self.occupancy_network = occupancy_network.to(device)
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
        progress_bar = tqdm(total=epochs * len(self.training_data.dataset))
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
                volumes = volumes.to(self.device)
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
                progress_bar.set_description('Epoch {}/{}, Loss={:.4f}'.format(epoch + 1, epochs, loss.item()))
                # Save loss value and current epoch
                self.logging(metric_name='train_loss', value=loss.item())
                self.logging(metric_name='epoch', value=epoch)
            # Save best model
            average_loss = self.get_average_metric_for_epoch(metric_name='train_loss', epoch=epoch)
            if save_best_model and (best_loss > average_loss):
                torch.save(self.occupancy_network,
                           model_save_path + 'occupancy_network_' + self.device + '.pt')
        progress_bar.close()

    def test(self, draw: bool = True, side_len: int = 1, threshold: float = 0.5) -> Tuple[float, float, float, float]:
        # Init progress bar
        progress_bar = tqdm(total=len(self.test_data))
        # Get downsampling factor for input and calculate usampling factor
        upsample_factor = self.test_data.dataset.side_len ** 3
        # Calc no grads
        with torch.no_grad():
            # Iterate over test dataset
            for idx, batch in enumerate(self.test_data):
                # Update progress bar
                progress_bar.update(1)
                # Model into eval mode
                self.occupancy_network.eval()
                # Get batch data
                volume, coordinates, labels, actual = batch
                # Data to device
                volume = volume.to(self.device)
                coordinates = coordinates.to(self.device)
                labels = labels.to(self.device)
                actual = actual.to(self.device)
                # Make prediction
                prediction = self.occupancy_network(volume, coordinates)
                # Set offset
                prediction_offset = (prediction > threshold).float()
                # Reshape prediction offset tensor by removing dimension
                prediction_offset = torch.squeeze(prediction_offset)
                # Calc coordinates predicted as a weapon
                weapon_prediction = coordinates[prediction_offset == 1.0]
                # Reshape actual tensor
                actual_ = actual.reshape(-1, 3)
                # Draw weapon prediction
                if draw:
                    Misc.draw_test(weapon_prediction, actual_, volume, side_len, idx)
                # Calc intersection over union
                iou = Misc.intersection_over_union(prediction, coordinates, actual[0], threshold=threshold)
                self.logging('iou', iou.item())
                # Calc precision
                precision = Misc.precision(prediction, coordinates, actual[0], threshold=threshold)
                self.logging('precision', precision.item())
                # Calc recall
                recall = Misc.recall(prediction, coordinates, actual[0], threshold=threshold)
                self.logging('recall', recall.item())
                # Calc loss
                loss = self.loss_function(prediction, labels)
                self.logging('test loss', loss.item())
                # Get memory consumption of tensors (upsample volume to original)
                size_volume = Misc.get_tensor_size_mb(volume) * upsample_factor
                size_prediction = Misc.get_tensor_size_mb(prediction)
                size_actual = Misc.get_tensor_size_mb(actual)
                # TODO: replace with bounding box tensor
                self.logging('size_volume', size_volume)
                self.logging('size_prediction', size_prediction)
                self.logging('size_actual', size_actual)

            # Close progress bar
            progress_bar.close()
        # Get average metrics
        test_iou = self.get_average_metric('iou')
        test_precision = self.get_average_metric('precision')
        test_recall = self.get_average_metric('recall')
        test_loss = self.get_average_metric('test loss')
        test_size_volume = self.get_average_metric('size_volume')
        test_size_prediction = self.get_average_metric('size_prediction')
        test_size_actual = self.get_average_metric('size_actual')
        # Print metrics
        print('Intersection over union = {}'.format(test_iou))
        print('Precision = {}'.format(test_precision))
        print('Recall = {}'.format(test_recall))
        print('Test loss = {}'.format(test_loss))
        print('Average memory usage per sample: Original volume = {}MB, Label = {}MB, Prediction = {}MB'.format(test_size_volume, test_size_actual, test_size_prediction))
        return test_iou, test_precision, test_recall, test_loss

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
        return float(metric_average)

    def get_average_metric(self, metric_name: str) -> float:
        """
        Method calculates the average of a metric
        :param metric_name: (str) Name of the metric
        :return: (float) Average metric
        """
        # Convert lists to np.array
        metric = np.array(self.metrics[metric_name])
        # Calc mean
        metric_average = np.mean(metric)
        return float(metric_average)

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
