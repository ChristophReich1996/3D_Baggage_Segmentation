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
        """
        # Model into train mode
        self.occupancy_network.train()
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
                progress_bar.set_description('Epoch {}/{}, Loss={:.4f}'.format(epoch + 1, epoch + 1, loss.item()))
                # Save loss value and current epoch
                self.logging(metric_name='loss', value=loss.item())
                self.logging(metric_name='epoch', value=epoch)

            average_loss = self.get_average_metric_for_epoch(metric_name='loss', epoch=epoch)
            if save_best_model and (best_loss > average_loss):
                torch.save(self.occupancy_network,
                           model_save_path + '/occupancy_network_' + self.device + '_loss_' + str(
                               average_loss) + '_epoch_' + str(epoch) + '.pt')

    def test(self, test_dataset, draw=True, side_len=16) -> (np.array, np.array, np.array):
        """
        Testing method
        :param test_dataset: (torch.tensor) Test Set
        :param draw: (Bool) Draw result to file
        :param side_len: (int) Side length
        :return: (np.array) Mean losses test batch, (np.array) Mean precision test batch, (np.array) Mean recall test batch
        """
        # TODO only working with batchsize 1 currently (actual labels)

        loader_test = DataLoader(dataset=test_dataset, batch_size=self.batch_size, pin_memory=False, shuffle=True,
                                 collate_fn=self.collate_fn)
        checkpoint = torch.load("model/" + type(self.occupancy_network).__name__ + "_" + str(device) + ".pt",
                                map_location=lambda storage, loc: storage)
        self.occupancy_network.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()

        with torch.no_grad():
            losses_test_batch = []
            precision_test_batch = []
            recall_test_batch = []
            for batch in loader_test:
                self.occupancy_network.eval()
                # Makes predictions
                volume, coords, labels, actual = batch
                yhat = self.occupancy_network.inference(volume.to(device), coords.to(device))
                hits = torch.squeeze(yhat)
                # print("Activation Test", torch.sum(yhat).item())
                locs = coords[hits == 1]
                if draw:
                    to_write = locs.cpu().numpy().astype(np.short)
                    # Only each 10th as meshlab crashes otherwise
                    to_write_act = actual[::10, :].cpu().numpy().astype(np.short)
                    # mean (shape) centering
                    mean = np.array([volume.shape[2] * side_len / 2, volume.shape[3] * side_len / 2,
                                     volume.shape[4] * side_len / 2])
                    to_write_act = to_write_act - mean
                    to_write = to_write - mean  # np.mean(to_write, axis=0)

                    with open('outfile_auto.obj', 'w') as f:
                        for line in to_write:
                            f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                                    " " + "0.5" + " " + "0.5" + " " + "1.0" + "\n")
                        for line in to_write_act:
                            f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                                    " " + "0.19" + " " + "0.8" + " " + "0.19" + "\n")

                        # Corners of volume
                        f.write("v " + " " + "0" + " " + "0" + " " + "0" +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + "0" + " " + "0" +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + str(
                            volume.shape[3] * side_len) + " " + "0" +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + "0" + " " + str(volume.shape[3] * side_len) + " " + "0" +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + "0" + " " + "0" + " " + str(volume.shape[4] * side_len) +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + "0" + " " + str(
                            volume.shape[4] * side_len) +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + str(volume.shape[2] * side_len) + " " + str(
                            volume.shape[3] * side_len) + " " + str(volume.shape[4] * side_len) +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + "0" + " " + str(volume.shape[3] * side_len) + " " + str(
                            volume.shape[4] * side_len) +
                                " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                kd_tree = KDTree(actual.cpu().numpy(), leafsize=16)
                dist, _ = kd_tree.query(locs.cpu().numpy(), k=1)
                union = np.sum(dist == 0)
                precision = union / locs.shape[0]
                recall = union / actual.shape[0]
                loss_test_batch = self.loss_function(yhat, labels.to(device)).item()
                losses_test_batch.append(loss_test_batch)
                precision_test_batch.append(precision)
                recall_test_batch.append(recall)

        # TODO: check input size, dynamic
        summary(self.occupancy_network, input_size=(80, 52, 77))

        return np.mean(np.array(losses_test_batch)), np.mean(np.array(precision_test_batch)), np.mean(
            np.array(recall_test_batch))

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
