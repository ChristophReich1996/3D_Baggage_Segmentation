from typing import Callable, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data.dataloader
from datetime import datetime
from pykdtree.kdtree import KDTree
from torchsummary import summary

class OccupancyNetworkWrapper(object):
    """
    Implementation of a occupancy network wrapper including training and test method
    """

    def __init__(self, occupancy_network: nn.Module, occupancy_network_optimizer: torch.optim.Optimizer,
                 training_data: torch.utils.data.dataloader, validation_data: torch.utils.data.dataloader,
                 test_data: torch.utils.data.dataloader, loss_function: Callable[[torch.tensor], torch.tensor],
                 collate_fn: Callable[torch.tensor], device: str = 'cuda', batch_size: int = 2**2) -> None:
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
        self.loss_function = loss_function
        self.device = device
        self.batch_size = batch_size
        self.collate_fn = collate_fn

        # Better pass to train function
        # self.training_data = training_data
        # self.validation_data = validation_data
        # self.test_data = test_data

    def train(self, training_data, validation_data, load=False) -> None:
        """
        Training method
        :param training_data: (torch.tensor) Training Set
        :param validation_data: (torch.tensor) Validation Set
        :param load: (Bool) Determine whether or not model should be loaded from disk
        :return:
        """
        # Init dict for evaluation metrics
        self.metrics = dict()

        # Function vars ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        loss_best = float('inf')
        losses_train = []
        losses_val = []
        loader_train = DataLoader(dataset=training_data, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=self.collate_fn)
        loader_val = DataLoader(dataset=validation_data, batch_size=self.batch_size, pin_memory=True, shuffle=True, collate_fn=self.collate_fn)
        
        if load:
            self.occupancy_network.load_state_dict(torch.load( "model/"+ type(self.occupancy_network).__name__ + "_" + str(device) + ".pt"))
            self.occupancy_network_optimizer.load_state_dict(torch.load( "optimizer/"+ type(self.occupancy_network).__name__ + "_" + str(device) + ".pt"))

        # Auxiliary functions ++++++++++++++++++++++++++++++++++++++++++++++++++
        # Make a training step
        def _step_train(batch):
            volume, coords, labels = batch
            self.occupancy_network.train()
            yhat = self.occupancy_network(volume.to(device), coords.to(device))
            loss_train = self.loss_function(yhat, labels.to(device))
            loss_train.backward()
            self.occupancy_network_optimizer.step()
            self.occupancy_network_optimizer.zero_grad()
            return loss_train.item()


        # Logic ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for _ in range(self._size_iter):

            losses_train_batch = []
            for i, batch in enumerate(loader_train):
            
                # One step of training
                loss_train_batch = _step_train(batch)
                if i % 4 == 3:
                    print("Training Loss Batch", i, loss_train_batch,flush=True)

                if i % self._size_print_every == self._size_print_every-1:
                    loss_val = self._val(loader_val, losses_val)
                    print("Validation Loss", loss_val)

                    if loss_val < loss_best:
                        loss_best = loss_val
                        torch.save(self.occupancy_network.state_dict(), "model/"+ type(self.occupancy_network).__name__ + "_" + str(device) + ".pt")
                        torch.save(self.occupancy_network_optimizer.state_dict(), "optimizer/"+ type(self.occupancy_network).__name__ + "_" + str(device) + ".pt")

            loss_train = np.mean(losses_train_batch)
            losses_train.append(loss_train)
            print("Training Loss Iteration", loss_train,flush=True)

    def test(self, test_dataset, draw=True, side_len=16) -> (np.array, np.array, np.array):
        """
        Testing method
        :param test_dataset: (torch.tensor) Test Set
        :param draw: (Bool) Draw result to file
        :param side_len: (int) Side length
        :return: (np.array) Mean losses test batch, (np.array) Mean precision test batch, (np.array) Mean recall test batch
        """
        # TODO only working with batchsize 1 currently (actual labels)

        loader_test = DataLoader(dataset=test_dataset, batch_size=self.batch_size, pin_memory=False, shuffle=True, collate_fn=self.collate_fn)
        checkpoint = torch.load("model/"+ type(self.occupancy_network).__name__ + "_" + str(device) + ".pt", map_location=lambda storage, loc: storage)
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
                #print("Activation Test", torch.sum(yhat).item())
                locs = coords[hits == 1]
                if draw:
                    to_write = locs.cpu().numpy().astype(np.short)
                    # Only each 10th as meshlab crashes otherwise
                    to_write_act = actual[::10,:].cpu().numpy().astype(np.short)
                    #mean (shape) centering
                    mean = np.array([volume.shape[2] * side_len/2, volume.shape[3] * side_len/2, volume.shape[4] * side_len/2])
                    to_write_act = to_write_act - mean
                    to_write = to_write - mean# np.mean(to_write, axis=0)

                    with open('outfile_auto.obj','w') as f:
                        for line in to_write:
                            f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                             " " + "0.5" + " " + "0.5" + " " + "1.0" + "\n")
                        for line in to_write_act:
                            f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                            " " + "0.19" + " " + "0.8" + " " + "0.19" + "\n")

                        #Corners of volume
                        f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + str(volume.shape[2] * side_len)+  " " + "0" + " " + "0" + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                        
                        f.write("v " + " " + str(volume.shape[2] * side_len) +  " " + str(volume.shape[3] * side_len) + " " + "0" + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                        f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len) + " " + "0" + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len) + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                        f.write("v " + " " + str(volume.shape[2] * side_len)+  " " + "0" + " " + str(volume.shape[4] * side_len) + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                        
                        f.write("v " + " " + str(volume.shape[2] * side_len) +  " " + str(volume.shape[3] * side_len) + " " + str(volume.shape[4] * side_len)+ 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                        f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len) + " " + str(volume.shape[4] * side_len) + 
                            " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                kd_tree = KDTree(actual.cpu().numpy(), leafsize=16)
                dist, _ = kd_tree.query(locs.cpu().numpy(), k=1)
                union = np.sum(dist == 0)
                precision = union/locs.shape[0]
                recall = union/actual.shape[0]
                loss_test_batch = self.loss_function(yhat, labels.to(device)).item()
                losses_test_batch.append(loss_test_batch)
                precision_test_batch.append(precision)
                recall_test_batch.append(recall)
        
        # TODO: check input size, dynamic
        summary(self.occupancy_network, input_size=(80, 52, 77))

        return np.mean(np.array(losses_test_batch)), np.mean(np.array(precision_test_batch)), np.mean(np.array(recall_test_batch))

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
