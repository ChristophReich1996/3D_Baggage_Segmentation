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
                 device: str = 'cuda', batch_size: int = 2**2, collate_fn: Callable[torch.tensor]) -> None:
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

    def train(self, training_data, validation_data, load=False):
        """
        Training method
        :param training_data: (torch.tensor) Training Set
        :param validation_data: (torch.tensor) Validation Set
        :param load: (Bool) Determine whether or not model should be loaded from disk
        :return:
        """

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

    def test(self):
        """
        Testing method
        :return:
        """
        raise NotImplementedError()
