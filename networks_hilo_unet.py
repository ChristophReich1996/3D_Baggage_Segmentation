import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
from pykdtree.kdtree import KDTree
import functools
import random
import gc

from data_interface import sample_unet, many_to_one_collate_fn_sample_unet, many_to_one_collate_fn_sample_test, many_to_one_collate_fn_list
import utils
import layers
from config import device


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Base Network Class
class Network_Generator():
    def __init__(self, rate_learn, size_iter, size_print_every, oj_loss, optimizer, oj_model, collate_fn=None):
        """Container & Wrapper of nn.Module. Wraps training, testing, validating and inference of a nn.Module

        Arguments:
            rate_learn {float} -- Learning Rate
            size_iter {long} -- Number of training iterations
            size_print_every {long} -- Validate each ... batches
            oj_loss {nn.loss} -- Loss function to minimize
            optimizer {torch.optimizer} -- Optimizer to use
            oj_model {nn.Module} -- Network model to use

        Keyword Arguments:
            collate_fn {func} -- Custom collate function
        """

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._rate_learning = rate_learn
        self._size_batch = 1
        self._size_iter = size_iter
        self._size_print_every = size_print_every

        # (Function) Objects +++++++++++++++++++++++++++++++++++++++++++++++++++
        self._oj_model = oj_model
        self._oj_optimizer = optimizer(
            self._oj_model.parameters(), lr=self._rate_learning)
        self._oj_loss = oj_loss
        self.mse = nn.MSELoss(reduction='mean')
        self._collate_fn = collate_fn

    def test(self, test_dataset, side_len, name, down_fact, side_len_down, batch_size=1):
        """Test wrapped network

        Arguments:
            test_dataset {torch dataset} -- Dataset to test on network
            side_len {long} -- Extracted window size
            npoints {long} -- Number of points to sample
            name {string} -- Name of stored network file
            down_fact {long} -- Downsampled factor for downsampled window
            side_len_down {long} -- Extracted down sampled window size

        Returns:
            Test loss and IOU
        """
        # Get test set
        loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 pin_memory=False, shuffle=False, collate_fn=many_to_one_collate_fn_list)

        # Restore network for testing
        checkpoint = torch.load("model/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt",
                                map_location=lambda storage, loc: storage)
        print("Loaded:", "model/" + type(self._oj_model).__name__ +
              "_" + str(device) + name + ".pt")
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        # Test with same settings as for training and additionally calculate IOU
        iou_per_batch = []
        with torch.no_grad():
            for batch_count, batch in enumerate(loader_test):
                self._oj_model.eval()
                volume_in, label_in = batch
                shapes = np.max(
                    np.array([volume_in[i].shape for i in range(batch_size)]), axis=0)

                intersection_batch = torch.zeros((batch_size)).to(device)
                union_batch = torch.zeros((batch_size)).to(device)

                # Consider any extractable window of size side_len**3 to write total object
                for vol_x in range(0, shapes[1], side_len):
                    print("batch", batch_count, "x", vol_x)
                    print(torch.mean(
                        intersection_batch / (union_batch + 0.00001)).item(), flush=True)
                    for vol_y in range(0, shapes[2], side_len):
                        for vol_z in range(0, shapes[3], side_len):
                            # Get extracted window and downsampled extracted window, SEE sample in data_interface.py for further information

                            samples = []
                            samples_id = []
                            for i in range(batch_size):
                                # Sample volume, to get original volume back,
                                # as well as downsampled volume
                                # Continue if volume too small vor the given position
                                if volume_in[i].shape[1] - vol_x <= 0 or \
                                        volume_in[i].shape[2] - vol_y <= 0 or \
                                        volume_in[i].shape[3] - vol_z <= 0:
                                    continue

                                samp = sample_unet(np.expand_dims(volume_in[i], axis=0), label_in[i], side_len=side_len, test=False,
                                                   down_fact=down_fact, side_len_down=side_len_down, position=(vol_x, vol_y, vol_z))

                                if samp is None:
                                    continue
                                samples.append(samp)
                                # Remember if volume is part of batch
                                samples_id.append(i)

                            if len(samples) == 0:
                                continue

                            # Put samples together to get a batch that can be given to network
                            batch_in = many_to_one_collate_fn_sample_unet(
                                samples, down=True)
                            volume, labels, volume_down = batch_in

                            # Device tensors
                            volume_d = volume.to(device)
                            labels_d = labels.to(device)
                            self._oj_model.eval()

                            # Eval network
                            yhat = self._oj_model(
                                volume_d, volume_down.to(device))
                            intersection, union = layers.IOU_unet_val_parts(
                                yhat, labels_d, volume.shape[0], threshold=0.5)

                            # Save counts and calculate iou later for total volumes
                            intersection_batch[samples_id] += intersection
                            union_batch[samples_id] += union

                        gc.collect()

                iou_batch = torch.mean(intersection_batch / union_batch).item()
                print(iou_batch)
                iou_per_batch.append(iou_batch)

        return np.mean(np.array(iou_per_batch))

    def draw(self, test_dataset, side_len, name, down_fact, side_len_down, batch_size=1):
        """Draws one(!) example

        Arguments:
            test_dataset {torch dataset} -- Dataset to test on network
            side_len {long} -- Extracted window size
            npoints {long} -- Number of points to sample
            name {string} -- Name of stored network file
            down_fact {long} -- Downsampled factor for downsampled window
            side_len_down {long} -- Extracted down sampled window size

        Returns:
            Test loss and IOU
        """
        # Get test set
        loader_test = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 pin_memory=False, shuffle=True, collate_fn=many_to_one_collate_fn_list)

        # Restore network for testing
        checkpoint = torch.load("model/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt",
                                map_location=lambda storage, loc: storage)
        print("Loaded:", "model/" + type(self._oj_model).__name__ +
              "_" + str(device) + name + ".pt")
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        # Test with same settings as for training and additionally calculate IOU
        with torch.no_grad():
            for batch in loader_test:
                self._oj_model.eval()
                volume_in, label_in = batch
                shapes = np.max(
                    np.array([volume_in[i].shape for i in range(batch_size)]), axis=0)

                # Consider any extractable window of size side_len**3 to write total object
                with open('outfile_auto_unet.obj', 'w') as f:
                    with open('outfile_auto_unet_org.obj', 'w') as fo:
                        for vol_x in range(0, shapes[1], side_len):
                            print("x", vol_x)
                            for vol_y in range(0, shapes[2], side_len):
                                for vol_z in range(0, shapes[3], side_len):
                                    # Get extracted window and downsampled extracted window, SEE sample in data_interface.py for further information

                                    samples = []
                                    samples_id = []
                                    for i in range(batch_size):
                                        # Sample volume, to get original volume back,
                                        # as well as downsampled volume, coords, and corresponding labels
                                        if volume_in[i].shape[1] - vol_x <= 0 or \
                                                volume_in[i].shape[2] - vol_y <= 0 or \
                                                volume_in[i].shape[3] - vol_z <= 0:
                                            continue

                                        samp = sample_unet(np.expand_dims(volume_in[i], axis=0), label_in[i], side_len=side_len, test=False,
                                                           down_fact=down_fact, side_len_down=side_len_down, position=(vol_x, vol_y, vol_z))

                                        if samp is None:
                                            continue
                                        samples.append(samp)
                                        # Remember if volume is part of batch
                                        samples_id.append(i)
                                    if len(samples) == 0:
                                        continue

                                    # Put samples together to get a batch that can be given to network
                                    batch_in = many_to_one_collate_fn_sample_unet(
                                        samples, down=True)
                                    volume, labels, volume_down = batch_in

                                    # Device tensors
                                    volume_d = volume.to(device)
                                    labels_d = labels.to(device)
                                    self._oj_model.eval()
                                    # Eval network
                                    yhat = self._oj_model(
                                        volume_d, volume_down.to(device))

                                    # Write one example
                                    to_write = (yhat[0] >= 0.5).cpu(
                                    ).numpy().astype(np.short)
                                    to_write_l = labels_d[0].cpu(
                                    ).numpy().astype(np.short)
                                    for i_write in range(to_write.shape[1]):
                                        for j_write in range(to_write.shape[2]):
                                            for k_write in range(to_write.shape[3]):
                                                if to_write[0, i_write, j_write, k_write]:
                                                    f.write("v " + " " + str(i_write + vol_x) + " " + str(j_write + vol_y) + " " + str(k_write + vol_z) +
                                                            " " + "0.0" + " " + "0.0" + " " + "0.5  " + "\n")
                                                if to_write_l[0, i_write, j_write, k_write] == 1.0:
                                                    fo.write("v " + " " + str(i_write + vol_x) + " " + str(j_write + vol_y) + " " + str(k_write + vol_z) +
                                                             " " + "0.0" + " " + "0.5" + " " + "0.0  " + "\n")

                                gc.collect()
                break

    def _val(self, loader_val, losses_val, npoints, side_len, down_fact, side_len_down, iou=False, num_iterations=1, cache_vol=[], cache_labels=[]):
        """Validate wrapped network

        Arguments:
            loader_val {torch dataloader} -- Dataloader to validate on
            side_len {long} -- Extracted window size
            npoints {long} -- Number of points to sample
            losses_val {list} -- Store Loss Val TODO: refactor to not use it
            down_fact {long} -- Downsampled factor for downsampled window
            side_len_down {long} -- Extracted down sampled window size

        Keyword Arguments:
            iou {bool} -- Whether to additionally calculate iou or not (default: {False})

        Returns:
            [float] -- Valdiation Loss
        """
        timer = utils.Timer()
        with torch.no_grad():
            losses_val_batch = []
            losses_iou_batch = []
            # Cache vol and labels to process 8 at a time
            if len(cache_vol) == 0:
                t2 = utils.Timer()
                for j, batch in enumerate(loader_val):
                    cache_vol.append(batch[0])
                    cache_labels.append(batch[1])
                print(t2.stop())
            for iteration in range(num_iterations):
                for j in range(0, len(cache_vol), 8):
                    # After 8 volumes and labels are cache, validate
                    samples = []
                    for i in range(8):
                        if i+j > len(cache_vol):
                            break
                            # Sample volume, to get original volume back,
                            # as well as downsampled volume, coords, and corresponding labels
                        ones = 0
                        while(ones == 0):
                            samp = sample_unet(cache_vol[i + j], cache_labels[i + j], side_len=side_len, test=False,
                                               down_fact=down_fact, side_len_down=side_len_down)
                            ones = 1  # torch.sum(samp[1]).item()
                        if samp is None:
                            continue
                        samples.append(samp)
                    # Put samples together to get a batch that can be given to network
                    batch_in = many_to_one_collate_fn_sample_unet(
                        samples, down=True)
                    volume, labels, volume_down = batch_in
                    # Device tensors
                    volume_d = volume.to(device)
                    labels_d = labels.to(device)
                    self._oj_model.eval()
                    # Eval network
                    yhat = self._oj_model(volume_d, volume_down.to(device))
                    # Computes validation loss
                    loss_val_batch = self._oj_loss(
                        yhat, labels_d).item()
                    # If iou set, calc IOU
                    if iou:
                        loss_iou_batch = layers.IOU_unet_val(
                            yhat, labels_d, volume.shape[0]).item()
                        losses_iou_batch.append(loss_iou_batch)
                    losses_val_batch.append(loss_val_batch)
                    # Urgently needed due to incredibly bad garbage collection by python
                    gc.collect()

            loss_val = np.mean(losses_val_batch)
            loss_iou = np.mean(losses_iou_batch)
            losses_val.append((loss_val))
            print(timer.stop())
            if iou:
                return loss_val, loss_iou
            else:
                return loss_val

    def train(self, train_dataset, val_dataset, side_len, npoints=0, name="", load=False,
              cache_size=500, win_sampled_size=16, down_fact=0, side_len_down=0, cache_type='fifo'):
        """Train wrapped network

        Arguments:
            train_dataset {torch dataset} -- Dataset to train on network
            val_dataset {torch dataset} -- Dataset to validate on network
            side_len {long} -- Extracted window size
            npoints {long} -- Number of points to sample TODO: refactor, not needed
            name {string} -- Name of stored network file

        Keyword Arguments:
            load {bool} -- Whether to restore a given network (default: {False})
            cache_size {long} -- Number of examples to cache (default: {1000})
            win_sampled_size {long} -- Number of samples to form a batch (default: {16})
            down_fact {long} -- Downsampled factor for downsampled window (default: {0})
            side_len_down {long} -- Extracted down sampled window size (default: {0})
        """

        # Function vars ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        loss_best = 0
        losses_train = []
        losses_val = []
        # Create loader for validation and training sets
        loader_train = DataLoader(dataset=train_dataset, batch_size=self._size_batch,
                                  num_workers=8, pin_memory=True, shuffle=True, collate_fn=self._collate_fn)
        loader_val = DataLoader(dataset=val_dataset, batch_size=self._size_batch,
                                num_workers=8, pin_memory=True, shuffle=True, collate_fn=self._collate_fn)
        cache_vol_val = []
        cache_labels_val = []
        # Load model and optimizer
        if load:
            self._oj_model.load_state_dict(torch.load(
                "model/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt"))
            self._oj_optimizer.load_state_dict(torch.load(
                "optimizer/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt"))

        # Auxiliary functions ++++++++++++++++++++++++++++++++++++++++++++++++++
        # Make a training step
        def _step_train(batch, optimize=True):
            # Split batch in components
            volume, labels, volume_low = batch
            self._oj_model.train()
            # Eval network
            yhat = self._oj_model(volume.to(device), volume_low.to(device))
            # Calculate loss
            loss_train = self._oj_loss(yhat, labels.to(device))
            # Calc gradients
            loss_train.backward()
            # Do a optimizer step and set gradients to zero
            if optimize:
                self._oj_optimizer.step()
                self._oj_optimizer.zero_grad()
            labels_on_device = labels.to(device)

            return loss_train.item(), self.mse(yhat, labels_on_device).item(), \
                torch.squeeze(torch.mean(nn.BCELoss(reduction='none')(yhat.reshape(win_sampled_size, -1, 1),
                                                                      labels_on_device.reshape(win_sampled_size, -1, 1)), dim=1))

        # Logic ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for _ in range(self._size_iter):

            losses_train_batch = []
            # Create caches for "async" queue
            cache_vol = [None for x in range(cache_size)]
            cache_labels = [None for x in range(cache_size)]
            # Remember how often an element has been drawn
            draw_counts = np.array([0 for x in range(cache_size)])
            cache_act_size = 0

            for i_, batch in enumerate(loader_train):
                # First fill cache, if filled, replace most often draw example

                if cache_type == 'fifo':
                    where_to_put = i_ % cache_size
                    # Cache volume and labels
                    cache_vol[where_to_put] = batch[0]
                    cache_labels[where_to_put] = batch[1]

                    cache_act_size += 1
                    cache_act_size = min(cache_act_size, cache_size)
                    # Urgently needed due to incredibly bad garbage collection by python
                    gc.collect()

                    # One step of training

                    # Randomly draw win_sampled_size many examples to form a batch
                    indices = np.random.choice(
                        cache_act_size, win_sampled_size)

                elif cache_type == 'counts':
                    # First fill cache, if filled, replace most often draw example
                    where_to_put = np.argmax(
                        draw_counts) if cache_act_size == cache_size else i_
                    # Cache volume and labels
                    cache_vol[where_to_put] = batch[0]
                    cache_labels[where_to_put] = batch[1]
                    # Reset draw count
                    draw_counts[where_to_put] = 0
                    cache_act_size += 1
                    cache_act_size = min(cache_act_size, cache_size)
                    # Urgently needed due to incredibly bad garbage collection by python
                    gc.collect()

                    # One step of training

                    # Randomly draw win_sampled_size many examples to form a batch
                    sum_draw_counts = np.exp(-draw_counts[:cache_act_size])
                    indices = np.random.choice(
                        cache_act_size, win_sampled_size, p=sum_draw_counts/np.sum(sum_draw_counts))

                elif cache_type == 'hardness':
                    # First fill cache, if filled, replace most often draw example
                    where_to_put = np.argmin(
                        draw_counts) if cache_act_size == cache_size else i_
                    # Cache volume and labels
                    cache_vol[where_to_put] = batch[0]
                    cache_labels[where_to_put] = batch[1]
                    # Reset draw count
                    draw_counts[where_to_put] = 10
                    cache_act_size += 1
                    cache_act_size = min(cache_act_size, cache_size)
                    # Urgently needed due to incredibly bad garbage collection by python
                    gc.collect()

                    # One step of training

                    # Randomly draw win_sampled_size many examples to form a batch
                    sum_draw_counts = np.exp(draw_counts[:cache_act_size])
                    indices = np.random.choice(
                        cache_act_size, win_sampled_size, p=sum_draw_counts/np.sum(sum_draw_counts))

                samples = []
                for i in indices:
                    ones = 0
                    while(ones == 0):
                        samp = sample_unet(cache_vol[i], cache_labels[i],
                                           side_len=side_len, test=False, down_fact=down_fact, side_len_down=side_len_down)
                        ones = 1  # torch.sum(samp[1]).item()
                    if samp is None:
                        continue
                    samples.append(samp)
                # Create a batch
                batch_in = many_to_one_collate_fn_sample_unet(
                    samples, down=True)
                # Perform a training step
                loss_train_batch, mse_train_batch, loss_train_samples = _step_train(
                    batch_in, True)

                # Update draw_counts
                for x, i in enumerate(indices):
                    if cache_type == 'hardness':
                        draw_counts[i] = loss_train_samples[x]
                    elif cache_type == 'counts':
                        draw_counts[i] += 1

                # Print training loss
                if i_ % 32 == 32-1:
                    print("Training Loss Batch", _, i_, type(self._oj_loss).__name__, loss_train_batch,
                          "MSE", mse_train_batch, flush=True)

                # Validate
                if i_ % self._size_print_every == self._size_print_every-1:
                    loss_val, iou = self._val(loader_val, losses_val, npoints=npoints, side_len=side_len,
                                              down_fact=down_fact, side_len_down=side_len_down, iou=True,
                                              cache_vol=cache_vol_val, cache_labels=cache_labels_val)
                    print("Validation Loss", loss_val,
                          "IOU", iou, "Best IOU", loss_best)
                    # Save if improved iou
                    if iou > loss_best:
                        loss_best = iou
                        torch.save(self._oj_model.state_dict(
                        ), "model/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt")
                        torch.save(self._oj_optimizer.state_dict(
                        ), "optimizer/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt")
            loss_train = np.mean(losses_train_batch)
            losses_train.append(loss_train)


class Res_Auto_3d_Model_Unet_Parallel(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Unet_Parallel, self).__init__()

        self.model = nn.DataParallel(Res_Auto_3d_Model_Unet())

    def forward(self, volume, volume_low):
        return self.model(volume, volume_low)


class Res_Auto_3d_Model_Unet(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Unet, self).__init__()
        self.activation = nn.SELU
        self.encode = nn.Sequential(layers.Res_Block_Down_3D(1, 32, 3, 1, self.activation(), True),
                                    layers.Res_Block_Down_3D(
                                        32, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(
                                        64, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(
                                        64, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(
                                        64, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(64, 32, 3, 1, self.activation(), False))

        self.encode_low = nn.Sequential(layers.Res_Block_Down_3D(1, 32, 3, 1, self.activation(), True),
                                        layers.Res_Block_Down_3D(
                                            32, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(
                                            64, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(
                                            64, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(
                                            64, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(64, 32, 3, 1, self.activation(), False))

        self.decode = nn.Sequential(
            layers.Res_Block_Up_3D(64, 64, 3, 1, self.activation()),
            layers.Res_Block_Down_3D(
                64, 64, 3, 1, self.activation(), False),
            layers.Res_Block_Down_3D(
                64, 64, 3, 1, self.activation(), False),
            layers.Res_Block_Down_3D(
                64, 64, 3, 1, self.activation(), False),
            layers.Res_Block_Down_3D(
                64, 64, 3, 1, self.activation(), False),
            layers.Res_Block_Down_3D(
                64, 32, 3, 1, self.activation(), False),
            layers.Res_Block_Down_3D(
                32, 1, 3, 1, nn.Sigmoid(), False))

    def forward(self, volume, volume_low):
        # Encode high res extracted window
        encoded_high = self.encode(volume)
        # Do same with downsampled window
        encoded_low = self.encode_low(volume_low)
        # Cat encodings of windows with coordinates and decode
        out = self.decode(
            torch.cat((encoded_high, encoded_low), dim=1))
        # Return occupancy value
        return out
