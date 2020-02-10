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

from data_interface import sample, many_to_one_collate_fn_sample, many_to_one_collate_fn_sample_test, many_to_one_collate_fn_list
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

    def test(self, test_dataset, side_len, npoints, name, down_fact, side_len_down, batch_size=8):
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
                                 pin_memory=False, shuffle=True, collate_fn=many_to_one_collate_fn_list)

        # Restore network for testing
        checkpoint = torch.load("model/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt",
                                map_location=lambda storage, loc: storage)
        print("Loaded:", "model/" + type(self._oj_model).__name__ +
              "_" + str(device) + name + ".pt")
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        losses_test_batch = []
        losses_iou_batch = []
        # Test with same settings as for training and additionally calculate IOU
        with torch.no_grad():
            for batch_counter, batch in enumerate(loader_test):
                self._oj_model.eval()
                volume_in, label_in = batch
                shapes = np.max(
                    np.array([volume_in[i].shape for i in range(batch_size)]), axis=0)
                intersection = torch.zeros((batch_size, 1))
                union = torch.zeros((batch_size, 1))
                loss_iou_batch = 0

                # Consider any extractable window of size side_len**3 to write total object
                for vol_x in range(0, shapes[1], side_len):
                    print("x", vol_x, "Batch", batch_counter, flush=True)
                    print(loss_iou_batch)
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

                                samp = sample(np.expand_dims(volume_in[i], axis=0), label_in[i], npoints=npoints, side_len=side_len, test=False,
                                              down_fact=down_fact, side_len_down=side_len_down, share_box=0.0,
                                              position=(vol_x, vol_y, vol_z))

                                if samp is None:
                                    continue
                                samples.append(samp)
                                samples_id.append(i)
                            if len(samples) == 0:
                                continue

                            # Put samples together to get a batch that can be given to network
                            batch_in = many_to_one_collate_fn_sample(
                                samples, down=True)
                            volume, coords, labels, volume_down = batch_in

                            self._oj_model.eval()
                            # Eval network
                            yhat = self._oj_model(volume.to(device), coords.to(
                                device), volume_down.to(device))
                            # Computes validation loss
                            loss_val_batch = self._oj_loss(
                                yhat, labels.to(device)).item()
                            # If iou set, calcl IOU
                            intersection_patch, union_patch = layers.IOU_parts(
                                coords, yhat, labels, volume.shape[0], threshold=0.7)
                            intersection[samples_id] += intersection_patch
                            union[samples_id] += union_patch
                            losses_test_batch.append(loss_val_batch)
                            # Urgently needed due to incredibly bad garbage collection by python
                            gc.collect()
                            loss_iou_batch = torch.mean(intersection / (union + 0.00001))
                losses_iou_batch.append(loss_iou_batch)

        loss_test = np.mean(losses_test_batch)
        loss_iou = np.mean(losses_iou_batch)

        return loss_test, loss_iou

    # TODO: ONLY WORKING WITH BATCH_SIZE == 1 CURRENTLY
    def draw(self, test_dataset, side_len, name="", down_fact=0, side_len_down=0):
        """Draws one object with custom algorithm (not that of the ONet paper)

        Arguments:
            test_dataset {torch dataset} -- Dataset to draw one object fom
            side_len {long} -- !!!! SKIP SIZE FOR INITIAL REQUEST !!! TODO: refactor to down_sample

        Keyword Arguments:
            name {string} -- Name of stored network file
            down_fact {long} -- Downsampled factor for downsampled window
            side_len_down {long} -- Extracted down sampled window size
        """
        # Get test set
        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch,
                                 pin_memory=False, shuffle=True, collate_fn=self._collate_fn)

        # Restore network for testing
        checkpoint = torch.load("model/" + type(self._oj_model).__name__ + "_" + str(
            device) + name + ".pt", map_location=lambda storage, loc: storage)
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()

        # "historically grown"
        down_sample = side_len
        win_size = 32

        # Create tensor to write object
        to_write = np.empty((0, 3))
        with torch.no_grad():
            for batch in loader_test:
                self._oj_model.eval()
                volume_in, label_in = batch

                # Consider any extractable window of size side_len**3 to write total object
                for vol_x in range(0, volume_in[0].shape[1], win_size):
                    for vol_y in range(0, volume_in[0].shape[2], win_size):
                        for vol_z in range(0, volume_in[0].shape[3], win_size):
                            print("x", vol_x, "y", vol_y, "z", vol_z)
                            # Get extracted window and downsampled extracted window, SEE sample in data_interface.py for further information
                            side_len = down_sample
                            volume = volume_in[:, :, vol_x:vol_x+win_size,
                                               vol_y:vol_y+win_size, vol_z:vol_z+win_size]

                            if down_fact > 0 and side_len_down > 0:
                                x_start_down = int(vol_x + side_len/2)
                                y_start_down = int(vol_y + side_len/2)
                                z_start_down = int(vol_z + side_len/2)

                                pad_x_front = max(
                                    down_fact * side_len_down - x_start_down, 0)
                                pad_y_front = max(
                                    down_fact * side_len_down - y_start_down, 0)
                                pad_z_front = max(
                                    down_fact * side_len_down - z_start_down, 0)

                                pad_x_back = max(
                                    down_fact * side_len_down - (volume_in[0].shape[1] - x_start_down), 0)
                                pad_y_back = max(
                                    down_fact * side_len_down - (volume_in[0].shape[2] - y_start_down), 0)
                                pad_z_back = max(
                                    down_fact * side_len_down - (volume_in[0].shape[3] - z_start_down), 0)

                                volume_down = np.pad(volume_in[0], ((
                                    0, 0), (pad_x_front, pad_x_back), (pad_y_front, pad_y_back), (pad_z_front, pad_z_back)))
                                volume_down = torch.from_numpy(volume_down[:, x_start_down + pad_x_front - side_len_down * down_fact:x_start_down + pad_x_front + side_len_down * down_fact,
                                                                           y_start_down + pad_y_front - side_len_down * down_fact:y_start_down + pad_y_front + side_len_down * down_fact,
                                                                           z_start_down + pad_z_front - side_len_down * down_fact:z_start_down + pad_z_front + side_len_down * down_fact]).to(device)
                                volume_down = nn.functional.avg_pool3d(
                                    volume_down, down_fact, down_fact)
                                volume_down = torch.unsqueeze(
                                    volume_down, dim=0).float()

                            volume = np.pad(volume, ((0, 0), (0, 0),
                                                     (0, max(win_size - volume[0].shape[1], 0)), (0, max(win_size - volume[0].shape[2], 0)), (0, max(win_size - volume[0].shape[3], 0))))
                            volume = torch.from_numpy(volume).float()

                            # Create initial query +++++++++
                            # Sample in regular grid over extracted window with down_sample as distance bewtween points in grid
                            x = torch.arange(0, volume.shape[2]/down_sample)
                            y = torch.arange(0, volume.shape[3]/down_sample)
                            z = torch.arange(0, volume.shape[4]/down_sample)

                            x, y, z = torch.meshgrid(x, y, z)
                            query = torch.cat((torch.unsqueeze(x.reshape(-1), dim=1), torch.unsqueeze(
                                y.reshape(-1), dim=1),  torch.unsqueeze(z.reshape(-1), dim=1)), dim=1)
                            query = query.float().to(device) * side_len
                            active = 1

                            # Generate basic offsets
                            neutral = torch.FloatTensor([[0, 0, 0]]).to(device)
                            above = torch.FloatTensor([[0, 0, 1]]).to(device)
                            left = torch.FloatTensor([[0, 1, 0]]).to(device)
                            behind = torch.FloatTensor([[1, 0, 0]]).to(device)
                            # TODO catch negative values, no error, in best case network handles this
                            offsets = [left, behind, left+behind, neutral +
                                       above, above+left, above+behind, above+left+behind]

                            # Loop to refine grid ++++++++++
                            while active > 0 and side_len >= 1:
                                # Get scaled offsets to check for neighbours
                                offsets_s = [
                                    neutral] + [torch.relu(offset * side_len - 1) for offset in offsets]
                                # Expand query to neighbours
                                coords = [
                                    query + offset for offset in offsets_s]
                                # Request all coordinates (query)
                                acts = [self._oj_model.inference(volume.to(device), coord.to(
                                    device), volume_down.to(device)) for coord in coords]
                                # Check for hits
                                masks = [act == 1 for act in acts]
                                # Check of all or non of neighbors is activated
                                sum_masks = functools.reduce(
                                    lambda a, b: a & b, masks)
                                sum_masks_inv = functools.reduce(
                                    lambda a, b: a.logical_not() & b.logical_not(), masks)

                                # First one: All points within neighbours are hits as well
                                # Second one: None of the points within neighbours are hits
                                next_query_mask = torch.squeeze(
                                    (sum_masks | sum_masks_inv).logical_not())
                                mask_full = torch.squeeze(sum_masks)
                                query_to_write = query[mask_full]
                                query_to_write[:, 0] += vol_x - win_size/2
                                query_to_write[:, 1] += vol_y - win_size/2
                                query_to_write[:, 2] += vol_z - win_size/2

                                # Write Firste one
                                for i in range(side_len):
                                    for j in range(side_len):
                                        for k in range(side_len):
                                            to_write = np.append(
                                                to_write, (query_to_write + left * i + above * j + behind * k).cpu().numpy(), axis=0)

                                # Query undecided neighbourshoods on higher resolution
                                # To do that, split one dice into four samller ones
                                side_len = int(side_len/2)
                                query_next = query[next_query_mask]
                                query = torch.empty((0, 3)).to(device)

                                for offset in [neutral] + offsets:
                                    query = torch.cat(
                                        (query, query_next + offset * side_len), dim=0)

                                active = query.shape[0]

                # Write actual object, predicted object and total volume

                to_write = to_write.astype(np.short)
                to_write_labels = label_in.astype(np.short)

                with open('outfile_auto.obj', 'w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                                " " + "0.0" + " " + "0.0" + " " + "0.5  " + "\n")

                vol = volume_in[0]
                maximum = np.max(vol)
                vol = vol/maximum
                vol[vol - 0.17 < 0] = 0

                with open('outfile_org.obj', 'w') as f:
                    for i in range(vol.shape[1]):
                        for j in range(vol.shape[2]):
                            for k in range(vol.shape[3]):
                                color = vol[0][i][j][k]
                                if color == 0:
                                    continue
                                f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) +
                                        " " + str(color.item()) + " " + str(0.5) + " " + str(0.5) + "\n")

                with open('outfile_auto_labels.obj', 'w') as f:
                    for line in to_write_labels:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                                " " + "0.0" + " " + "1.0" + " " + "0.0" + "\n")

    def draw_fast(self, test_dataset, side_len, name="", down_fact=0, side_len_down=0, npoints=2**13):
        """Draws one object with custom algorithm (not that of the ONet paper)

        Arguments:
            test_dataset {torch dataset} -- Dataset to draw one object fom
            side_len {long} -- !!!! SKIP SIZE FOR INITIAL REQUEST !!! TODO: refactor to down_sample

        Keyword Arguments:
            name {string} -- Name of stored network file
            down_fact {long} -- Downsampled factor for downsampled window
            side_len_down {long} -- Extracted down sampled window size
        """
        # Get test set
        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch,
                                 pin_memory=False, shuffle=True, collate_fn=self._collate_fn)

        # Restore network for testing
        checkpoint = torch.load("model/" + type(self._oj_model).__name__ + "_" + str(
            device) + name + ".pt", map_location=lambda storage, loc: storage)
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()

        # "historically grown"
        down_sample = 1
        win_size = 16

        # Create tensor to write object
        to_write = np.empty((0, 3))
        with torch.no_grad():
            for batch in loader_test:
                self._oj_model.eval()
                volume_in, label_in = batch

                # Consider any extractable window of size side_len**3 to write total object
                for vol_x in range(0, volume_in[0].shape[1], win_size):
                    for vol_y in range(0, volume_in[0].shape[2], win_size):
                        for vol_z in range(0, volume_in[0].shape[3], win_size):
                            print("x", vol_x, "y", vol_y, "z", vol_z)
                            # Get extracted window and downsampled extracted window, SEE sample in data_interface.py for further information
                            side_len = down_sample
                            volume = volume_in[:, :, vol_x:vol_x+win_size,
                                               vol_y:vol_y+win_size, vol_z:vol_z+win_size]

                            if down_fact > 0 and side_len_down > 0:
                                x_start_down = int(vol_x + side_len/2)
                                y_start_down = int(vol_y + side_len/2)
                                z_start_down = int(vol_z + side_len/2)

                                pad_x_front = max(
                                    down_fact * side_len_down - x_start_down, 0)
                                pad_y_front = max(
                                    down_fact * side_len_down - y_start_down, 0)
                                pad_z_front = max(
                                    down_fact * side_len_down - z_start_down, 0)

                                pad_x_back = max(
                                    down_fact * side_len_down - (volume_in[0].shape[1] - x_start_down), 0)
                                pad_y_back = max(
                                    down_fact * side_len_down - (volume_in[0].shape[2] - y_start_down), 0)
                                pad_z_back = max(
                                    down_fact * side_len_down - (volume_in[0].shape[3] - z_start_down), 0)

                                volume_down = np.pad(volume_in[0], ((
                                    0, 0), (pad_x_front, pad_x_back), (pad_y_front, pad_y_back), (pad_z_front, pad_z_back)))
                                volume_down = torch.from_numpy(volume_down[:, x_start_down + pad_x_front - side_len_down * down_fact:x_start_down + pad_x_front + side_len_down * down_fact,
                                                                           y_start_down + pad_y_front - side_len_down * down_fact:y_start_down + pad_y_front + side_len_down * down_fact,
                                                                           z_start_down + pad_z_front - side_len_down * down_fact:z_start_down + pad_z_front + side_len_down * down_fact]).to(device)
                                volume_down = nn.functional.avg_pool3d(
                                    volume_down, down_fact, down_fact)
                                volume_down = torch.unsqueeze(
                                    volume_down, dim=0).float()

                            volume = np.pad(volume, ((0, 0), (0, 0),
                                                     (0, max(win_size - volume[0].shape[1], 0)), (0, max(win_size - volume[0].shape[2], 0)), (0, max(win_size - volume[0].shape[3], 0))))
                            volume = torch.from_numpy(volume).float()

                            x_n = np.random.randint(
                                volume.shape[2], size=(int(npoints), 1))
                            y_n = np.random.randint(
                                volume.shape[3], size=(int(npoints), 1))
                            z_n = np.random.randint(
                                volume.shape[4], size=(int(npoints), 1))
                            coords_sampled = np.concatenate(
                                (x_n, y_n, z_n), axis=1).astype(np.float)
                            coords_sampled_torch = torch.from_numpy(
                                coords_sampled).float()

                            yhat = (self._oj_model(volume.to(device), coords_sampled_torch.to(
                                device), volume_down.to(device)) > 0.7).float()

                            coords_hit = coords_sampled_torch[torch.squeeze(
                                yhat == 1)].cpu().numpy()
                            coords_hit[:, 0] += vol_x - win_size/2
                            coords_hit[:, 1] += vol_y - win_size/2
                            coords_hit[:, 2] += vol_z - win_size/2
                            to_write = np.append(
                                to_write, coords_hit, axis=0)

                # Write actual object, predicted object and total volume

                to_write = to_write.astype(np.short)
                to_write_labels = label_in.astype(np.short)

                with open('outfile_auto.obj', 'w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                                " " + "0.0" + " " + "0.0" + " " + "0.5  " + "\n")

                vol = volume_in[0]
                maximum = np.max(vol)
                vol = vol/maximum
                vol[vol - 0.01 < 0] = 0

                with open('outfile_org.obj', 'w') as f:
                    for i in range(vol.shape[1]):
                        for j in range(vol.shape[2]):
                            for k in range(vol.shape[3]):
                                color = vol[0][i][j][k]
                                if color == 0:
                                    continue
                                f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) +
                                        " " + str(color.item()) + " " + str(0.5) + " " + str(0.5) + "\n")

                with open('outfile_auto_labels.obj', 'w') as f:
                    for line in to_write_labels:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) +
                                " " + "0.0" + " " + "1.0" + " " + "0.0" + "\n")
    # Validate, ignore grads

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
                            samp = sample(cache_vol[i + j], cache_labels[i + j], npoints=npoints, side_len=side_len, test=False,
                                          down_fact=down_fact, side_len_down=side_len_down, share_box=0.0)
                            ones = torch.sum(samp[2]).item()
                        if samp is None:
                            continue
                        samples.append(samp)
                    # Put samples together to get a batch that can be given to network
                    batch_in = many_to_one_collate_fn_sample(
                        samples, down=True)
                    volume, coords, labels, volume_down = batch_in

                    self._oj_model.eval()
                    # Eval network
                    yhat = self._oj_model(volume.to(device), coords.to(
                        device), volume_down.to(device))
                    # Computes validation loss
                    loss_val_batch = self._oj_loss(
                        yhat, labels.to(device)).item()
                    # If iou set, calcl IOU
                    if iou:
                        loss_iou_batch = layers.IOU(
                            coords, yhat, labels, volume.shape[0], threshold=0.5)
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

    def train(self, train_dataset, val_dataset, side_len, npoints, name, load=False,
              cache_size=500, win_sampled_size=16, down_fact=0, side_len_down=0, cache_type='fifo'):
        """Train wrapped network

        Arguments:
            train_dataset {torch dataset} -- Dataset to train on network
            val_dataset {torch dataset} -- Dataset to validate on network
            side_len {long} -- Extracted window size
            npoints {long} -- Number of points to sample
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
            volume, coords, labels, volume_low = batch
            self._oj_model.train()
            # Eval network
            yhat = self._oj_model(volume.to(device), coords.to(
                device), volume_low.to(device))
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
                        samp = sample(cache_vol[i], cache_labels[i], npoints=npoints,
                                      side_len=side_len, test=False, down_fact=down_fact, side_len_down=side_len_down)
                        ones = torch.sum(samp[2]).item()
                    if samp is None:
                        continue
                    samples.append(samp)
                # Create a batch
                batch_in = many_to_one_collate_fn_sample(samples, down=True)
                # Perform a training step
                loss_train_batch, mse_train_batch, loss_train_samples = _step_train(
                    batch_in, True)

                # Update draw_counts
                for x, i in enumerate(indices):
                    if cache_type == 'hardness':
                        draw_counts[i] = loss_train_samples[x]
                    elif cache_type == 'counts':
                        draw_counts[i] += 1

                # Print training los
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

                    # Save network if improved iou
                    if iou > loss_best:
                        loss_best = iou
                        torch.save(self._oj_model.state_dict(
                        ), "model/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt")
                        torch.save(self._oj_optimizer.state_dict(
                        ), "optimizer/" + type(self._oj_model).__name__ + "_" + str(device) + name + ".pt")
            loss_train = np.mean(losses_train_batch)
            losses_train.append(loss_train)


class Res_Auto_3d_Model_Occu_Parallel(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Occu_Parallel, self).__init__()
        self.model = nn.DataParallel(Res_Auto_3d_Model_Occu())

    def forward(self, volume, coords, volume_low):
        return self.model(volume, coords, volume_low)


class Res_Auto_3d_Model_Occu(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Occu, self).__init__()
        self.activation = nn.SELU
        self.encode = nn.Sequential(layers.Res_Block_Down_3D(1, 32, 3, 1, self.activation(), True),
                                    layers.Res_Block_Down_3D(
                                        32, 64, 3, 1, self.activation(), True),
                                    layers.Res_Block_Down_3D(
                                        64, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(
                                        64, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(
                                        64, 64, 3, 1, self.activation(), False),
                                    layers.Res_Block_Down_3D(64, 32, 3, 1, self.activation(), False))

        self.encode_low = nn.Sequential(layers.Res_Block_Down_3D(1, 32, 3, 1, self.activation(), True),
                                        layers.Res_Block_Down_3D(
                                            32, 64, 3, 1, self.activation(), True),
                                        layers.Res_Block_Down_3D(
                                            64, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(
                                            64, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(
                                            64, 64, 3, 1, self.activation(), False),
                                        layers.Res_Block_Down_3D(64, 32, 3, 1, self.activation(), False))

        self.decode = nn.Sequential(layers.Res_Block_Up_Flat(4096 + 3, 512, nn.SELU()),  # 6912
                                    layers.Res_Block_Up_Flat(
                                        512, 512, nn.SELU()),
                                    # layers.Res_Block_Up_Flat(
                                    #    512, 512, nn.SELU()),
                                    # layers.Res_Block_Up_Flat(
                                    #    512, 512, nn.SELU()),
                                    # layers.Res_Block_Up_Flat(
                                    #    512, 256, nn.SELU()),
                                    layers.Res_Block_Up_Flat(512, 1, nn.Sigmoid()))

    def forward(self, volume, coords, volume_low):
        # Encode high res extracted window
        encoded_high = self.encode(volume)
        encoded_high = encoded_high.view(encoded_high.shape[0], -1)
        # Repeat encoding for all coordinates
        encoded_high = torch.repeat_interleave(
            encoded_high, int(coords.shape[0]/volume.shape[0]), dim=0)
        # Do same with downsampled window
        encoded_low = self.encode_low(volume_low)
        encoded_low = encoded_low.view(encoded_low.shape[0], -1)
        encoded_low = torch.repeat_interleave(
            encoded_low, int(coords.shape[0]/volume.shape[0]), dim=0)
        # Cat encodings of windows with coordinates and decode
        out = self.decode(
            torch.cat((encoded_high, encoded_low, coords), dim=1))
        # Return occupancy value
        return out
