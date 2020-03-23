from torch import nn
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import itk
from config import device
import utils
from pykdtree.kdtree import KDTree
import random
import argparse


class WeaponDataset(data.Dataset):
    def __init__(self, target_path, length, dim_max=640, offset=0):
        # Path of data and labels
        self.target_path = target_path
        # Size of dataset
        self.length = length
        # Offset, which labels/volumes to skip
        self.offset = offset
        # Split fixed hardcoded, as same weapon is stored in increasing files
        self.index_wrapper = utils.file_permutation()

    def __getitem__(self, index):
       # t = utils.Timer()
        index = index + self.offset
        index = self.index_wrapper[index]
        # Load labels and volumes, skip if not found
        try:
            t = utils.Timer()
            volume_n = np.load(self.target_path + str(index) + ".npy")
            label_n = np.load(self.target_path + str(index) + "_label.npy")
        except Exception as e:
            return self.__getitem__((index+1) % self.__len__())
        return volume_n.astype(np.float), label_n.astype(int).astype(np.float)

    def __len__(self):
        return self.length

    def write_obj(self, index):
        instance = self.__getitem__(index)
        # Write volume to obj, but skip low values
        vol = instance[0]
        maximum = np.max(vol)
        vol = vol/maximum
        vol[vol - 0.05 < 0] = 0
        with open('outfile_org.obj', 'w') as f:
            for i in range(vol.shape[1]):
                for j in range(vol.shape[2]):
                    for k in range(vol.shape[3]):
                        color = vol[0][i][j][k]
                        if color == 0:
                            continue
                        f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) +
                                " " + str(color) + " " + str(0.5) + " " + str(0.5) + "\n")
        # Write corresponding labels to obj
        with open('outfile_labels.obj', 'w') as f:
            label = instance[1]
            for i in range(label.shape[0]):
                f.write("v " + " " + str(label[i][0]) + " " + str(label[i][1]) + " " + str(label[i][2]) +
                        " " + str(0) + " " + str(0) + " " + str(1) + "\n")


class WeaponDatasetGeneratorLowRes():
    def __init__(self, root, target_path, start_index=0, end_index=-1, threshold_min=0, threshold_max=50000,
                 dim_max=640, side_len=8):
        # Only use values that are needed
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        # Downsample factor
        self.side_len = side_len
        # x-axis dim, as it is varying
        self.dim_max = dim_max
        # Folder path of originals
        self.target_path = target_path
        # Data paths
        self.data = []
        # Label paths not joined
        mixed_labels = []

        # Read files
        t1 = utils.Timer()
        for direc, _, files in os.walk(root):
            for file in files:
                if file.endswith(".mha"):
                    if "label" in file:
                        # label annotation file
                        mixed_labels.append(os.path.join(direc, file))
                    else:
                        # regular data file
                        self.data.append(os.path.join(direc, file))

        # Label paths joined
        self.labels = [None] * len(self.data)
        # match data files with label files
        for i, d in enumerate(self.data):
            name, _ = os.path.splitext(d)
            for l in mixed_labels:
                if l.startswith(name):
                    self.labels[i] = l
                    break

        self.data = self.data[start_index:end_index]
        self.labels = self.labels[start_index:end_index]
        print("File paths", t1.stop())

    def generate_data(self):
        for index in range(len(self.data)):
            print(index, "/", len(self.data), flush=True)
            # Load data paths
            data_file = self.data[index]
            label_file = self.labels[index]

            # Check if label in place, otherwise next file
            # Load label file
            labels = itk.imread(label_file)
            try:
                offsets_n = np.flip(np.array(labels.GetMetaDataDictionary()[
                                    "DomainFirst"].split(" "), dtype=np.int), 0)
            except:
                continue

            t2 = utils.Timer()
            # Load volume
            image = itk.imread(data_file)

            # Scale thresholds to 0 - 1
            volume_n = itk.GetArrayFromImage(image)
            volume_n = (volume_n - self.threshold_min).astype(np.float) / \
                float(self.threshold_max - self.threshold_min)
            volume_n = np.expand_dims(volume_n, axis=0)

            #print("Read image", t2.stop())

            t3 = utils.Timer()
            # Enforce x axis dim and avg pool to low resolution
            volume_n = torch.from_numpy(volume_n).to(device)[
                :, 0:self.dim_max, :, :]
            volume_n = nn.functional.pad(
                volume_n, (0, 0, 0, 0, 0, self.dim_max-volume_n.shape[1]))
            volume_pooled_tg = nn.functional.avg_pool3d(
                volume_n, self.side_len, self.side_len)
            #print("Downsampling", t3.stop())
            t4 = utils.Timer()

            #print("Padding", t4.stop())
            np.save(self.target_path + str(index) + ".npy",
                    volume_pooled_tg.cpu().numpy().astype(np.float32))

            # Process labels like volumes
            labels_n = itk.GetArrayFromImage(labels)
            labels_n = np.pad(
                labels_n, ((offsets_n[0], 0), (offsets_n[1], 0), (offsets_n[2], 0)))
            labels_n = torch.unsqueeze(torch.from_numpy(labels_n).to(
                device).float(), dim=0)[:, 0:self.dim_max, :, :]
            labels_n = nn.functional.pad(
                labels_n, (0, volume_n.shape[3]-labels_n.shape[3], 0, volume_n.shape[2]-labels_n.shape[2], 0, self.dim_max-labels_n.shape[1]))
            labels_n = nn.functional.max_pool3d(
                labels_n, self.side_len, self.side_len)
            labels_indices_n = np.argwhere(labels_n.cpu().numpy())

            x_n = np.expand_dims(labels_indices_n[:, 1], axis=1)
            y_n = np.expand_dims(labels_indices_n[:, 2], axis=1)
            z_n = np.expand_dims(labels_indices_n[:, 3], axis=1)
            np.save(self.target_path + str(index) + "_label.npy",
                    np.concatenate((x_n, y_n, z_n), axis=1).astype(np.uint16))


class WeaponDatasetGeneratorHighRes():
    def __init__(self, root, target_path, start_index=0, end_index=-1, threshold_min=0, threshold_max=50000,
                 dim_max=640):
        """
        For Comments see WeaponDatasetGeneratorLowRes
        generate_data changes commented
        """
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.target_path = target_path

        self.data = []
        mixed_labels = []
        t1 = utils.Timer()
        for direc, _, files in os.walk(root):
            for file in files:
                if file.endswith(".mha"):
                    if "label" in file:
                        mixed_labels.append(os.path.join(direc, file))
                    else:
                        self.data.append(os.path.join(direc, file))

        self.labels = [None] * len(self.data)

        for i, d in enumerate(self.data):
            name, _ = os.path.splitext(d)
            for l in mixed_labels:
                if l.startswith(name):
                    self.labels[i] = l
                    break
        self.data = self.data[start_index:end_index]
        self.labels = self.labels[start_index:end_index]
        print("File paths", t1.stop())

    def generate_data(self):
        # range(0, len(self.data)):
        for index in range(len(self.data)):
            print(index, "/", len(self.data), flush=True)
            data_file = self.data[index]
            label_file = self.labels[index]

            # Check if label in place
            labels = itk.imread(label_file)
            try:
                offsets_n = np.flip(np.array(labels.GetMetaDataDictionary()[
                                    "DomainFirst"].split(" "), dtype=np.int), 0)
            except:
                continue

            # Read and scale volume (see WeaponDatasetGeneratorLowRes)
            image = itk.imread(data_file)
            volume_n = itk.GetArrayFromImage(image)
            volume_n = (volume_n - self.threshold_min).astype(np.float) / \
                float(self.threshold_max - self.threshold_min)
            volume_n = np.expand_dims(volume_n, axis=0)

            # Read labels and find offsets from labels within volume
            labels_n = itk.GetArrayFromImage(labels)
            labels_indices_n = np.argwhere(labels_n)
            x_n = np.expand_dims(labels_indices_n[:, 0] + offsets_n[0], axis=1)
            y_n = np.expand_dims(labels_indices_n[:, 1] + offsets_n[1], axis=1)
            z_n = np.expand_dims(labels_indices_n[:, 2] + offsets_n[2], axis=1)
            coords = np.concatenate((x_n, y_n, z_n), axis=1).astype(np.uint16)

            max_x, max_y, max_z = np.max(coords, axis=0)
            min_x, min_y, min_z = np.min(coords, axis=0)

            # Slice volume down to bouding box of labels
            offset = 50
            start_x = int(max(min_x-offset, 0))
            start_y = int(max(min_y-offset, 0))
            start_z = int(max(min_z-offset, 0))

            end_x = max_x+offset
            end_y = max_y+offset
            end_z = max_z+offset

            volume_n = volume_n[:, start_x:end_x, start_y:end_y, start_z:end_z]

            np.save(self.target_path + str(index) +
                    ".npy", volume_n.astype(np.float32))

            # Adapt label coords to new size of volume
            coords[:, 0] -= start_x
            coords[:, 1] -= start_y
            coords[:, 2] -= start_z
            np.save(self.target_path + str(index) + "_label.npy", coords)


def sample(volume_n, label_n, npoints=2**10, side_len=32, sampling='one', down_fact=0, side_len_down=0, share_box=0.5, test=False, position=None):
    """Do sampling of volumes with arbitrary resolution 
    1. Extract window of size side_len**3
    2. Sample npoints points from extracted window, create labels for points
    3. Extracted by down_fact downsampled window from volume around extracted window ( size = (side_len_down * 2)**3)


    Returns:
        Window, Points, Labels, (If test: All coords of actual object), Downsampled Window
    """
    volume_n = np.squeeze(volume_n, axis=0)
    sampling_shapes_tc = [0, volume_n.shape[1],
                          volume_n.shape[2], volume_n.shape[3]]

    # Get starting and end points of extracted window
    ones = 0
    while(ones == 0):
        if side_len != -1:
            x_start = int(np.random.randint(sampling_shapes_tc[1], size=(1)))
            y_start = int(np.random.randint(sampling_shapes_tc[2], size=(1)))
            z_start = int(np.random.randint(sampling_shapes_tc[3], size=(1)))

            if position is not None:
                x_start = position[0]
                y_start = position[1]
                z_start = position[2]

            x_end = x_start + side_len
            y_end = y_start + side_len
            z_end = z_start + side_len
        # Can be used for low res volumes, if total volume should be used
        else:
            x_start = 0
            y_start = 0
            z_start = 0

            x_end = sampling_shapes_tc[1]
            y_end = sampling_shapes_tc[2]
            z_end = sampling_shapes_tc[3]
            # Only use labels of extracted window
        mask = (label_n[:, 0] >= x_start) & (label_n[:, 0] < x_end) & \
            (label_n[:, 1] >= y_start) & (label_n[:, 1] < y_end) & \
            (label_n[:, 2] >= z_start) & (label_n[:, 2] < z_end)
        ones = np.sum(mask) + (1 if random.random() < 0.1 else 0)
        if position is not None:
            break
    # Downsampled volume +++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Check if downsampeld window should be created as well
    if down_fact > 0 and side_len_down > 0:
        # Start in the middle of the extracted window
        x_start_down = int(x_start + side_len/2)
        y_start_down = int(y_start + side_len/2)
        z_start_down = int(z_start + side_len/2)

        # Pad if extraced window is to small
        pad_x_front = max(down_fact * side_len_down - x_start_down, 0)
        pad_y_front = max(down_fact * side_len_down - y_start_down, 0)
        pad_z_front = max(down_fact * side_len_down - z_start_down, 0)

        pad_x_back = max(down_fact * side_len_down -
                         (sampling_shapes_tc[1] - x_start_down), 0)
        pad_y_back = max(down_fact * side_len_down -
                         (sampling_shapes_tc[2] - y_start_down), 0)
        pad_z_back = max(down_fact * side_len_down -
                         (sampling_shapes_tc[3] - z_start_down), 0)

        volume_down = np.pad(volume_n, ((0, 0), (pad_x_front, pad_x_back),
                                        (pad_y_front, pad_y_back), (pad_z_front, pad_z_back)))
        volume_down = torch.from_numpy(volume_down[:, x_start_down + pad_x_front - side_len_down * down_fact:x_start_down + pad_x_front + side_len_down * down_fact,
                                                   y_start_down + pad_y_front - side_len_down * down_fact:y_start_down + pad_y_front + side_len_down * down_fact,
                                                   z_start_down + pad_z_front - side_len_down * down_fact:z_start_down + pad_z_front + side_len_down * down_fact]).to(device)

        # Do downsampling
        volume_down = nn.functional.avg_pool3d(
            volume_down, down_fact, down_fact)

    # Downsampled volume +++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Adapat labels to extracted window coordinates
    label_n = label_n[mask]
    label_n[:, 0] -= x_start
    label_n[:, 1] -= y_start
    label_n[:, 2] -= z_start

    # Do the actual window extraction
    volume_n = volume_n[:, x_start:x_end, y_start:y_end, z_start:z_end]
    sampling_shapes_tc = [0, volume_n.shape[1],
                          volume_n.shape[2], volume_n.shape[3]]

    # Sample coordinates from extracted window and get corresponding label
    if label_n.shape[0] > 0:

        # Get coords with label 1
        # Can fail if not enough label 1s in extracted window
        try:
            coords_one = label_n[np.random.choice(
                label_n.shape[0], int(npoints * share_box), replace=False), :]
        except:
            coords_one = label_n[np.random.choice(
                label_n.shape[0], int(npoints * share_box), replace=True), :]

        # Get random coords from window and check label with kd tree
        x_n = np.random.randint(sampling_shapes_tc[1], size=(
            int(npoints * (1-share_box)), 1))
        y_n = np.random.randint(sampling_shapes_tc[2], size=(
            int(npoints * (1-share_box)), 1))
        z_n = np.random.randint(sampling_shapes_tc[3], size=(
            int(npoints * (1-share_box)), 1))
        coords_zero = np.concatenate((x_n, y_n, z_n), axis=1).astype(np.float)
        kd_tree = KDTree(label_n, leafsize=16)
        dist, _ = kd_tree.query(coords_zero, k=1)
        labels_zero = np.expand_dims(dist == 0, axis=1).astype(float)

        coords = np.concatenate((coords_one, coords_zero), axis=0)
        labels = np.concatenate(
            (np.ones((coords_one.shape[0], 1)), labels_zero), axis=0)

    else:
        # Get random coords from window and check label with kd tree
        x_n = np.random.randint(sampling_shapes_tc[1], size=(int(npoints), 1))
        y_n = np.random.randint(sampling_shapes_tc[2], size=(int(npoints), 1))
        z_n = np.random.randint(sampling_shapes_tc[3], size=(int(npoints), 1))
        coords_zero = np.concatenate((x_n, y_n, z_n), axis=1).astype(np.float)
        labels_zero = np.zeros((coords_zero.shape[0], 1))

        coords = coords_zero
        labels = labels_zero

    # If Volume smaller then side_len pad
    if side_len != -1:
        volume_n = np.pad(volume_n, ((0, 0), (0, max(side_len - sampling_shapes_tc[1], 0)),
                                     (0, max(side_len - sampling_shapes_tc[2], 0)), (0, max(side_len - sampling_shapes_tc[3], 0))))

    # Return depending on required output
    if test:
        out = [torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(),
               torch.from_numpy(labels).float(), torch.from_numpy(label_n.astype(int)).float()]
    else:
        out = [torch.from_numpy(volume_n).float(), torch.from_numpy(
            coords).float(), torch.from_numpy(labels).float()]

    if down_fact > 0 and side_len_down > 0:
        out = out + [volume_down.float()]

    return out


def sample_low(volume_n, label_n, npoints=2**10, share_box=0.5, test=False, side_len=8):
    volume_n = np.squeeze(volume_n, axis=0)
    sampling_shapes_tc = [0, volume_n.shape[1] * side_len,
                          volume_n.shape[2] * side_len, volume_n.shape[3] * side_len]
    coords_one = label_n[np.random.choice(label_n.shape[0], int(
        npoints * share_box), replace=False), :]

    # Mixed Coords
    x_n = np.random.randint(sampling_shapes_tc[1], size=(
        int(npoints * (1-share_box)), 1))
    y_n = np.random.randint(sampling_shapes_tc[2], size=(
        int(npoints * (1-share_box)), 1))
    z_n = np.random.randint(sampling_shapes_tc[3], size=(
        int(npoints * (1-share_box)), 1))
    coords_zero = np.concatenate((x_n, y_n, z_n), axis=1)
    kd_tree = KDTree(label_n, leafsize=16)
    dist, _ = kd_tree.query(coords_zero, k=1)
    labels_zero = np.expand_dims(dist == 0, axis=1).astype(float)

    coords = np.concatenate((coords_one, coords_zero), axis=0)
    labels = np.concatenate(
        (np.ones((coords_one.shape[0], 1)), labels_zero), axis=0)

    #print("Access time", t.stop())
    if test:
        return torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(), torch.from_numpy(labels).float(), torch.from_numpy(label_n.astype(int)).float()
    else:
        return torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(), torch.from_numpy(labels).float()


def sample_unet(volume_n, label_n, side_len=32, down_fact=0, side_len_down=0, test=False, position=None):
    """Do sampling of volumes with arbitrary resolution 
    1. Extract window of size side_len**3
    2. Sample npoints points from extracted window, create labels for points
    3. Extracted by down_fact downsampled window from volume around extracted window ( size = (side_len_down * 2)**3)


    Returns:
        Window, Points, Labels, (If test: All coords of actual object), Downsampled Window
    """
    volume_n = np.squeeze(volume_n, axis=0)
    sampling_shapes_tc = [0, volume_n.shape[1],
                          volume_n.shape[2], volume_n.shape[3]]

    # Get starting and end points of extracted window
    ones = 0
    while(ones == 0):
        if side_len != -1:
            x_start = int(np.random.randint(sampling_shapes_tc[1], size=(1)))
            y_start = int(np.random.randint(sampling_shapes_tc[2], size=(1)))
            z_start = int(np.random.randint(sampling_shapes_tc[3], size=(1)))

            if position is not None:
                x_start = position[0]
                y_start = position[1]
                z_start = position[2]

            x_end = x_start + side_len
            y_end = y_start + side_len
            z_end = z_start + side_len
        # Can be used for low res volumes, if total volume should be used
        else:
            x_start = 0
            y_start = 0
            z_start = 0

            x_end = sampling_shapes_tc[1]
            y_end = sampling_shapes_tc[2]
            z_end = sampling_shapes_tc[3]
            # Only use labels of extracted window
        mask = (label_n[:, 0] >= x_start) & (label_n[:, 0] < x_end) & \
            (label_n[:, 1] >= y_start) & (label_n[:, 1] < y_end) & \
            (label_n[:, 2] >= z_start) & (label_n[:, 2] < z_end)
        ones = np.sum(mask) + (1 if random.random() < 0.1 else 0)
        if position is not None:
            break
    # Downsampled volume +++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Check if downsampeld window should be created as well
    if down_fact > 0 and side_len_down > 0:
        # Start in the middle of the extracted window
        x_start_down = int(x_start + side_len/2)
        y_start_down = int(y_start + side_len/2)
        z_start_down = int(z_start + side_len/2)

        # Pad if extraced window is to small
        pad_x_front = max(down_fact * side_len_down - x_start_down, 0)
        pad_y_front = max(down_fact * side_len_down - y_start_down, 0)
        pad_z_front = max(down_fact * side_len_down - z_start_down, 0)

        pad_x_back = max(down_fact * side_len_down -
                         (sampling_shapes_tc[1] - x_start_down), 0)
        pad_y_back = max(down_fact * side_len_down -
                         (sampling_shapes_tc[2] - y_start_down), 0)
        pad_z_back = max(down_fact * side_len_down -
                         (sampling_shapes_tc[3] - z_start_down), 0)

        volume_down = np.pad(volume_n, ((0, 0), (pad_x_front, pad_x_back),
                                        (pad_y_front, pad_y_back), (pad_z_front, pad_z_back)))
        volume_down = torch.from_numpy(volume_down[:, x_start_down + pad_x_front - side_len_down * down_fact:x_start_down + pad_x_front + side_len_down * down_fact,
                                                   y_start_down + pad_y_front - side_len_down * down_fact:y_start_down + pad_y_front + side_len_down * down_fact,
                                                   z_start_down + pad_z_front - side_len_down * down_fact:z_start_down + pad_z_front + side_len_down * down_fact]).to(device)

        # Do downsampling
        volume_down = nn.functional.avg_pool3d(
            volume_down, down_fact, down_fact)

    # Downsampled volume +++++++++++++++++++++++++++++++++++++++++++++++++++++

    # Adapat labels to extracted window coordinates
    label_n = label_n[mask]
    label_n[:, 0] -= x_start
    label_n[:, 1] -= y_start
    label_n[:, 2] -= z_start
    label_n = label_n.astype(int)

    # Do the actual window extraction
    volume_n = volume_n[:, x_start:x_end, y_start:y_end, z_start:z_end]
    sampling_shapes_tc = [0, volume_n.shape[1],
                          volume_n.shape[2], volume_n.shape[3]]

    # If Volume smaller then side_len pad
    if side_len != -1:
        volume_n = np.pad(volume_n, ((0, 0), (0, max(side_len - sampling_shapes_tc[1], 0)),
                                     (0, max(side_len - sampling_shapes_tc[2], 0)), (0, max(side_len - sampling_shapes_tc[3], 0))))

    # Return depending on required output
    label_volume = np.squeeze(np.zeros_like(volume_n))
    label_volume[label_n[:, 0], label_n[:, 1], label_n[:, 2]] = 1
    label_volume = np.expand_dims(label_volume, axis=0)
    out = [torch.from_numpy(volume_n).float(),
           torch.from_numpy(label_volume).float()]

    if down_fact > 0 and side_len_down > 0:
        out = out + [volume_down.float()]

    return out
# Custom collate_fn's that create batches of samples


def many_to_one_collate_fn(batch):
    volumes = np.stack([elm[0] for elm in batch], axis=0)
    labels = np.stack([elm[1] for elm in batch], axis=0).reshape(-1, 3)
    return volumes, labels


def many_to_one_collate_fn_list(batch):
    volumes = [elm[0] for elm in batch]
    labels = [elm[1] for elm in batch]
    return volumes, labels


def many_to_one_collate_fn_sample(batch, down=False):
    volumes = torch.stack([elm[0] for elm in batch], dim=0)
    coords = torch.stack([elm[1] for elm in batch], dim=0).view(-1, 3)
    labels = torch.stack([elm[2] for elm in batch], dim=0).view(-1, 1)
    if down:
        low_volumes = torch.stack([elm[3] for elm in batch], dim=0)
        return volumes, coords, labels, low_volumes
    else:
        return volumes, coords, labels


def many_to_one_collate_fn_sample_unet(batch, down=False):
    volumes = torch.stack([elm[0] for elm in batch], dim=0)
    labels = torch.stack([elm[1] for elm in batch], dim=0)
    if down:
        low_volumes = torch.stack([elm[2] for elm in batch], dim=0)
        return volumes, labels, low_volumes
    else:
        return volumes, labels


def many_to_one_collate_fn_sample_test(batch):
    volumes = torch.stack([elm[0] for elm in batch], dim=0)
    coords = torch.stack([elm[1] for elm in batch], dim=0).view(-1, 3)
    labels = torch.stack([elm[2] for elm in batch], dim=0).view(-1, 1)
    actual = torch.stack([elm[3] for elm in batch], dim=0).view(-1, 3)
    return volumes, coords, labels, actual


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create and Check High and Low Resolution Weapon Datasets')
    # Create/Check high or low resolution datasets
    parser.add_argument('-r', required='True', choices=['low', 'high'])
    # Create or check a dataset
    parser.add_argument('-a', required='True', choices=['generate', 'check'])
    global args
    args = parser.parse_args()

    if args.r == "high":
        if args.a == "check":
            dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                                    length=2000)
            dataset.write_obj(1500)
            _, coords, labels, vol, _ = sample(np.expand_dims(dataset[1500][0], axis=0), dataset[1500][1], npoints=2**10,
                                               side_len=32, test=True, down_fact=2, side_len_down=16)

            with open('outfile_labels_drawn.obj', 'w') as f:
                coords = coords[torch.squeeze(labels == 1)]
                for i in range(coords.shape[0]):
                    f.write("v " + " " + str(coords[i][0].item()) + " " + str(coords[i][1].item()) + " " + str(coords[i][2].item()) +
                            " " + str(0) + " " + str(0) + " " + str(1) + "\n")

            # Write corresponding labels to obj
            with open('outfile_labels.obj', 'w') as f:
                label = vol
                for i in range(label.shape[0]):
                    f.write("v " + " " + str(label[i][0].item()) + " " + str(label[i][1].item()) + " " + str(label[i][2].item()) +
                            " " + str(0) + " " + str(0) + " " + str(1) + "\n")

        else:
            dataset_gen = WeaponDatasetGeneratorHighRes(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                                                        target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1_full/")
            dataset_gen.generate_data()
    else:
        if args.a == "check":
            dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
                                    length=2)
            dataset.write_obj(0)
        else:
            dataset_gen = WeaponDatasetGeneratorLowRes(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                                                       target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
                                                       side_len=8)
            dataset_gen.generate_data()
