import torch
from torch.utils import data
import numpy as np
from pykdtree.kdtree import KDTree

import Misc


class WeaponDataset(data.Dataset):
    def __init__(self, target_path: str, length: int, dim_max: int = 640, npoints: int = 2 ** 10, side_len: int = 32,
                 sampling: str = 'one', offset: int = 0, test: bool = False):
        self.npoints = npoints
        self.side_len = side_len
        self.dim_max = int(dim_max / side_len)
        self.sampling = sampling
        self.target_path = target_path
        self.length = length
        self.offset = offset
        self.test = test
        self.index_wrapper = Misc.FilePermutation()

    def __getitem__(self, index):
        # Calc index
        index = index + self.offset
        index = self.index_wrapper[index]
        # Load volume and label
        volume_n = np.load(self.target_path + str(index) + ".npy")
        label_n = np.load(self.target_path + str(index) + "_label.npy")

        sampling_shapes_tc = [0, volume_n.shape[1] * self.side_len, volume_n.shape[2] * self.side_len,
                              volume_n.shape[3] * self.side_len]
        share_box = 0.5

        if self.sampling == 'default':
            # Mixed Coords
            x_n = np.random.randint(sampling_shapes_tc[1], size=(int(self.npoints), 1))
            y_n = np.random.randint(sampling_shapes_tc[2], size=(int(self.npoints), 1))
            z_n = np.random.randint(sampling_shapes_tc[3], size=(int(self.npoints), 1))
            coords_zero = np.concatenate((x_n, y_n, z_n), axis=1)
            kd_tree = KDTree(label_n, leafsize=16)
            dist, _ = kd_tree.query(coords_zero, k=1)
            labels_zero = np.expand_dims(dist == 0, axis=1).astype(float)

            coords = coords_zero
            labels = labels_zero

        elif self.sampling == 'one_fast':
            # Coords with one as label
            coords_one = label_n[np.random.choice(label_n.shape[0], int(self.npoints * share_box), replace=False), :]

            # Mixed Coords
            x_n = np.random.randint(sampling_shapes_tc[1], size=(int(self.npoints * (1 - share_box)), 1))
            y_n = np.random.randint(sampling_shapes_tc[2], size=(int(self.npoints * (1 - share_box)), 1))
            z_n = np.random.randint(sampling_shapes_tc[3], size=(int(self.npoints * (1 - share_box)), 1))
            coords_zero = np.concatenate((x_n, y_n, z_n), axis=1)

            coords = np.concatenate((coords_one, coords_zero), axis=0)
            labels = np.concatenate((np.ones((coords_one.shape[0], 1)), np.zeros((coords_zero.shape[0], 1))), axis=0)

        elif self.sampling == 'one':
            # Coords with one as label
            coords_one = label_n[np.random.choice(label_n.shape[0], int(self.npoints * share_box), replace=False), :]

            # Mixed Coords
            x_n = np.random.randint(sampling_shapes_tc[1], size=(int(self.npoints * (1 - share_box)), 1))
            y_n = np.random.randint(sampling_shapes_tc[2], size=(int(self.npoints * (1 - share_box)), 1))
            z_n = np.random.randint(sampling_shapes_tc[3], size=(int(self.npoints * (1 - share_box)), 1))
            coords_zero = np.concatenate((x_n, y_n, z_n), axis=1)
            kd_tree = KDTree(label_n, leafsize=16)
            dist, _ = kd_tree.query(coords_zero, k=1)
            labels_zero = np.expand_dims(dist == 0, axis=1).astype(float)

            coords = np.concatenate((coords_one, coords_zero), axis=0)
            labels = np.concatenate((np.ones((coords_one.shape[0], 1)), labels_zero), axis=0)

        else:
            raise NotImplementedError
        # print("Access time", t.stop())
        if self.test:
            return torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(), torch.from_numpy(
                labels).float(), torch.from_numpy(label_n.astype(int)).float()
        else:
            return torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(), torch.from_numpy(
                labels).float()

    def __len__(self) -> int:
        return self.length

    def get_side_len(self):
        return self.side_len

    def write_obj(self, index):
        vol = self.__getitem__(index)[0].cpu().numpy()
        maximum = np.max(vol)
        vol = vol / maximum
        vol[vol - 0.15 < 0] = 0

        with open('outfile_org.obj', 'w') as f:
            for i in range(vol.shape[1]):
                for j in range(vol.shape[2]):
                    for k in range(vol.shape[3]):
                        color = vol[0][i][j][k]
                        if color == 0:
                            continue
                        f.write("v " + " " + str(i * self.side_len) + " " + str(j * self.side_len) + " " + str(
                            k * self.side_len) +
                                " " + str(color) + " " + str(0.5) + " " + str(0.5) + "\n")
        with open('outfile_labels.obj', 'w') as f:
            label = self.__getitem__(index)[3].cpu().numpy()
            for i in range(label.shape[0]):
                f.write("v " + " " + str(label[i][0]) + " " + str(label[i][1]) + " " + str(label[i][2]) +
                        " " + str(0) + " " + str(0) + " " + str(1) + "\n")
