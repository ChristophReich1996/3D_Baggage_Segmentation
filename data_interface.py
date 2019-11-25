import numpy as np
import torch
from torch import nn

import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import itk
import time


class WeaponDataset(data.Dataset):
    def __init__(self, root, threshold_min=0, threshold_max=30000, npoints=2**10, side_len=32):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.npoints = npoints
        self.side_len = side_len

        self.data = []
        mixed_labels = []

        
        for direc, _, files in os.walk(root):
            for file in files:
                if file.endswith(".mha"):
                    if "label" in file:
                        # label annotation file
                        mixed_labels.append(os.path.join(direc,file))
                    else:
                        # regular data file
                        self.data.append(os.path.join(direc,file))

        self.labels = [None] * len(self.data)
        # match data files with label files
        for i, d in enumerate(self.data):
            name, extension = os.path.splitext(d)
            for l in mixed_labels:
                if l.startswith(name):
                    self.labels[i] = l
                    break

    def __getitem__(self, index):
        data_file = self.data[index]
        label_file = self.labels[index]
        num_labels = 1

        # load data using itk
        image = itk.imread(data_file)
        labels = itk.imread(label_file)

        offsets_n = np.flip(np.array(labels.GetMetaDataDictionary()["DomainFirst"].split(" "), dtype=np.int), 0)

        volume_n = itk.GetArrayFromImage(image)
        volume_n = (volume_n - self.threshold_min).astype(np.float) / float(self.threshold_max - self.threshold_min)
        volume_n = np.expand_dims(volume_n, axis = 0) 
        labels_n = itk.GetArrayFromImage(labels)
        labels_dims = labels_n.shape

        labels_indices_n = np.argwhere(labels_n)
        x_n = labels_indices_n[:, 0] + offsets_n[0]
        y_n = labels_indices_n[:, 1] + offsets_n[1]
        z_n = labels_indices_n[:, 2] + offsets_n[2]
        labels_expanded_n = np.zeros(volume_n.shape)
        labels_expanded_n[0, x_n, y_n, z_n] = 1

        volume_with_labels_tc = torch.from_numpy(np.expand_dims(np.concatenate((volume_n, labels_expanded_n), axis = 0), axis=0))
        volume_with_labels_pooled_tc = nn.functional.max_pool3d(volume_with_labels_tc, self.side_len, self.side_len)
        
        # TODO: thresholding 
        #data = (volume >= 0.0) & (volume <= 1.0)

        sampling_shapes_tc = volume_with_labels_pooled_tc.shape

        x_tc = torch.LongTensor(self.npoints).random_(0, sampling_shapes_tc[2])
        y_tc = torch.LongTensor(self.npoints).random_(0, sampling_shapes_tc[3])
        z_tc = torch.LongTensor(self.npoints).random_(0, sampling_shapes_tc[4])

        coords_tc = torch.cat((torch.unsqueeze(x_tc,dim=1), torch.unsqueeze(y_tc,dim=1), torch.unsqueeze(z_tc,dim=1)), axis = 1)
        labels_tc = torch.unsqueeze(volume_with_labels_pooled_tc[0,1,x_tc,y_tc,z_tc], dim=1)

        return  volume_with_labels_pooled_tc[:,0,:,:,:], coords_tc, labels_tc


    def __len__(self):
        return len(self.data)

    def set_side_len(self, side_len):
        self.side_len=side_len

    def get_side_len(self):
        return self.side_len


if __name__ == '__main__':
    print("Generating WeaponDataset...")
    train = True
    dataset = WeaponDataset(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                        threshold_min=1700,
                        threshold_max=2700,
                        npoints=50000,
                        side_len=64)

    elm = dataset[0]
    print(elm[0].shape, elm[1].shape, elm[2].shape)
