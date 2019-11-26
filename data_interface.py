from torch import nn
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import itk
from config import device


class WeaponDataset(data.Dataset):
    def __init__(self, root, start_index=0, end_index=-1, threshold_min=0, threshold_max=30000, 
                dim_max=640, npoints=2**10, side_len=32, sampling='default'):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.npoints = npoints
        self.side_len = side_len
        self.dim_max = int(dim_max / side_len)
        self.sampling = sampling

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

        self.data = self.data[start_index:end_index]
        self.labels = self.labels[start_index:end_index]


    def __getitem__(self, index):
        data_file = self.data[index]
        label_file = self.labels[index]
        num_labels = 1

        # load data using itk
        image = itk.imread(data_file)
        labels = itk.imread(label_file)
        try:
            offsets_n = np.flip(np.array(labels.GetMetaDataDictionary()["DomainFirst"].split(" "), dtype=np.int), 0)
        except:
            return self.__getitem__((index+1) % self.__len__())

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
        volume_with_labels_pooled_tc = volume_with_labels_pooled_tc[:,:,0:self.dim_max,:,:]
        volume_with_labels_pooled_tc = nn.functional.pad(volume_with_labels_pooled_tc, 
                                                        (0,0,0,0,0,self.dim_max-volume_with_labels_pooled_tc.shape[2]))
        
        # TODO: thresholding 
        #data = (volume >= 0.0) & (volume <= 1.0)

        sampling_shapes_tc = volume_with_labels_tc.shape

        if self.sampling == 'default':
            x_tc = torch.LongTensor(self.npoints).random_(0, sampling_shapes_tc[2])
            y_tc = torch.LongTensor(self.npoints).random_(0, sampling_shapes_tc[3])
            z_tc = torch.LongTensor(self.npoints).random_(0, sampling_shapes_tc[4])
        
        elif self.sampling == 'boxed':
            share_box = 1.
            x_in_tc = torch.LongTensor(int(self.npoints * share_box)).random_(int(offsets_n[0]), int(offsets_n[0] + labels_dims[0]))
            y_in_tc = torch.LongTensor(int(self.npoints * share_box)).random_(int(offsets_n[1]), int(offsets_n[1] + labels_dims[1]))
            z_in_tc = torch.LongTensor(int(self.npoints * share_box)).random_(int(offsets_n[2]), int(offsets_n[2] + labels_dims[2]))

            x_out_tc = torch.LongTensor(int(self.npoints * (1-share_box))).random_(0, sampling_shapes_tc[2])
            y_out_tc = torch.LongTensor(int(self.npoints * (1-share_box))).random_(0, sampling_shapes_tc[3])
            z_out_tc = torch.LongTensor(int(self.npoints * (1-share_box))).random_(0, sampling_shapes_tc[4])

            x_tc = torch.cat((x_in_tc, x_out_tc), dim=0)
            y_tc = torch.cat((y_in_tc, y_out_tc), dim=0)
            z_tc = torch.cat((z_in_tc, z_out_tc), dim=0)

        else:
            raise NotImplementedError


        coords_tc = torch.cat((torch.unsqueeze(x_tc,dim=1), torch.unsqueeze(y_tc,dim=1), torch.unsqueeze(z_tc,dim=1)), axis = 1)
        labels_tc = torch.unsqueeze(volume_with_labels_tc[0,1,x_tc,y_tc,z_tc], dim=1)

        return  volume_with_labels_pooled_tc[:,0,:,:,:].float(), coords_tc.float(), labels_tc.float()


    def __len__(self):
        return len(self.data)

    def set_side_len(self, side_len):
        self.side_len=side_len

    def get_side_len(self):
        return self.side_len

def many_to_one_collate_fn(batch):
    volumes = torch.stack([elm[0] for elm in batch], dim=0)
    coords = torch.stack([elm[1] for elm in batch], dim=0).view(-1,3)
    labels = torch.stack([elm[2] for elm in batch], dim=0).view(-1,1)
    
    return volumes, coords, labels




if __name__ == '__main__':
    print("Generating WeaponDataset...")
    train = True 
    dataset = WeaponDataset(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                        threshold_min=1700,
                        threshold_max=2700,
                        npoints=2**14,
                        side_len=32)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=many_to_one_collate_fn)
    print(len(dataset))
    for full, coords, labels in dataloader:
        print(full.shape, coords.shape, labels.shape)

                                        
