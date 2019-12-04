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

class WeaponDataset(data.Dataset):
    def __init__(self, target_path, length, dim_max=640, npoints=2**10, side_len=32, sampling='one', offset=0, test=False):
        self.npoints = npoints
        self.side_len = side_len
        self.dim_max = int(dim_max / side_len)
        self.sampling = sampling
        self.target_path=target_path
        self.length = length
        self.offset = offset
        self.test=test
        self.index_wrapper = utils.file_permutation()


    def __getitem__(self, index):
       # t = utils.Timer()
        index = index + self.offset
        index = self.index_wrapper[index]
        try:
            volume_n = np.load(self.target_path +str(index) + ".npy")
            label_n = np.load(self.target_path +str(index) + "_label.npy")
        except:
            return self.__getitem__((index+1) % self.__len__())
        sampling_shapes_tc = [0, volume_n.shape[1] * self.side_len, volume_n.shape[2] * self.side_len, volume_n.shape[2] * self.side_len]
        share_box=0.5

        if self.sampling == 'default':
            # Mixed Coords
            x_n = np.random.randint(sampling_shapes_tc[1], size=(int(self.npoints),1))
            y_n = np.random.randint(sampling_shapes_tc[2], size=(int(self.npoints),1))
            z_n = np.random.randint(sampling_shapes_tc[3], size=(int(self.npoints),1))
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
            x_n = np.random.randint(sampling_shapes_tc[1], size=(int(self.npoints * (1-share_box)),1))
            y_n = np.random.randint(sampling_shapes_tc[2], size=(int(self.npoints * (1-share_box)),1))
            z_n = np.random.randint(sampling_shapes_tc[3], size=(int(self.npoints * (1-share_box)),1))
            coords_zero = np.concatenate((x_n, y_n, z_n), axis=1)

            coords = np.concatenate((coords_one, coords_zero), axis=0)
            labels = np.concatenate((np.ones((coords_one.shape[0], 1)), np.zeros((coords_zero.shape[0], 1))), axis=0)

        elif self.sampling == 'one':
            # Coords with one as label
            coords_one = label_n[np.random.choice(label_n.shape[0], int(self.npoints * share_box), replace=False), :]

            # Mixed Coords
            x_n = np.random.randint(sampling_shapes_tc[1], size=(int(self.npoints * (1-share_box)),1))
            y_n = np.random.randint(sampling_shapes_tc[2], size=(int(self.npoints * (1-share_box)),1))
            z_n = np.random.randint(sampling_shapes_tc[3], size=(int(self.npoints * (1-share_box)),1))
            coords_zero = np.concatenate((x_n, y_n, z_n), axis=1)
            kd_tree = KDTree(label_n, leafsize=16)
            dist, _ = kd_tree.query(coords_zero, k=1)
            labels_zero = np.expand_dims(dist == 0, axis=1).astype(float)

            coords = np.concatenate((coords_one, coords_zero), axis=0)
            labels = np.concatenate((np.ones((coords_one.shape[0], 1)), labels_zero), axis=0)

        else:
            raise NotImplementedError
        #print("Access time", t.stop())
        if self.test:
            return torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(), torch.from_numpy(labels).float(), torch.from_numpy(label_n.astype(int)).float()
        else:
            return torch.from_numpy(volume_n).float(), torch.from_numpy(coords).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return self.length

    def get_side_len(self):
        return self.side_len

    def write_obj(self, index):
        vol = self.__getitem__(index)[0].cpu().numpy()
        maximum = np.max(vol)
        with open('outfile_org.obj','w') as f:
            for i in range(vol.shape[1]):
                for j in range(vol.shape[2]):
                    for k in range(vol.shape[3]):
                        color = utils.get_colour(vol[0][i][j][k], maximum)
                        f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) + 
                                " " + str(color[0]) + " " + str(color[1]) + " " + str(color[2]) + "\n")

                


class WeaponDatasetGenerator():
    def __init__(self, root, target_path,start_index=0, end_index=-1, threshold_min=0, threshold_max=30000, 
                dim_max=640, side_len=16):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.side_len = side_len
        self.dim_max = int(dim_max / side_len)
        self.target_path=target_path

        self.data = []
        mixed_labels = []
        t1 = utils.Timer()
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
            print(index, "/",len(self.data))
            data_file = self.data[index]
            label_file = self.labels[index]

            # Check if label in place
            labels = itk.imread(label_file)
            try:
                offsets_n = np.flip(np.array(labels.GetMetaDataDictionary()["DomainFirst"].split(" "), dtype=np.int), 0)
            except:
                continue
            # load data using itk
            t2 = utils.Timer()
            # First take care of volume
            image = itk.imread(data_file)

            volume_n = itk.GetArrayFromImage(image)
            volume_n = (volume_n - self.threshold_min).astype(np.float) / float(self.threshold_max - self.threshold_min)
            volume_n = np.expand_dims(volume_n, axis = 0) 

            print("Read image", t2.stop())

            t3 = utils.Timer()
            volume_pooled_tg = nn.functional.max_pool3d(torch.from_numpy(volume_n).to(device), self.side_len, self.side_len)
            print("Downsampling", t3.stop())
            t4 = utils.Timer()
            volume_pooled_tg = volume_pooled_tg[:,0:self.dim_max,:,:]
            volume_pooled_tg = nn.functional.pad(volume_pooled_tg, 
                                                            (0,0,0,0,0,self.dim_max-volume_pooled_tg.shape[1]))
            print("Padding", t4.stop())
            np.save(self.target_path +str(index) + ".npy", volume_pooled_tg.cpu().numpy().astype(np.float32))

            # Take care of labels and store coords
            labels_n = itk.GetArrayFromImage(labels)

            labels_indices_n = np.argwhere(labels_n)
            x_n = np.expand_dims(labels_indices_n[:, 0] + offsets_n[0], axis=1)
            y_n = np.expand_dims(labels_indices_n[:, 1] + offsets_n[1], axis=1)
            z_n = np.expand_dims(labels_indices_n[:, 2] + offsets_n[2], axis=1)
            np.save(self.target_path +str(index) + "_label.npy", np.concatenate((x_n,y_n,z_n), axis=1).astype(np.uint16))


        

def many_to_one_collate_fn(batch):
    volumes = torch.stack([elm[0] for elm in batch], dim=0)
    coords = torch.stack([elm[1] for elm in batch], dim=0).view(-1,3)
    labels = torch.stack([elm[2] for elm in batch], dim=0).view(-1,1)
    return volumes, coords, labels

def many_to_one_collate_fn_test(batch):
    volumes = torch.stack([elm[0] for elm in batch], dim=0)
    coords = torch.stack([elm[1] for elm in batch], dim=0).view(-1,3)
    labels = torch.stack([elm[2] for elm in batch], dim=0).view(-1,1)
    actual = torch.stack([elm[3] for elm in batch], dim=0).view(-1,3    )
    return volumes, coords, labels, actual



if __name__ == '__main__':
    dataset_gen = WeaponDatasetGenerator(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
                        target_path="../../../../fastdata/Smiths_LKA_Weapons/len_1/",
                        side_len=1)

    #dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_32/train/",
    #                    npoints=2**14,
    #                    length=1,
    #                    side_len=32)

    dataset_gen.generate_data()
    #dataset.write_obj(0)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=many_to_one_collate_fn, num_workers=8)
    #print(len(dataset))
    #for full, coords, labels in dataloader:
     #   print(full.shape, coords.shape, labels.shape)

                                        
