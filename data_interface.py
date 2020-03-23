from torch import nn
import torch.utils.data as data
import os
import os.path
import torch
import numpy as np
import itk
import time 
from pykdtree.kdtree import KDTree

device = 'cuda'

class Timer():
    def __init__(self):
        self.start = time.process_time()

    def stop(self):
        return time.process_time() - self.start

class WeaponDatasetGenerator():
    def __init__(self, root, target_path,start_index=0, end_index=-1, threshold_min=0, threshold_max=50000, 
                dim_max=640, side_len=16):
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.side_len = side_len
        self.dim_max = int(dim_max / side_len)
        self.target_path=target_path

        self.data = []
        mixed_labels = []
        t1 = Timer()
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
            t2 = Timer()
            # First take care of volume
            image = itk.imread(data_file)

            volume_n = itk.GetArrayFromImage(image)
            volume_n = (volume_n - self.threshold_min).astype(np.float) / float(self.threshold_max - self.threshold_min)
            volume_n = np.expand_dims(volume_n, axis = 0) 

            print("Read image", t2.stop())

            t3 = Timer()
            volume_pooled_tg = nn.functional.avg_pool3d(torch.from_numpy(volume_n).to(device), self.side_len, self.side_len)
            print("Downsampling", t3.stop())
            t4 = Timer()
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

# if __name__ == '__main__':
#     """
#     dataset_gen = WeaponDatasetGenerator(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
#                         target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
#                         side_len=8)
    
#     dataset = WeaponDataset(target_path="../../../../fastdata/Smiths_LKA_Weapons/len_8/",
#                         npoints=2**14,
#                         length=2000,
#                         side_len=8,
#                         test=True)
#     """
#     dataset_gen = BoundedDatasetGenerator(root="../../../projects_students/Smiths_LKA_Weapons/ctix-lka-20190503/",
#                         target_path="../../../../fastdata/Smiths_LKA_Weapons/bounded/",
#                         side_len=8)
#     dataset_gen.generate_data("Res_Auto_3d_Model_Occu_Parallel_cuda.pt")


                                        
