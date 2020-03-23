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

from data_interface import sample_low, many_to_one_collate_fn_sample, many_to_one_collate_fn_sample_test
import utils
import layers
from config import device



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Base Network Class
class Network_Generator():
    def __init__(self, rate_learn, size_iter, size_print_every, oj_loss, optimizer, oj_model, collate_fn=None):

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._rate_learning = rate_learn
        self._size_batch = 1
        self._size_iter = size_iter
        self._size_print_every = size_print_every

        # (Function) Objects +++++++++++++++++++++++++++++++++++++++++++++++++++
        self._oj_model = oj_model
        self._oj_optimizer = optimizer(self._oj_model.parameters(), lr = self._rate_learning)
        self._oj_loss = oj_loss
        self.mse = nn.MSELoss(reduction='mean')
        self._collate_fn = collate_fn

    # TODO only working with batchsize 1 currently (actual labels)
    def test(self, test_dataset, side_len, npoints):
        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        checkpoint = torch.load("model/"+ type(self._oj_model).__name__ + "_" + str(device) + ".pt", map_location=lambda storage, loc: storage)
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        losses_test_batch = []
        return self._val(loader_test, losses_test_batch, npoints=npoints, side_len=side_len)

        """
        with torch.no_grad():
            
            precision_test_batch = []
            recall_test_batch = []


                
                kd_tree = KDTree(actual.cpu().numpy(), leafsize=16)
                dist, _ = kd_tree.query(locs.cpu().numpy(), k=1)
                union = np.sum(dist == 0)
                precision = union/locs.shape[0]
                recall = union/actual.shape[0]
                loss_test_batch = self._oj_loss(yhat, labels.to(device)).item()
                losses_test_batch.append(loss_test_batch)
                precision_test_batch.append(precision)
                recall_test_batch.append(recall)
         
        return np.mean(np.array(losses_test_batch)), np.mean(np.array(precision_test_batch)), np.mean(np.array(recall_test_batch))
        """


        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        checkpoint = torch.load("model/"+ type(self._oj_model).__name__ + "_" + str(device) + ".pt", map_location=lambda storage, loc: storage)
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        side_len_clean = 1
        down_sample=side_len

        with torch.no_grad():
            for batch in loader_test:
                self._oj_model.eval()
                # Makes predictions ++++++++++++
                volume, coords_in, labels, actual = many_to_one_collate_fn_sample_test([sample(batch[0], batch[1], npoints=2**10, side_len=32, test=True)])
                yhat = self._oj_model(volume.to(device), coords_in.to(device))
                print(self.mse(yhat, labels.to(device)).item())
                # Create initial query +++++++++
                x = torch.arange(0, volume.shape[2]/down_sample)
                y = torch.arange(0, volume.shape[3]/down_sample)
                z = torch.arange(0, volume.shape[4]/down_sample)
            
                x,y,z = torch.meshgrid(x,y,z)
                query = torch.cat((torch.unsqueeze(x.reshape(-1), dim=1), torch.unsqueeze(y.reshape(-1), dim=1),  torch.unsqueeze(z.reshape(-1), dim=1)), dim=1)
                query = query.float().to(device) * side_len
                active = 1

                to_write = np.empty((0, 3))
                # Generate basic offsets
                neutral = torch.FloatTensor([[0,0,0]]).to(device)
                above = torch.FloatTensor([[0,0,1]]).to(device)
                left = torch.FloatTensor([[0,1,0]]).to(device)
                behind = torch.FloatTensor([[1,0,0]]).to(device)
                # TODO catch negative values, no error, in best case network handles this
                offsets = [left, behind, left+behind, neutral+above, above+left, above+behind, above+left+behind]

                # Loop to refine grid ++++++++++
                while active > 0 and side_len >= 1:
                    print(active)
                    # Get scaled offsets to check for neighbours
                    offsets_s = [neutral] + [torch.relu(offset * side_len - 1) for offset in offsets]
                    coords = [query + offset for offset in offsets_s]
                    acts = [self._oj_model.inference(volume.to(device), coord.to(device)) for coord in coords]
                    print(torch.sum(torch.stack(acts)))
                    masks = [act == 1 for act in acts]
                    sum_masks = functools.reduce(lambda a,b : a & b, masks)
                    sum_masks_inv = functools.reduce(lambda a,b : a.logical_not() & b.logical_not(), masks)
                    next_query_mask = torch.squeeze((sum_masks | sum_masks_inv).logical_not())
                    mask_full = torch.squeeze(sum_masks)
                    query_to_write = query[mask_full]

                    for i in range(side_len):
                        for j in range(side_len):
                            for k in range(side_len):
                                to_write = np.append(to_write, (query_to_write + left * i + above * j + behind * k).cpu().numpy(), axis=0)

                    side_len = int(side_len/2)
                    query_next = query[next_query_mask]
                    query = torch.empty((0,3)).to(device)
                    
                    for offset in [neutral] + offsets:
                        query = torch.cat((query, query_next + offset * side_len), dim=0)

                    active = query.shape[0]

                to_write = to_write.astype(np.short)
                print(to_write.shape, volume.shape)
                to_write_labels = actual.cpu().numpy().astype(np.short)
                print(to_write_labels.shape)
                with open('outfile_auto.obj','w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                            " " + "0.0" + " " + "0.0" + " " + "0.5  " + "\n")
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")           
        
                vol = volume[0]
                maximum = torch.max(vol)
                vol = vol/maximum
                vol[vol - 0.15 < 0] = 0

                with open('outfile_org.obj','w') as f:
                    for i in range(vol.shape[1]):
                        for j in range(vol.shape[2]):
                            for k in range(vol.shape[3]):
                                color = vol[0][i][j][k]
                                if color == 0:
                                    continue
                                f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) + 
                                        " " + str(color.item()) + " " + str(0.5) + " " + str(0.5) + "\n")
                            #Corners of volume
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")  

                with open('outfile_auto_labels.obj','w') as f:
                    for line in to_write_labels:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                        " " + "0.0" + " " + "1.0" + " " + "0.0" + "\n")
                            #Corners of volume
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")                          
        return 0
    
    def draw_low(self, test_dataset, side_len, name=""):
        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        checkpoint = torch.load("model/"+ type(self._oj_model).__name__ + "_" + str(device) + name + ".pt", map_location=lambda storage, loc: storage)
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        side_len_clean = 1
        down_sample=side_len
        to_write = np.empty((0, 3))
        with torch.no_grad():
            for batch in loader_test:
                self._oj_model.eval()
                # Makes predictions ++++++++++++
                volume, label_in = batch
                volume = torch.from_numpy(volume).float()
                # Create initial query +++++++++
                x = torch.arange(0, volume.shape[2]/down_sample)
                y = torch.arange(0, volume.shape[3]/down_sample)
                z = torch.arange(0, volume.shape[4]/down_sample)
    
                x,y,z = torch.meshgrid(x,y,z)
                query = torch.cat((torch.unsqueeze(x.reshape(-1), dim=1), torch.unsqueeze(y.reshape(-1), dim=1),  torch.unsqueeze(z.reshape(-1), dim=1)), dim=1)
                query = query.float().to(device) * side_len
                active = 1

        
                # Generate basic offsets
                neutral = torch.FloatTensor([[0,0,0]]).to(device)
                above = torch.FloatTensor([[0,0,1]]).to(device)
                left = torch.FloatTensor([[0,1,0]]).to(device)
                behind = torch.FloatTensor([[1,0,0]]).to(device)
                # TODO catch negative values, no error, in best case network handles this
                offsets = [left, behind, left+behind, neutral+above, above+left, above+behind, above+left+behind]

                # Loop to refine grid ++++++++++
                while active > 0 and side_len >= 1:
                    print(active)
                    # Get scaled offsets to check for neighbours
                    offsets_s = [neutral] + [torch.relu(offset * side_len - 1) for offset in offsets]
                    coords = [query + offset for offset in offsets_s]
                    acts = [self._oj_model.inference(volume.to(device), coord.to(device)) for coord in coords]
                    print(torch.sum(torch.stack(acts)))
                    masks = [act == 1 for act in acts]
                    sum_masks = functools.reduce(lambda a,b : a & b, masks)
                    sum_masks_inv = functools.reduce(lambda a,b : a.logical_not() & b.logical_not(), masks)
                    next_query_mask = torch.squeeze((sum_masks | sum_masks_inv).logical_not())
                    mask_full = torch.squeeze(sum_masks)
                    query_to_write = query[mask_full]
  
                    for i in range(side_len):
                        for j in range(side_len):
                            for k in range(side_len):
                                to_write = np.append(to_write, (query_to_write + left * i + above * j + behind * k).cpu().numpy(), axis=0)

                    side_len = int(side_len/2)
                    query_next = query[next_query_mask]
                    query = torch.empty((0,3)).to(device)
                    
                    for offset in [neutral] + offsets:
                        query = torch.cat((query, query_next + offset * side_len), dim=0)

                    active = query.shape[0]

                to_write = to_write.astype(np.short)
                print(to_write.shape, volume.shape)
                to_write_labels = label_in.astype(np.short)
                #print(to_write_labels.shape)
                with open('outfile_auto.obj','w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                            " " + "0.0" + " " + "0.0" + " " + "0.5  " + "\n")
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")           
        
                vol = volume[0]
                maximum = torch.max(vol)
                vol = vol/maximum
                vol[vol - 0.05 < 0] = 0

                with open('outfile_org.obj','w') as f:
                    for i in range(vol.shape[1]):
                        for j in range(vol.shape[2]):
                            for k in range(vol.shape[3]):
                                color = vol[0][i][j][k]
                                if color == 0:
                                    continue
                                f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) + 
                                        " " + str(color.item()) + " " + str(0.5) + " " + str(0.5) + "\n")
                            #Corners of volume
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")  
                
                with open('outfile_auto_labels.obj','w') as f:
                    for line in to_write_labels:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                        " " + "0.0" + " " + "1.0" + " " + "0.0" + "\n")
                            #Corners of volume
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")                          
                
    def draw(self, test_dataset, side_len, name=""):
        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        checkpoint = torch.load("model/"+ type(self._oj_model).__name__ + "_" + str(device) + name + ".pt", map_location=lambda storage, loc: storage)
        self._oj_model.load_state_dict(checkpoint)
        del checkpoint  # dereference seems crucial
        torch.cuda.empty_cache()
        side_len_clean = 1
        down_sample=side_len
        win_size = 32
        to_write = np.empty((0, 3))
        with torch.no_grad():
            for batch in loader_test:
                self._oj_model.eval()
                # Makes predictions ++++++++++++
                volume_in, label_in = batch
                for vol_x in range(0, volume_in[0].shape[1], win_size):
                    for vol_y in range(0, volume_in[0].shape[2], win_size):
                        for vol_z in range(0, volume_in[0].shape[3], win_size):
                            side_len=down_sample
                            volume = volume_in[:,:,vol_x:vol_x+win_size, vol_y:vol_y+win_size, vol_z:vol_z+win_size]
                            volume = np.pad(volume, ((0,0),(0,0),
                                    (0, max(win_size - volume[0].shape[1],0)), (0, max(win_size - volume[0].shape[2],0)), (0, max(win_size - volume[0].shape[3],0))))
                            volume = torch.from_numpy(volume).float()
                            # Create initial query +++++++++
                            x = torch.arange(0, volume.shape[2]/down_sample)
                            y = torch.arange(0, volume.shape[3]/down_sample)
                            z = torch.arange(0, volume.shape[4]/down_sample)
                
                            x,y,z = torch.meshgrid(x,y,z)
                            query = torch.cat((torch.unsqueeze(x.reshape(-1), dim=1), torch.unsqueeze(y.reshape(-1), dim=1),  torch.unsqueeze(z.reshape(-1), dim=1)), dim=1)
                            query = query.float().to(device) * side_len
                            active = 1

                    
                            # Generate basic offsets
                            neutral = torch.FloatTensor([[0,0,0]]).to(device)
                            above = torch.FloatTensor([[0,0,1]]).to(device)
                            left = torch.FloatTensor([[0,1,0]]).to(device)
                            behind = torch.FloatTensor([[1,0,0]]).to(device)
                            # TODO catch negative values, no error, in best case network handles this
                            offsets = [left, behind, left+behind, neutral+above, above+left, above+behind, above+left+behind]

                            # Loop to refine grid ++++++++++
                            while active > 0 and side_len >= 1:
                                print(active)
                                # Get scaled offsets to check for neighbours
                                offsets_s = [neutral] + [torch.relu(offset * side_len - 1) for offset in offsets]
                                coords = [query + offset for offset in offsets_s]
                                acts = [self._oj_model.inference(volume.to(device), coord.to(device)) for coord in coords]
                                print(torch.sum(torch.stack(acts)))
                                masks = [act == 1 for act in acts]
                                sum_masks = functools.reduce(lambda a,b : a & b, masks)
                                sum_masks_inv = functools.reduce(lambda a,b : a.logical_not() & b.logical_not(), masks)
                                next_query_mask = torch.squeeze((sum_masks | sum_masks_inv).logical_not())
                                mask_full = torch.squeeze(sum_masks)
                                query_to_write = query[mask_full]
                                query_to_write[:,0] += vol_x
                                query_to_write[:,1] += vol_y
                                query_to_write[:,2] += vol_z

                                for i in range(side_len):
                                    for j in range(side_len):
                                        for k in range(side_len):
                                            to_write = np.append(to_write, (query_to_write + left * i + above * j + behind * k).cpu().numpy(), axis=0)

                                side_len = int(side_len/2)
                                query_next = query[next_query_mask]
                                query = torch.empty((0,3)).to(device)
                                
                                for offset in [neutral] + offsets:
                                    query = torch.cat((query, query_next + offset * side_len), dim=0)

                                active = query.shape[0]

                to_write = to_write.astype(np.short)
                print(to_write.shape, volume.shape)
                #to_write_labels = actual.cpu().numpy().astype(np.short)
                #print(to_write_labels.shape)
                with open('outfile_auto.obj','w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                            " " + "0.0" + " " + "0.0" + " " + "0.5  " + "\n")
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")           
        
                vol = volume_in[0]
                maximum = np.max(vol)
                vol = vol/maximum
                vol[vol - 0.17 < 0] = 0

                with open('outfile_org.obj','w') as f:
                    for i in range(vol.shape[1]):
                        for j in range(vol.shape[2]):
                            for k in range(vol.shape[3]):
                                color = vol[0][i][j][k]
                                if color == 0:
                                    continue
                                f.write("v " + " " + str(i) + " " + str(j) + " " + str(k) + 
                                        " " + str(color.item()) + " " + str(0.5) + " " + str(0.5) + "\n")
                            #Corners of volume
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")  
                """
                with open('outfile_auto_labels.obj','w') as f:
                    for line in to_write_labels:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + 
                        " " + "0.0" + " " + "1.0" + " " + "0.0" + "\n")
                            #Corners of volume
                    f.write("v " + " " + "0"+ " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + "0" + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + "0"+ " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")

                    f.write("v " + " " + str(volume.shape[2] * side_len_clean)+  " " + "0" + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                    
                    f.write("v " + " " + str(volume.shape[2] * side_len_clean) +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean)+ 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")
                
                    f.write("v " + " " + "0" +  " " + str(volume.shape[3] * side_len_clean) + " " + str(volume.shape[4] * side_len_clean) + 
                        " " + "1.0" + " " + "0.5" + " " + "0.5" + "\n")                          
                """
    # Validate, ignore grads
    def _val(self, loader_val, losses_val, npoints, side_len):
        with torch.no_grad():
            losses_val_batch = []
            cache_vol = []
            cache_labels = []

            for j, batch in enumerate(loader_val):
                cache_vol.append(batch[0])
                cache_labels.append(batch[1])

                if j % 8 == 7:
                    samples = []
                    for i in range(8):
                        samp = sample_low(cache_vol[i], cache_labels[i], test=False)
                        if samp is None:
                             continue
                        samples.append(samp)
                    batch_in = many_to_one_collate_fn_sample(samples)
                    volume, coords, labels = batch_in

                    self._oj_model.eval()
                    yhat = self._oj_model(volume.to(device), coords.to(device))

                    # Computes validation loss
                    loss_val_batch = self._oj_loss(yhat, labels.to(device)).item()
                    losses_val_batch.append(loss_val_batch)

                    cache_vol = []
                    cache_labels = []

            loss_val = np.mean(losses_val_batch)
            losses_val.append((loss_val))
            return loss_val


    def train(self, train_dataset, val_dataset, side_len, npoints, name, load=False, cache_size=1000, win_sampled_size=16):

        # Function vars ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        loss_best = float('inf')
        losses_train = []
        losses_val = []
        loader_train = DataLoader(dataset=train_dataset, batch_size=self._size_batch, num_workers=8, pin_memory=True, shuffle=True, collate_fn=self._collate_fn)
        loader_val = DataLoader(dataset=val_dataset, batch_size=self._size_batch, num_workers=8,pin_memory=True, shuffle=True, collate_fn=self._collate_fn)
        if load:
            self._oj_model.load_state_dict(torch.load( "model/"+ type(self._oj_model).__name__ + "_" + str(device) + name + ".pt"))
            self._oj_optimizer.load_state_dict(torch.load( "optimizer/"+ type(self._oj_model).__name__ + "_" + str(device)+ name + ".pt"))

        # Auxiliary functions ++++++++++++++++++++++++++++++++++++++++++++++++++
        # Make a training step
        def _step_train(batch):
            volume, coords, labels = batch
            self._oj_model.train()
            yhat = self._oj_model(volume.to(device), coords.to(device))
            loss_train = self._oj_loss(yhat, labels.to(device))
            coords_reshaped = coords.reshape(win_sampled_size, -1, 3)
            labels_reshaped = labels.reshape(win_sampled_size, -1, 1)
            yhat_reshaped = yhat.reshape(win_sampled_size, -1, 1)
            max_diff = []
            min_diff = []
            for i in range(win_sampled_size):
                try:
                    max_diff.append(
                        torch.abs(torch.max(coords_reshaped[i][torch.squeeze(yhat_reshaped[i] > 0.5)], dim=0)[0] - 
                        torch.max(coords_reshaped[i][torch.squeeze(labels_reshaped[i] == 1.0)], dim=0)[0]))
                    min_diff.append(torch.abs(
                        torch.min(coords_reshaped[i][torch.squeeze(labels_reshaped[i] == 1.0)], dim=0)[0] -
                        torch.min(coords_reshaped[i][torch.squeeze(yhat_reshaped[i] > 0.5)], dim=0)[0]))
                except:
                    pass
            try:
                print("Max Diff Mean", torch.mean(torch.stack(max_diff), dim=0))
                print("Min Diff Mean", torch.mean(torch.stack(min_diff), dim=0))
                print("Max Diff", torch.max(torch.stack(max_diff), dim=0)[0])
                print("Min Diff", torch.max(torch.stack(min_diff), dim=0)[0])
            except:
                pass
            loss_train.backward()
            self._oj_optimizer.step()
            self._oj_optimizer.zero_grad()
        
            return loss_train.item(), self.mse(yhat, labels.to(device)).item()


        # Logic ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for _ in range(self._size_iter):

            losses_train_batch = []

            cache_vol = []
            cache_labels = []

            for i_, batch in enumerate(loader_train):
                cache_vol.append(batch[0])
                cache_labels.append(batch[1])

                if len(cache_vol) > cache_size:
                    del cache_vol[0]
                    del cache_labels[0]
                    
                # One step of training
                for j in range(1):
                    indices = np.random.choice(len(cache_vol), win_sampled_size)
                    samples = []
                    for i in indices:
                        samp = sample_low(cache_vol[i], cache_labels[i], npoints=npoints, test=False)
                        if samp is None:
                             continue
                        samples.append(samp)
                    batch_in = many_to_one_collate_fn_sample(samples)
                    loss_train_batch, mse_train_batch = _step_train(batch_in)
                    print("Training Loss Batch", i_, loss_train_batch, mse_train_batch, flush=True)

                if i_ % self._size_print_every == self._size_print_every-1:
                    loss_val = self._val(loader_val, losses_val, npoints=npoints, side_len=side_len)
                    print("Validation Loss", loss_val)

                    if loss_val < loss_best:
                        loss_best = loss_val
                        torch.save(self._oj_model.state_dict(), "model/"+ type(self._oj_model).__name__ + "_" + str(device) + name + ".pt")
                        torch.save(self._oj_optimizer.state_dict(), "optimizer/"+ type(self._oj_model).__name__ + "_" + str(device) + name + ".pt")
            loss_train = np.mean(losses_train_batch)
            losses_train.append(loss_train)
            print("Training Loss Iteration", loss_train,flush=True)


class Res_Auto_3d_Model_Occu_Parallel(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Occu_Parallel,self).__init__()

        self.model = nn.DataParallel(Res_Auto_3d_Model_Occu())

    def forward(self, volume, coords):
        return self.model(volume, coords)

    def bounding_box(self, volume, side_len):
        x = torch.arange(0, volume.shape[1])
        y = torch.arange(0, volume.shape[2])
        z = torch.arange(0, volume.shape[3])
        x,y,z = torch.meshgrid(x,y,z)
        query = torch.cat((torch.unsqueeze(x.reshape(-1), dim=1), torch.unsqueeze(y.reshape(-1), dim=1),  torch.unsqueeze(z.reshape(-1), dim=1)), dim=1)
        query = query.float().to(device) * side_len
        volume = torch.unsqueeze(volume, dim=1).float()
        mask = torch.squeeze(self.inference(volume, query) == 1)
        hits = query[mask]

        maxes = torch.max(hits, dim=0)[0].int()
        mines = torch.min(hits, dim=0)[0].int()
        return mines[0].item(), maxes[0].item(), mines[1].item(), maxes[1].item(), mines[2].item(), maxes[2].item()

    def inference(self, volume, coords):
        return (torch.sign(self.forward(volume, coords) - 0.9) + 1) / 2

class Res_Auto_3d_Model_Occu(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Occu,self).__init__()

        self.encode = nn.Sequential(layers.Res_Block_Down_3D(1, 64, 3, 1, nn.SELU(), False),
                                    layers.Res_Block_Down_3D(64, 64, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(64, 64, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(64, 64, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(64, 3, 3, 1, nn.SELU(), False))

        self.decode = nn.Sequential(layers.Res_Block_Up_Flat(1620 + 3, 512, nn.SELU()),
                                    layers.Res_Block_Up_Flat(512, 512, nn.SELU()),
                                    layers.Res_Block_Up_Flat(512, 1, nn.Sigmoid()))

    def forward(self, volume, coords):
        out = self.encode(volume)
        out = out.view(out.shape[0],-1)
        out = self.decode(torch.cat((torch.repeat_interleave(out, int(coords.shape[0]/volume.shape[0]), dim=0), coords), dim=1))
        return out
    
