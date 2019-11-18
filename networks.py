import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms

import layers
from data_interface import sample, bounding_box
from config import device

import sparseconvnet as scn


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Base Network Class
class Network_Generator():
    def __init__(self, rate_learn, size_batch, size_iter, size_print_every, oj_loss, optimizer, oj_model, collate_fn=None):

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._rate_learning = rate_learn
        self._size_batch = size_batch
        self._size_iter = size_iter
        self._size_print_every = size_print_every

        # (Function) Objects +++++++++++++++++++++++++++++++++++++++++++++++++++
        self._oj_model = oj_model
        self._oj_optimizer = optimizer(self._oj_model.parameters(), lr = self._rate_learning)
        self._oj_loss = oj_loss
        self._collate_fn = collate_fn


    def test(self, test_dataset):
        losses_test = [] # not needed TODO
        loader_test = DataLoader(dataset=test_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        self._oj_model.load_state_dict(torch.load( "model/"+ type(self._oj_model).__name__ + "_" + str(device) + ".pt"))

        with torch.no_grad():
            losses_val_batch = []
            for batch in loader_test:
                self._oj_model.eval()
                # Makes predictions
                feats, coors = batch
                coors_de, labels_de = sample(2**9, self._size_batch, coors)
                yhat = self._oj_model((feats, coors, coors_de.float()))


                hits = torch.squeeze(torch.round(yhat))
                print(yhat, torch.sum(yhat))
                locs = coors_de[hits == 1]
                print(locs.shape)
                to_write = locs.cpu().numpy().astype(np.short)
                with open('outfile_auto.obj','w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n")

                locs = coors[coors[:, 3] == 0]
                to_write = locs.cpu().numpy().astype(np.short)
                with open('outfile_org.obj','w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n")

                locs = coors[coors[:, 3] == 1]
                to_write = locs.cpu().numpy().astype(np.short)
                with open('outfile_org1.obj','w') as f:
                    for line in to_write:
                        f.write("v " + " " + str(line[0]) + " " + str(line[1]) + " " + str(line[2]) + "\n")

                loss_val_batch = self._oj_loss(yhat, labels_de).item()
                losses_val_batch.append(loss_val_batch)

        return loss_val_batch

    # Validate, ignore grads
    def _val(self, loader_val, losses_val):
        with torch.no_grad():
            losses_val_batch = []
            for batch in loader_val:

                self._oj_model.eval()
                # Makes predictions
                feats, coors = batch
                coors_de, labels_de = sample(2**9, self._size_batch, coors)
                yhat = self._oj_model((feats, coors, coors_de.float()))

                # Computes validation loss
                loss_val_batch = self._oj_loss(yhat, labels_de).item()
                losses_val_batch.append(loss_val_batch)

            loss_val = np.mean(losses_val_batch)
            losses_val.append((loss_val))
            return loss_val


    def train(self, train_dataset, val_dataset, load=False):

        # Function vars ++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        loss_best = float('inf')
        losses_train = []
        losses_val = []
        loader_train = DataLoader(dataset=train_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        loader_val = DataLoader(dataset=val_dataset, batch_size=self._size_batch, pin_memory=False, shuffle=True, collate_fn=self._collate_fn)
        if load:
            self._oj_model.load_state_dict(torch.load( "model/"+ type(self._oj_model).__name__ + "_" + str(device) + ".pt"))

        # Auxiliary functions ++++++++++++++++++++++++++++++++++++++++++++++++++
        # Make a training step
        def _step_train(batch):
            feats, coors = batch
            self._oj_model.train()
            coors_de, labels_de = sample(2**9, self._size_batch, coors)
            yhat = self._oj_model((feats, coors, coors_de.float()))
            loss_train = self._oj_loss(yhat, labels_de)
            loss_train.backward()
            self._oj_optimizer.step()
            self._oj_optimizer.zero_grad()
            return loss_train.item()


        # Logic ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for _ in range(self._size_iter):

            losses_train_batch = []
            for i, batch in enumerate(loader_train):

                # One step of training
                loss_train_batch = _step_train(batch)
                print("Training Loss Batch", i, loss_train_batch,flush=True)

                if i % self._size_print_every == self._size_print_every-1:
                    loss_val = self._val(loader_val, losses_val)
                    print("Validation Loss", loss_val)

                    if loss_val < loss_best:
                        loss_best = loss_val
                        torch.save(self._oj_model.state_dict(), "model/"+ type(self._oj_model).__name__ + "_" + str(device) + ".pt")

            loss_train = np.mean(losses_train_batch)
            losses_train.append(loss_train)
            print("Training Loss Iteration", loss_train,flush=True)


class Res_Auto_2d_Model(nn.Module):
    def __init__(self):
        super(Res_Auto_2d_Model,self).__init__()

        self.encode = nn.Sequential(layers.Res_Block_Down_2D(1, 4, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_2D(4, 4, 3, 1, nn.SELU(), True))
        self.decode = nn.Sequential(layers.Res_Block_Up_2D(4, 4, 3, 1 , nn.SELU()),
                                    layers.Res_Block_Up_2D(4, 1, 3, 1 , nn.SELU()))

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)

        return out

class Res_Auto_3d_Model(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model,self).__init__()
        self.encode = nn.Sequential(layers.Res_Block_Down_3D(1, 1, 3, 1, ME.MinkowskiSELU(), True),

                                    layers.Res_Block_Down_3D(1, 1, 3, 1, ME.MinkowskiSELU(), True))
        self.decode = nn.Sequential(layers.Res_Block_Up_3D(1, 1, 3, 1 , ME.MinkowskiSELU()),
                                    layers.Res_Block_Up_3D(1, 1, 3, 1 , ME.Minkowski.SELU()))

    def forward(self, x):
        out = self.encode(x)
        out = self.decode(out)

        return out

class Res_Auto_3d_Model_Sparse_Occu(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Sparse_Occu,self).__init__()

        self.input = scn.Sequential().add(scn.InputLayer(3, torch.LongTensor([32, 32, 32]), mode=3))
        self.encode = scn.Sequential().add(
                                    layers.Res_Block_Down_3D_Sparse(1, 16, 3, 1, scn.SELU(), True)).add(
                                    layers.Res_Block_Down_3D_Sparse(16, 32, 3, 1, scn.SELU(), True)).add(
                                    layers.Res_Block_Down_3D_Sparse(32, 32, 3, 1, scn.SELU(), True)).add(
                                    layers.Res_Block_Down_3D_Sparse(32, 4, 3, 1, scn.SELU(), True)).add(
                                    layers.Res_Block_Down_3D_Sparse(4, 4, 3, 1, scn.SELU(), True)).add(
                                    layers.Res_Block_Down_3D_Sparse(4, 1, 3, 1, scn.SELU(), True))


        self.bottleneck = scn.Sequential().add(layers.Bottleneck_Sparse_to_Dense(4, 1, 64, nn.SELU()))

        self.decode = nn.Sequential(layers.Res_Block_Up_Flat(64 + 3, 128, nn.SELU()),
                                    layers.Res_Block_Up_Flat(128, 256, nn.SELU()),
                                    layers.Res_Block_Up_Flat(256, 128, nn.SELU()),
                                    layers.Res_Block_Up_Flat(128, 1, nn.Sigmoid()))

    def forward(self, x):
        feats, coors, coors_de = x
        out = self.input([coors.to(device), feats.to(device)])
        out = self.encode(out)
        out = self.bottleneck(out)
        out = self.decode(torch.cat((torch.repeat_interleave(out, 2**9, dim=0), coors_de), dim=1))
        print(torch.sum(out))
        return out

class Res_Auto_3d_Model_Occu(nn.Module):
    def __init__(self):
        super(Res_Auto_3d_Model_Occu,self).__init__()

        self.encode = nn.Sequential(
                                    layers.Res_Block_Down_3D(1, 1, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(1, 4, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(4, 4, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(4, 1, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(1, 1, 3, 1, nn.SELU(), True),
                                    layers.Res_Block_Down_3D(1, 1, 3, 1, nn.SELU(), True))


        self.bottleneck = scn.Sequential().add(layers.Bottleneck_Sparse_to_Dense(4, 1, 64, nn.SELU()))

        self.decode = nn.Sequential(layers.Res_Block_Up_Flat(64 + 3, 128, nn.SELU()),
                                    layers.Res_Block_Up_Flat(128, 256, nn.SELU()),
                                    layers.Res_Block_Up_Flat(256, 128, nn.SELU()),
                                    layers.Res_Block_Up_Flat(128, 1, nn.Sigmoid()))

    def forward(self, x):
        feats, coors, coors_de = x
        coors = coors.long()
        inp = torch.zeros((2**4, 1, 256, 256, 256)).to(device)
        inp[coors[:,3],0, coors[:,0], coors[:,1], coors[:,2]] = 1
        out = self.encode(inp)
        out = out.view(out.shape[0],-1)
        out = self.decode(torch.cat((torch.repeat_interleave(out, 2**9, dim=0), coors_de), dim=1))
        print(torch.sum(out))
        return out
