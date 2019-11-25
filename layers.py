import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from config import device


# 3D +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class Res_Block_Down_3D(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, size_filter, size_stride, fn_act, pool_avg):
        super(Res_Block_Down_3D, self).__init__()

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._pool_avg = pool_avg
        self._size_in_channels = size_in_channels
        self._size_out_channels = size_out_channels

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_conv1 = nn.Conv3d(size_in_channels,size_out_channels, size_filter, size_stride, padding=(int(size_filter/2),int(size_filter/2), int(size_filter/2)))
        self.layer_norm1 = nn.BatchNorm3d(size_out_channels)

        self.fn_act = fn_act
        self.fn_identity = nn.Identity()

        self.layer_conv2= nn.Conv3d(size_out_channels,size_out_channels, size_filter, size_stride, padding=(int(size_filter/2),int(size_filter/2), int(size_filter/2)))
        self.layer_norm2 = nn.BatchNorm3d(size_out_channels)

        if self._pool_avg:
            # TODO: Automatic dimension caluclation
            self.layer_pool = nn.AvgPool3d((2,2,2),stride=2)


    def forward(self, x):
        identity = self.fn_identity(x)

        out = self.layer_conv1(x)
        out = self.layer_norm1(out)
        out = self.fn_act(out)
        out = self.layer_conv2(out)
        out = self.layer_norm2(out)

        identity = F.pad(identity, (0, 0, 0, 0, 0, 0, 0, abs(self._size_in_channels - self._size_out_channels)))

        #out += identity
        out = self.fn_act(out)

        if self._pool_avg:
            out = self.layer_pool(out)
        return out

def residual(nIn, nOut):
    if nIn != nOut:
        return scn.NetworkInNetwork(nIn, nOut, False)
    else:
        return scn.Identity()

class Res_Block_Down_3D_Sparse(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, size_filter, size_stride, fn_act, pool_avg):
        super(Res_Block_Down_3D_Sparse, self).__init__()

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._pool_avg = pool_avg
        self._size_in_channels = size_in_channels
        self._size_out_channels = size_out_channels

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_conv1 = scn.SubmanifoldConvolution(3, size_in_channels,size_out_channels, size_filter, True)
        self.layer_norm1 = scn.BatchNormalization(size_out_channels)

        self.fn_act = fn_act

        self.layer_conv2 = scn.SubmanifoldConvolution(3, size_out_channels,size_out_channels, size_filter, True)
        self.layer_norm2 = scn.BatchNormalization(size_out_channels)

        self.x = scn.Sequential().add(self.layer_conv1).add(self.layer_norm1).add(
                                      self.fn_act).add(
                                      self.layer_conv2).add(self.layer_norm2)

        self.block = scn.ConcatTable().add(
            self.x).add(
            residual(size_in_channels, size_out_channels))

        self.add_res = scn.AddTable()


        if self._pool_avg:
            # TODO: Automatic dimension caluclation
            self.layer_pool = scn.AveragePooling(3, (2,2,2), 2)


    def forward(self, x):
        out = self.block(x)
        out = self.add_res(out)

        if self._pool_avg:
            out = self.layer_pool(out)
        return out

class Bottleneck_Sparse_to_Dense(nn.Module):
        def __init__(self, cube_size, size_in_channels, size_out_channels, fn_act, dimension=3):
            super(Bottleneck_Sparse_to_Dense, self).__init__()
            self.dense = scn.SparseToDense(dimension,size_in_channels)
            self.linear = nn.Linear(size_in_channels * cube_size**3, size_out_channels)
            self.fn_act = fn_act

        def forward(self, x):
            out = self.dense(x)
            out = out.view(out.shape[0],-1)
            return self.fn_act(self.linear(out))

class Res_Block_Up_Flat(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, fn_act):
        super(Res_Block_Up_Flat, self).__init__()

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_flat1 = nn.Linear(size_in_channels, size_out_channels)
        self.layer_norm1 = nn.BatchNorm1d(size_out_channels)

        self.fn_act = fn_act
        if size_in_channels != size_out_channels:
            self.fn_identity = nn.Linear(size_in_channels, size_out_channels, bias=False)
        else:
            self.fn_identity = nn.Identity()


        self.layer_flat2= nn.Linear(size_out_channels, size_out_channels)
        self.layer_norm2 = nn.BatchNorm1d(size_out_channels)



    def forward(self, x):
        identity = self.fn_identity(x)

        out = self.layer_flat1(x)
        out = self.layer_norm1(out)
        out = self.fn_act(out)
        out = self.layer_flat2(out)
        out = self.layer_norm2(out)

        out += identity
        out = self.fn_act(out)


        return out

class Res_Block_Up_1D(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, fn_act):
        super(Res_Block_Up_1D, self).__init__()

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_flat1 = nn.Linear(size_in_channels, size_out_channels)
        self.layer_norm1 = nn.BatchNorm1d(size_out_channels)

        self.fn_act = fn_act
        if size_in_channels != size_out_channels:
            self.fn_identity = nn.Linear(size_in_channels, size_out_channels, bias=False)
        else:
            self.fn_identity = nn.Identity()


        self.layer_flat2= nn.Linear(size_out_channels, size_out_channels)
        self.layer_norm2 = nn.BatchNorm1d(size_out_channels)



    def forward(self, x):
        identity = self.fn_identity(x)

        out = self.layer_flat1(x)
        out = self.layer_norm1(out)
        out = self.fn_act(out)
        out = self.layer_flat2(out)
        out = self.layer_norm2(out)

        out += identity
        out = self.fn_act(out)


        return out


# Utilities ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class BinaryFunction(Function):
    @staticmethod
    def forward(ctx, input):
        #ctx.save_for_backward(input)
        return torch.sign(input)

    @staticmethod
    def backward(ctx, grad_output):
        #input, = ctx.saved_tensors
        #grad_output[input>1]=0
        #grad_output[input<-1]=0
        return grad_output


class Binary(nn.Module):
    def __init__(self):
        super(Binary, self).__init__()
        self.fn = BinaryFunction.apply

    def forward(self, x):
        return (self.fn(x) + 1) / 2

class ScnBinary(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ScnBinary, self).__init__()
        self.fn = BinaryFunction.apply

    def forward(self, input):
        output = scn.SparseConvNetTensor()
        output.features = self.fn(input.features)
        output.metadata = input.metadata
        output.spatial_size = input.spatial_size
        return output

    def __repr__(self):
        return self.__class__.__name__ + '()'


def dice_loss(input, target):
    smoothness = 1.

    i_ = input.reshape(-1)
    t_ = target.reshape(-1)
    intersection = (i_ * t_).sum()

    return -1 * ((2. * intersection + smoothness) / (i_.sum() + t_.sum() + smoothness))
