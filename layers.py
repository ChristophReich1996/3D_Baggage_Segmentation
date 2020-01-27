import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np
from config import device
from pykdtree.kdtree import KDTree

# 3D +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Res_Block_Down_3D(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, size_filter, size_stride, fn_act, pool_avg):
        super(Res_Block_Down_3D, self).__init__()

        # Params +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        self._pool_avg = pool_avg
        self._size_in_channels = size_in_channels
        self._size_out_channels = size_out_channels

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_conv1 = nn.Conv3d(size_in_channels, size_out_channels, size_filter, size_stride, padding=(
            int(size_filter/2), int(size_filter/2), int(size_filter/2)))
        self.layer_norm1 = nn.BatchNorm3d(size_out_channels)

        self.fn_act = fn_act
        self.fn_identity = nn.Identity()

        self.layer_conv2 = nn.Conv3d(size_out_channels, size_out_channels, size_filter, size_stride, padding=(
            int(size_filter/2), int(size_filter/2), int(size_filter/2)))
        self.layer_norm2 = nn.BatchNorm3d(size_out_channels)

        if self._pool_avg:
            # TODO: Automatic dimension caluclation
            self.layer_pool = nn.AvgPool3d((2, 2, 2), stride=2)

    def forward(self, x):
        identity = self.fn_identity(x)

        out = self.layer_conv1(x)
        out = self.layer_norm1(out)
        out = self.fn_act(out)
        out = self.layer_conv2(out)
        out = self.layer_norm2(out)

        identity = F.pad(identity, (0, 0, 0, 0, 0, 0, 0, abs(
            self._size_in_channels - self._size_out_channels)))

        #out += identity
        out = self.fn_act(out)

        if self._pool_avg:
            out = self.layer_pool(out)
        return out


class Res_Block_Up_Flat(nn.Module):
    def __init__(self, size_in_channels, size_out_channels, fn_act):
        super(Res_Block_Up_Flat, self).__init__()

        # Nodes ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        self.layer_flat1 = nn.Linear(size_in_channels, size_out_channels)
        self.layer_norm1 = nn.BatchNorm1d(size_out_channels)

        self.fn_act = fn_act
        if size_in_channels != size_out_channels:
            self.fn_identity = nn.Linear(
                size_in_channels, size_out_channels, bias=False)
        else:
            self.fn_identity = nn.Identity()

        self.layer_flat2 = nn.Linear(size_out_channels, size_out_channels)
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
            self.fn_identity = nn.Linear(
                size_in_channels, size_out_channels, bias=False)
        else:
            self.fn_identity = nn.Identity()

        self.layer_flat2 = nn.Linear(size_out_channels, size_out_channels)
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


# Scores ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class DiceLoss(nn.Module):
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def forward(self, y_true, y_pred):
        y_true = y_true.reshape(self.batch_size, -1, 1)
        y_pred = y_pred.reshape(self.batch_size, -1, 1)
        y_true = torch.squeeze(y_true)
        y_pred = torch.squeeze(y_pred)
        numerator = 2 * torch.sum(y_true * y_pred, dim=1)
        denominator = torch.sum(y_true + y_pred, dim=1)
        quot = 1 - numerator / denominator

        invert_index = torch.sum(y_true, dim=1) == 0
        quot[invert_index] = torch.sum(
            y_pred[invert_index], dim=1)/y_true.shape[1]
        return torch.mean(quot)


class FocalLoss(nn.Module):
    # See Kaggle
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def IOU(coords, yhat, labels, batch_size, threshold=0.7):
    coords = coords.reshape(batch_size, -1, 3)
    yhat = yhat.reshape(batch_size, -1, 1)
    labels = labels.reshape(batch_size, -1, 1)

    iou_batch = []
    for i in range(batch_size):
        yhat_i = coords[i][torch.squeeze(yhat[i] >= threshold)].cpu().numpy()
        labels_i = coords[i][torch.squeeze(labels[i] == 1)].cpu().numpy()
        if labels_i.shape[0] > 0:
            kd_tree = KDTree(labels_i, leafsize=16)
            dist, _ = kd_tree.query(yhat_i, k=1)
            hits = (dist == 0)
            hits_sum = np.sum(hits)
            iou_batch.append(
                hits_sum / (yhat_i.shape[0] + labels_i.shape[0] - hits_sum))
        else:
            iou_batch.append(
                1 - (yhat_i.shape[0] / (yhat_i.shape[0] + coords[i].shape[0] - yhat_i.shape[0])))

    return np.mean(np.array(iou_batch))
