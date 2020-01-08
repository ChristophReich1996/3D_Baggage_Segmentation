import torch
import torchsummary
import numpy as np
import matplotlib.pyplot as plt


import Model
import Dataset
import ModelWrapper

if __name__ == '__main__':
    ONet = Model.OccupancyNetwork()
    torchsummary.summary(ONet, input_size=[(1, 64, 64, 64), (1, 183)], device='cpu')