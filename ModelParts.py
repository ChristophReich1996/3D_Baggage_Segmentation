import torch
import torch.nn as nn
import torch.nn.functional as F

import Misc


class VolumeDecoderBlock(nn.Module):
    '''
    Basic Volume Decoder Block
    '''

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, activation: str = 'selu', downsampling: str = 'avarage pool',
                 downsampling_factor: int = 2, normalization: str = 'batchnorm') -> None:
        """
        Constructor method
        :param input_channels: (int) Number of input channels
        :param output_channels: (int) Number of output channels
        :param kernel_size: (int) Filter size of convolution
        :param stride: (int) Stride factor of convolution
        :param padding: (int) Padding used in every convolution
        :param activation: (str) Type of activation function
        :param downsampling: (str) Type of downsampling operation
        :param downsampling_factor: (int) Downsampling factor to use
        :param normalization: (str) Type of normalization operation used
        """
        # Call super constructor
        super(VolumeDecoderBlock, self).__init__()
        # Init activations
        self.activation_1 = Misc.get_activation(activation=activation)
        self.activation_2 = Misc.get_activation(activation=activation)
        # Init normalizations
        self.normalization_1 = Misc.get_normalization_3d(normalization=normalization, channels=output_channels)
        self.normalization_2 = Misc.get_normalization_3d(normalization=normalization, channels=output_channels)
        # Init convolutions
        self.convolution_1 = nn.Conv3d(in_channels=input_channels, out_channels=output_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        self.convolution_2 = nn.Conv3d(in_channels=output_channels, out_channels=output_channels,
                                       kernel_size=kernel_size, stride=stride, padding=padding)
        # Init downsampling operation

