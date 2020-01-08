import torch
import torch.nn as nn
import torch.nn.functional as F

import Misc


class VolumeDecoderBlock(nn.Module):
    """
    Basic Volume Decoder Block
    """

    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 1, activation: str = 'selu', downsampling: str = 'avarage pool',
                 downsampling_factor: int = 2, normalization: str = 'batchnorm', dropout_rate: float = 0.0) -> None:
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
        :param dropout_rate: (float) Dropout rate to perform after every stage
        """
        # Call super constructor
        super(VolumeDecoderBlock, self).__init__()
        # Save dropout rate
        self.dropout_rate = dropout_rate
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
        self.donwsampling = Misc.get_downsampling_3d(downsampling=downsampling, factor=downsampling_factor,
                                                     channels=output_channels)

    def forward(self, input: torch.tensor) -> torch.tensor:
        """
        Forward pass of the basic volume decoder block
        :param input: (torch.tensor) Input volume with shape (batch size, channels_in, x_in, y_in, z_in)
        :return: (torch.tensor) Output tensor with shape (batch size, channels_out, x_out, y_out, z_out)
        """
        # First stage
        output_convolution_1 = self.convolution_1(input)
        output_normalization_1 = self.normalization_1(output_convolution_1)
        output_activation_1 = self.activation_1(output_normalization_1)
        if self.dropout_rate > 0.0:
            output_activation_1 = F.dropout(output_activation_1, p=self.dropout_rate)
        # Second stage
        output_convolution_2 = self.convolution_2(output_activation_1)
        output_normalization_2 = self.normalization_2(output_convolution_2)
        output_activation_2 = self.activation_2(output_normalization_2)
        if self.dropout_rate > 0.0:
            output_activation_2 = F.dropout(output_activation_2, p=self.dropout_rate)
        # Downsampling stage
        output_downsampling = self.donwsampling(output_activation_2)
        return output_downsampling
